#[cfg(test)]
mod test;

use crate::ai::model::{Conversation, ConversationSummary};
use crate::{
    ai::model::{AiFlags, AiTool, ChatState},
    ai::service::AiService,
    Error,
};
use actix_http::header;
use actix_web::{delete, get, post, put, web, HttpResponse, Responder};
use itertools::Itertools;
use trustify_auth::authenticator::user::UserDetails;
use trustify_auth::{authorizer::Require, Ai};
use trustify_common::db::query::Query;
use trustify_common::db::Database;
use trustify_common::model::{Paginated, PaginatedResults};
use uuid::Uuid;

pub fn configure(config: &mut utoipa_actix_web::service_config::ServiceConfig, db: Database) {
    let service = AiService::new(db.clone());
    config
        .app_data(web::Data::new(service))
        .service(completions)
        .service(flags)
        .service(tools)
        .service(tool_call)
        .service(create_conversation)
        .service(update_conversation)
        .service(list_conversations)
        .service(get_conversation)
        .service(delete_conversation);
}

#[utoipa::path(
    tag = "ai",
    operation_id = "completions",
    request_body = ChatState,
    responses(
        (status = 200, description = "The resulting completion", body = ChatState),
        (status = 400, description = "The request was invalid"),
        (status = 404, description = "The AI service is not enabled")
    )
)]
#[post("/v1/ai/completions")]
pub async fn completions(
    service: web::Data<AiService>,
    request: web::Json<ChatState>,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let response = service.completions(&request).await?;
    Ok(HttpResponse::Ok().json(response))
}

#[utoipa::path(
    tag = "ai",
    operation_id = "aiFlags",
    responses(
        (status = 200, description = "The resulting Flags", body = AiFlags),
        (status = 404, description = "The AI service is not enabled")
    )
)]
#[get("/v1/ai/flags")]
// Gets the flags for the AI service
pub async fn flags(
    service: web::Data<AiService>,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    Ok(HttpResponse::Ok().json(AiFlags {
        completions: service.completions_enabled(),
    }))
}

#[utoipa::path(
    tag = "ai",
    operation_id = "aiTools",
    responses(
        (status = 200, description = "The resulting list of tools", body = Vec<AiTool>),
        (status = 404, description = "The AI service is not enabled")
    )
)]
#[get("/v1/ai/tools")]
// Gets the list of tools that are available to assist AI services.
pub async fn tools(
    service: web::Data<AiService>,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let tools = &service
        .local_tools
        .iter()
        .map(|tool| AiTool {
            name: tool.name(),
            description: tool.description(),
            parameters: tool.parameters(),
        })
        .collect_vec();
    Ok(HttpResponse::Ok().json(tools))
}

#[utoipa::path(
    tag = "ai",
    operation_id = "aiToolCall",
    request_body = serde_json::Value,
    params(
        ("name", Path, description = "Name of the tool to call")
    ),
    responses(
        (status = 200, description = "The result of the tool call", body = String, content_type = "text/plain"),
        (status = 400, description = "The tool request was invalid"),
        (status = 404, description = "The tool was not found")
    )
)]
#[post("/v1/ai/tools/{name}")]
pub async fn tool_call(
    service: web::Data<AiService>,
    name: web::Path<String>,
    request: String,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let tool = service
        .local_tools
        .iter()
        .find(|tool| tool.name() == name.clone())
        .ok_or_else(|| actix_web::error::ErrorNotFound("Tool not found"))?;

    let result = tool
        .call(request.as_str())
        .await
        .map_err(|e| Error::BadRequest(e.to_string()))?;

    Ok(HttpResponse::Ok()
        .insert_header((header::CONTENT_TYPE, "text/plain"))
        .body(result))
}

#[utoipa::path(
    tag = "ai",
    operation_id = "createConversation",
    request_body = ChatState,
    responses(
        (status = 200, description = "The resulting conversation", body = Conversation),
        (status = 400, description = "The request was invalid"),
        (status = 404, description = "The AI service is not enabled")
    )
)]
#[post("/v1/ai/conversations")]
pub async fn create_conversation(
    service: web::Data<AiService>,
    db: web::Data<Database>,
    request: web::Json<ChatState>,
    user: UserDetails,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let user_id = user.id;

    // generate an assistant response
    let response = service.completions(&request).await?;

    // If summarizing the conversation takes a while, maybe we can figure out how to do it
    // in the background and update the record later.
    let summary = service.summarize(&response).await?;

    // store the new conversation
    let conversation = service
        .create_conversation(
            user_id.clone(),
            serde_json::to_value(&response).map_err(|e| Error::Internal(e.to_string()))?,
            summary,
            db.as_ref(),
        )
        .await?;

    let response = Conversation {
        id: conversation.id,
        state: response,
        updated_at: conversation.updated_at,
        seq: 0,
    };

    Ok(HttpResponse::Ok().json(response))
}

#[utoipa::path(
    tag = "ai",
    operation_id = "updateConversation",
    params(
        ("id", Path, description = "Opaque ID of the conversation")
    ),
    request_body = Conversation,
    responses(
        (status = 200, description = "The resulting conversation", body = Conversation),
        (status = 400, description = "The request was invalid"),
        (status = 404, description = "The AI service is not enabled or the conversation was not found")
    )
)]
#[put("/v1/ai/conversations/{id}")]
pub async fn update_conversation(
    service: web::Data<AiService>,
    db: web::Data<Database>,
    id: web::Path<Uuid>,
    user: UserDetails,
    request: web::Json<Conversation>,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let user_id = user.id;

    let conversation_id = id.into_inner();
    let conversation = service
        .fetch_conversation(conversation_id, db.as_ref())
        .await?;

    let response = match conversation {
        // the conversation_id might be invalid
        None => Err(Error::NotFound("conversation not found".to_string()))?,

        // Found the conversation
        Some(conversation) => {
            // verify that the conversation belongs to the user
            if conversation.user_id != user_id {
                // make this error look like a not found error to avoid leaking
                // existence of the conversation
                Err(Error::NotFound("conversation not found".to_string()))?;
            }

            // generate an assistant response
            let response = service.completions(&request.state).await?;

            // If summarizing the conversation takes a while, maybe we can figure out how to do it
            // in the background and update the record later.
            let summary = service.summarize(&response).await?;

            // update the conversation in the database
            let conversation = service
                .update_conversation(
                    conversation_id,
                    serde_json::to_value(&response).map_err(|e| Error::Internal(e.to_string()))?,
                    summary,
                    request.seq,
                    db.as_ref(),
                )
                .await?;

            Conversation {
                id: conversation.id,
                updated_at: conversation.updated_at,
                state: response,
                seq: request.seq,
            }
        }
    };

    Ok(HttpResponse::Ok().json(response))
}

#[utoipa::path(
    tag = "ai",
    operation_id = "listConversations",
    params(
        Query,
        Paginated,
    ),
    responses(
        (status = 200, description = "The resulting list of conversation summaries", body = PaginatedResults<ConversationSummary>),
        (status = 404, description = "The AI service is not enabled")
    )
)]
#[get("/v1/ai/conversations")]
// Gets the list of the user's previous conversations
pub async fn list_conversations(
    service: web::Data<AiService>,
    web::Query(search): web::Query<Query>,
    web::Query(paginated): web::Query<Paginated>,
    db: web::Data<Database>,
    user: UserDetails,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let user_id = user.id;

    let result = service
        .fetch_conversations(user_id, search, paginated, db.as_ref())
        .await?;

    let result = PaginatedResults {
        items: result
            .items
            .into_iter()
            .map(|c| ConversationSummary {
                id: c.id,
                summary: c.summary,
                updated_at: c.updated_at,
            })
            .collect(),
        total: result.total,
    };

    Ok(HttpResponse::Ok().json(result))
}

#[utoipa::path(
    tag = "ai",
    operation_id = "getConversation",
    params(
        ("id", Path, description = "Opaque ID of the conversation")
    ),
    responses(
        (status = 200, description = "The resulting conversation", body = Conversation),
        (status = 400, description = "The request was invalid"),
        (status = 404, description = "The AI service is not enabled or the conversation was not found")
    )
)]
#[get("/v1/ai/conversations/{id}")]
pub async fn get_conversation(
    service: web::Data<AiService>,
    db: web::Data<Database>,
    id: web::Path<Uuid>,
    user: UserDetails,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let user_id = user.id;

    let conversation = service
        .fetch_conversation(id.into_inner(), db.as_ref())
        .await?;

    match conversation {
        // the conversation_id might be invalid
        None => Err(Error::NotFound("conversation not found".to_string()))?,

        // Found the conversation
        Some(conversation) => {
            // verify that the conversation belongs to the user
            if conversation.user_id != user_id {
                // make this error look like a not found error to avoid leaking
                // existence of the conversation
                Err(Error::NotFound("conversation not found".to_string()))?;
            }

            Ok(HttpResponse::Ok().json(Conversation {
                id: conversation.id,
                updated_at: conversation.updated_at,
                state: serde_json::from_value(conversation.state)
                    .map_err(|e| Error::Internal(e.to_string()))?,
                seq: conversation.seq,
            }))
        }
    }
}

#[utoipa::path(
    tag = "ai",
    operation_id = "deleteConversation",
    params(
        ("id", Path, description = "Opaque ID of the conversation")
    ),
    responses(
        (status = 200, description = "The resulting conversation", body = Conversation),
        (status = 400, description = "The request was invalid"),
        (status = 404, description = "The AI service is not enabled or the conversation was not found")
    )
)]
#[delete("/v1/ai/conversations/{id}")]
pub async fn delete_conversation(
    service: web::Data<AiService>,
    db: web::Data<Database>,
    id: web::Path<Uuid>,
    user: UserDetails,
    _: Require<Ai>,
) -> actix_web::Result<impl Responder> {
    let user_id = user.id;
    let conversation_id = id.into_inner();

    let conversation = service
        .fetch_conversation(conversation_id, db.as_ref())
        .await?;

    match conversation {
        // the conversation_id might be invalid
        None => Err(Error::NotFound("conversation not found".to_string()))?,

        // Found the conversation
        Some(conversation) => {
            // verify that the conversation belongs to the user
            if conversation.user_id != user_id {
                // make this error look like a not found error to avoid leaking
                // existence of the conversation
                Err(Error::NotFound("conversation not found".to_string()))?;
            }

            let rows_affected = service
                .delete_conversation(conversation_id, db.as_ref())
                .await?;
            match rows_affected {
                0 => Ok(HttpResponse::NotFound().finish()),
                1 => Ok(HttpResponse::Ok().json(Conversation {
                    id: conversation.id,
                    updated_at: conversation.updated_at,
                    state: serde_json::from_value(conversation.state)
                        .map_err(|e| Error::Internal(e.to_string()))?,
                    seq: conversation.seq,
                })),
                _ => Err(Error::Internal("Unexpected number of rows affected".into()))?,
            }
        }
    }
}
