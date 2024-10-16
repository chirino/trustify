use crate::purl::service::PurlService;
use crate::Error;
use actix_web::{get, web, HttpResponse, Responder};
use sea_orm::prelude::Uuid;
use std::str::FromStr;
use trustify_common::db::query::Query;
use trustify_common::id::IdError;
use trustify_common::model::Paginated;
use trustify_common::purl::Purl;

#[utoipa::path(
    context_path= "/api",
    tag = "purl",
    params(
        ("key" = String, Path, description = "opaque identifier for a base PURL, or a URL-encoded pURL itself")
    ),
    responses(
        (status = 200, description = "Details for the versionless base PURL", body = BasePurlDetails),
    ),
)]
#[get("/v1/purl/base/{key}")]
pub async fn get_base_purl(
    service: web::Data<PurlService>,
    key: web::Path<String>,
) -> actix_web::Result<impl Responder> {
    if key.starts_with("pkg:") {
        let purl = Purl::from_str(&key).map_err(|e| Error::IdKey(IdError::Purl(e)))?;
        Ok(HttpResponse::Ok().json(service.base_purl_by_purl(&purl, ()).await?))
    } else {
        let uuid = Uuid::from_str(&key).map_err(|e| Error::IdKey(IdError::InvalidUuid(e)))?;
        Ok(HttpResponse::Ok().json(service.base_purl_by_uuid(&uuid, ()).await?))
    }
}

#[utoipa::path(
    context_path= "/api",
    tag = "purl",
    params(
        Query,
        Paginated,
    ),
    responses(
        (status = 200, description = "All relevant matching versionless base PURL", body = PaginatedBasePurlSummary),
    ),
)]
#[get("/v1/purl/base")]
pub async fn all_base_purls(
    service: web::Data<PurlService>,
    web::Query(search): web::Query<Query>,
    web::Query(paginated): web::Query<Paginated>,
) -> actix_web::Result<impl Responder> {
    Ok(HttpResponse::Ok().json(service.base_purls(search, paginated, ()).await?))
}
