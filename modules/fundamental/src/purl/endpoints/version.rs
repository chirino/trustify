use crate::purl::service::PurlService;
use crate::Error;
use actix_web::{get, web, HttpResponse, Responder};
use sea_orm::prelude::Uuid;
use std::str::FromStr;
use trustify_common::id::IdError;
use trustify_common::purl::Purl;

#[utoipa::path(
    tag = "purl",
    operation_id = "getVersionedPurl",
    context_path= "/api",
    params(
        ("key" = String, Path, description = "opaque ID identifier for a package version, or URL-ecnoded pURL itself")
    ),
    responses(
        (status = 200, description = "Details for the version of a PURL", body = VersionedPurlDetails),
    ),
)]
#[get("/v1/purl/version/{key}")]
pub async fn get_versioned_purl(
    service: web::Data<PurlService>,
    key: web::Path<String>,
) -> actix_web::Result<impl Responder> {
    if key.starts_with("pkg:") {
        let purl = Purl::from_str(&key).map_err(|e| Error::IdKey(IdError::Purl(e)))?;
        Ok(HttpResponse::Ok().json(service.versioned_purl_by_purl(&purl, ()).await?))
    } else {
        let uuid = Uuid::from_str(&key).map_err(|e| Error::IdKey(IdError::InvalidUuid(e)))?;
        Ok(HttpResponse::Ok().json(service.versioned_purl_by_uuid(&uuid, ()).await?))
    }
}
