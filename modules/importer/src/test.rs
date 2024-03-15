#![cfg(test)]

use super::model::ImportConfiguration;
use actix_web::{
    http::{header, StatusCode},
    test, App,
};
use serde_json::json;
use trustify_common::db::Database;

#[actix_web::test]
async fn test_default() {
    let _ = env_logger::try_init();

    let db = Database::for_test("test_default").await.unwrap();
    let app =
        test::init_service(App::new().configure(|svc| super::endpoints::configure(svc, db))).await;

    // create one

    let req = test::TestRequest::post()
        .uri("/api/v1/importer/foo")
        .set_json(json!({"foo":"bar"}))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::CREATED);

    // now list all

    let req = test::TestRequest::get()
        .uri("/api/v1/importer")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let result: Vec<ImportConfiguration> = test::read_body_json(resp).await;
    assert_eq!(
        result,
        vec![ImportConfiguration {
            name: "foo".into(),
            configuration: json!({"foo":"bar"})
        }]
    );

    // update it

    let req = test::TestRequest::put()
        .uri("/api/v1/importer/foo")
        .set_json(json!({"foo":"baz"}))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // get it

    let req = test::TestRequest::get()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let result: ImportConfiguration = test::read_body_json(resp).await;
    assert_eq!(
        result,
        ImportConfiguration {
            name: "foo".into(),
            configuration: json!({"foo":"baz"})
        }
    );

    // delete it

    let req = test::TestRequest::delete()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // get none

    let req = test::TestRequest::get()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[actix_web::test]
async fn test_oplock() {
    let _ = env_logger::try_init();

    let db = Database::for_test("test_oplock").await.unwrap();
    let app =
        test::init_service(App::new().configure(|svc| super::endpoints::configure(svc, db))).await;

    // create one

    let req = test::TestRequest::post()
        .uri("/api/v1/importer/foo")
        .set_json(json!({"foo":"bar"}))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::CREATED);

    // update it (no lock)

    let req = test::TestRequest::put()
        .uri("/api/v1/importer/foo")
        .set_json(json!({"foo":"baz"}))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // get it

    let req = test::TestRequest::get()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let etag = resp.headers().get(header::ETAG);
    assert!(etag.is_some());
    let etag = etag.cloned().unwrap();

    let result: ImportConfiguration = test::read_body_json(resp).await;
    assert_eq!(
        result,
        ImportConfiguration {
            name: "foo".into(),
            configuration: json!({"foo":"baz"})
        }
    );

    // update it (with lock)

    let req = test::TestRequest::put()
        .uri("/api/v1/importer/foo")
        .set_json(json!({"foo":"buz"}))
        .append_header((header::IF_MATCH, etag.clone()))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // get it

    let req = test::TestRequest::get()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let result: ImportConfiguration = test::read_body_json(resp).await;
    assert_eq!(
        result,
        ImportConfiguration {
            name: "foo".into(),
            configuration: json!({"foo":"buz"})
        }
    );

    // update it (with broken lock)

    let req = test::TestRequest::put()
        .uri("/api/v1/importer/foo")
        .set_json(json!({"foo":"boz"}))
        .append_header((header::IF_MATCH, etag.clone()))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::PRECONDITION_FAILED);

    // update it (with wrong name)

    let req = test::TestRequest::put()
        .uri("/api/v1/importer/foo2")
        .set_json(json!({"foo":"boz"}))
        .append_header((header::IF_MATCH, etag.clone()))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // get it (must not change)

    let req = test::TestRequest::get()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let old_etag = etag;
    let etag = resp.headers().get(header::ETAG);
    assert!(etag.is_some());
    let etag = etag.cloned().unwrap();
    assert_ne!(old_etag, etag);

    let result: ImportConfiguration = test::read_body_json(resp).await;
    assert_eq!(
        result,
        ImportConfiguration {
            name: "foo".into(),
            configuration: json!({"foo":"buz"})
        }
    );

    // delete it (wrong lock)

    let req = test::TestRequest::delete()
        .uri("/api/v1/importer/foo")
        .append_header((header::IF_MATCH, old_etag.clone()))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // get it (must still be there)

    let req = test::TestRequest::get()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let result: ImportConfiguration = test::read_body_json(resp).await;
    assert_eq!(
        result,
        ImportConfiguration {
            name: "foo".into(),
            configuration: json!({"foo":"buz"})
        }
    );

    // delete it (correct lock)

    let req = test::TestRequest::delete()
        .uri("/api/v1/importer/foo")
        .append_header((header::IF_MATCH, etag.clone()))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // get none

    let req = test::TestRequest::get()
        .uri("/api/v1/importer/foo")
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
