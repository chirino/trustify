use actix_http::Payload;
use actix_utils::future::{err, ok, Ready};
use actix_web::error::Error;
use actix_web::error::QueryPayloadError;
use actix_web::{FromRequest, HttpRequest};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use std::{fmt, ops, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpenapiQuery<T>(pub T);

impl<T> OpenapiQuery<T> {
    /// Unwrap into inner `T` value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T: DeserializeOwned> OpenapiQuery<T> {
    pub fn from_query(query_str: &str) -> Result<Self, serde_json::Error> {
        let mut doc = Map::new();
        for (k, v) in form_urlencoded::parse(query_str.as_bytes()) {

            let (k, is_array_key) = if k.ends_with("[]") {
                (k[..k.len() - 2].to_string(), true)
            } else {
                (k.to_string(), false)
            };

            match doc.entry(k) {
                // did the key exist in the doc?
                serde_json::map::Entry::Occupied(mut entry) => {
                    let value = entry.get_mut();
                    match value {
                        Value::Array(array) => {
                            array.push(Value::String(v.to_string()));
                        }
                        _ => {
                            // Field was previous set, move it to the first element of an array.
                            let mut array = Vec::new();
                            array.push(value.clone());
                            array.push(Value::String(v.to_string()));
                            *value = Value::Array(array);
                        }
                    }
                }
                serde_json::map::Entry::Vacant(entry) => {
                    if is_array_key {
                        let mut array = Vec::new();
                        array.push(Value::String(v.to_string()));
                        entry.insert(Value::Array(array));
                    } else {
                        entry.insert(Value::String(v.to_string()));
                    }
                }
            }
        }

        serde_json::from_value::<T>(Value::Object(doc)).map(Self)
    }
}

impl<T> ops::Deref for OpenapiQuery<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> ops::DerefMut for OpenapiQuery<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: fmt::Display> fmt::Display for OpenapiQuery<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// See [here](#Examples) for example of usage as an extractor.
impl<T: DeserializeOwned> FromRequest for OpenapiQuery<T> {
    type Error = Error;
    type Future = Ready<Result<Self, Error>>;

    #[inline]
    fn from_request(req: &HttpRequest, _: &mut Payload) -> Self::Future {
        let error_handler = req
            .app_data::<QueryConfig>()
            .and_then(|c| c.err_handler.clone());

        OpenapiQuery::from_query(req.query_string())
            .map(|val| ok(val))
            .unwrap_or_else(move |e| {
                log::debug!(
                    "Failed during Query extractor deserialization. \
                     Request path: {:?}",
                    req.path()
                );

                let e = if let Some(error_handler) = error_handler {
                    (error_handler)(
                        QueryPayloadError::Deserialize(serde::ser::Error::custom("test")),
                        req,
                    )
                } else {
                    e.into()
                };

                err(e)
            })
    }
}

#[derive(Clone, Default)]
pub struct QueryConfig {
    #[allow(clippy::type_complexity)]
    err_handler: Option<Arc<dyn Fn(QueryPayloadError, &HttpRequest) -> Error + Send + Sync>>,
}

impl QueryConfig {
    /// Set custom error handler
    pub fn error_handler<F>(mut self, f: F) -> Self
    where
        F: Fn(QueryPayloadError, &HttpRequest) -> Error + Send + Sync + 'static,
    {
        self.err_handler = Some(Arc::new(f));
        self
    }
}

#[cfg(test)]
mod tests {
    use actix_http::StatusCode;
    use serde::Deserialize;

    use super::*;
    use actix_web::{error::InternalError, test::TestRequest, HttpResponse};

    #[derive(Deserialize, Debug)]
    struct Id {
        id: String,
    }

    #[actix_rt::test]
    async fn test_service_request_extract() {
        let req = TestRequest::with_uri("/name/user1/").to_srv_request();
        assert!(OpenapiQuery::<Id>::from_query(req.query_string()).is_err());

        let req = TestRequest::with_uri("/name/user1/?id=test").to_srv_request();
        let mut s = OpenapiQuery::<Id>::from_query(req.query_string()).unwrap();

        assert_eq!(s.id, "test");
        assert_eq!(
            format!("{}, {:?}", s.id, s),
            "test, OpenapiQuery(Id { id: \"test\" })"
        );

        s.id = "test1".to_string();
        let s = s.into_inner();
        assert_eq!(s.id, "test1");
    }

    #[actix_rt::test]
    async fn test_request_extract() {
        let req = TestRequest::with_uri("/name/user1/").to_srv_request();
        let (req, mut pl) = req.into_parts();
        assert!(OpenapiQuery::<Id>::from_request(&req, &mut pl)
            .await
            .is_err());

        let req = TestRequest::with_uri("/name/user1/?id=test").to_srv_request();
        let (req, mut pl) = req.into_parts();

        let mut s = OpenapiQuery::<Id>::from_request(&req, &mut pl)
            .await
            .unwrap();
        assert_eq!(s.id, "test");
        assert_eq!(
            format!("{}, {:?}", s.id, s),
            "test, OpenapiQuery(Id { id: \"test\" })"
        );

        s.id = "test1".to_string();
        let s = s.into_inner();
        assert_eq!(s.id, "test1");
    }

    #[actix_rt::test]
    #[should_panic]
    async fn test_tuple_panic() {
        let req = TestRequest::with_uri("/?one=1&two=2").to_srv_request();
        let (req, mut pl) = req.into_parts();

        OpenapiQuery::<(u32, u32)>::from_request(&req, &mut pl)
            .await
            .unwrap();
    }

    #[actix_rt::test]
    async fn test_custom_error_responder() {
        let req = TestRequest::with_uri("/name/user1/")
            .app_data(QueryConfig::default().error_handler(|e, _| {
                let resp = HttpResponse::UnprocessableEntity().finish();
                InternalError::from_response(e, resp).into()
            }))
            .to_srv_request();

        let (req, mut pl) = req.into_parts();
        let query = OpenapiQuery::<Id>::from_request(&req, &mut pl).await;

        assert!(query.is_err());
        assert_eq!(
            query
                .unwrap_err()
                .as_response_error()
                .error_response()
                .status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }


    #[test]
    fn deserialize_struct() {

        #[derive(Debug, Deserialize, Eq, PartialEq)]
        struct VecStruct {
            vec: Vec<String>,
        }

        let req = TestRequest::with_uri("/name/user1/?vec=a&vec=b").to_srv_request();
        let s = OpenapiQuery::<VecStruct>::from_query(req.query_string()).unwrap();

        assert_eq!(s.vec, vec!["a", "b"]);

        let req = TestRequest::with_uri("/name/user1/?vec[]=a&vec[]=b").to_srv_request();
        let s = OpenapiQuery::<VecStruct>::from_query(req.query_string()).unwrap();

        assert_eq!(s.vec, vec!["a", "b"]);

        let req = TestRequest::with_uri("/name/user1/?vec=a").to_srv_request();
        let s = OpenapiQuery::<VecStruct>::from_query(req.query_string()).unwrap();

        assert_eq!(s.vec, vec!["a"]);

    }
}
