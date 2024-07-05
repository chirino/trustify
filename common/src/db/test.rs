use std::env;

use postgresql_embedded::{PostgreSQL, Settings, VersionReq};
use test_context::AsyncTestContext;
use tracing::{info_span, instrument, Instrument};

#[allow(dead_code)]
pub struct TrustifyContext {
    pub db: crate::db::Database,
    postgresql: Option<PostgreSQL>,
}

impl AsyncTestContext for TrustifyContext {
    #[instrument]
    #[allow(clippy::expect_used)]
    async fn setup() -> TrustifyContext {
        if env::var("EXTERNAL_TEST_DB").is_ok() {
            log::warn!("Using external database from 'DB_*' env vars");
            let config = crate::config::Database::from_env().expect("DB config from env");

            let db = if env::var("EXTERNAL_TEST_DB_BOOTSTRAP").is_ok() {
                crate::db::Database::bootstrap(&config).await
            } else {
                crate::db::Database::new(&config).await
            }
            .expect("Configuring the database");

            return TrustifyContext {
                db,
                postgresql: None,
            };
        }

        let version = VersionReq::parse("=16.3.0").expect("valid psql version");
        let settings = Settings {
            version,
            username: "postgres".to_string(),
            password: "trustify".to_string(),
            temporary: true,
            ..Default::default()
        };

        let postgresql = async {
            let mut postgresql = PostgreSQL::new(settings);
            postgresql
                .setup()
                .await
                .expect("Setting up the test database");
            postgresql
                .start()
                .await
                .expect("Starting the test database");
            postgresql
        }
        .instrument(info_span!("start database"))
        .await;

        let config = crate::config::Database {
            username: "postgres".into(),
            password: "trustify".into(),
            host: "localhost".into(),
            name: "test".into(),
            port: postgresql.settings().port,
        };
        let db = crate::db::Database::bootstrap(&config)
            .await
            .expect("Bootstrapping the test database");

        TrustifyContext {
            db,
            postgresql: Some(postgresql),
        }
    }

    async fn teardown(self) {
        // Perform any teardown you wish.
    }
}
