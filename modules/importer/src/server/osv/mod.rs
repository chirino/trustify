mod walker;

use crate::{
    model::OsvImporter,
    server::{
        common::walker::{CallbackError, Callbacks},
        context::RunContext,
        osv::walker::OsvWalker,
        report::{Phase, ReportBuilder, ScannerError},
        RunOutput,
    },
};
use osv::schema::Vulnerability;
use parking_lot::Mutex;
use std::{path::Path, path::PathBuf, sync::Arc};
use tokio::runtime::Handle;
use tokio_util::io::ReaderStream;
use tracing::instrument;
use trustify_entity::labels::Labels;
use trustify_module_ingestor::{
    graph::Graph,
    service::{Format, IngestorService},
};

struct Context {
    context: RunContext,
    source: String,
    labels: Labels,
    report: Arc<Mutex<ReportBuilder>>,
    ingestor: IngestorService,
}

impl Context {
    fn store(&self, path: &Path, osv: Vulnerability) -> anyhow::Result<()> {
        let data = serde_json::to_vec(&osv)?;

        self.report.lock().tick();

        Handle::current().block_on(async {
            self.ingestor
                .ingest(
                    Labels::new()
                        .add("source", &self.source)
                        .add("importer", self.context.name())
                        .add("file", path.to_string_lossy())
                        .extend(&self.labels.0),
                    None,
                    Format::OSV,
                    ReaderStream::new(data.as_slice()),
                )
                .await
        })?;

        Ok(())
    }
}

impl Callbacks<Vulnerability> for Context {
    fn loading_error(&mut self, path: PathBuf, message: String) {
        self.report
            .lock()
            .add_error(Phase::Validation, path.to_string_lossy(), message);
    }

    fn process(&mut self, path: &Path, osv: Vulnerability) -> Result<(), CallbackError> {
        if let Err(err) = self.store(path, osv) {
            self.report
                .lock()
                .add_error(Phase::Upload, path.to_string_lossy(), err.to_string());
        }

        self.context.check_canceled_sync(|| CallbackError::Canceled)
    }
}

impl super::Server {
    #[instrument(skip(self), ret)]
    pub async fn run_once_osv(
        &self,
        context: RunContext,
        osv: OsvImporter,
        continuation: serde_json::Value,
    ) -> Result<RunOutput, ScannerError> {
        let ingestor = IngestorService::new(Graph::new(self.db.clone()), self.storage.clone());

        let report = Arc::new(Mutex::new(ReportBuilder::new()));
        let continuation = serde_json::from_value(continuation).unwrap_or_default();

        // working dir

        let working_dir = self.create_working_dir("osv", &osv.source).await?;

        // run the walker

        let walker = OsvWalker::new(osv.source.clone())
            .continuation(continuation)
            .path(osv.path)
            .callbacks(Context {
                context,
                source: osv.source,
                labels: osv.common.labels,
                report: report.clone(),
                ingestor,
            });

        let continuation = match working_dir {
            Some(working_dir) => walker.working_dir(working_dir).run().await,
            None => walker.run().await,
        }
        .map_err(|err| ScannerError::Critical(err.into()))?;

        // extract the report

        let report = match Arc::try_unwrap(report) {
            Ok(report) => report.into_inner(),
            Err(report) => report.lock().clone(),
        }
        .build();

        // return

        Ok(RunOutput {
            report,
            continuation: serde_json::to_value(continuation).ok(),
        })
    }
}
