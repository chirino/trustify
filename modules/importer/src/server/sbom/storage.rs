use crate::server::common::storage::StorageError;
use crate::server::report::{Message, Phase};
use crate::server::{context::RunContext, report::ReportBuilder};
use parking_lot::Mutex;
use sbom_walker::validation::{
    ValidatedSbom, ValidatedVisitor, ValidationContext, ValidationError,
};
use std::sync::Arc;
use tokio_util::io::ReaderStream;
use trustify_entity::labels::Labels;
use trustify_module_ingestor::service::{Format, IngestorService};
use walker_common::{compression::decompress_opt, utils::url::Urlify};

pub struct StorageVisitor {
    pub context: RunContext,
    pub source: String,
    pub labels: Labels,
    pub ingestor: IngestorService,
    /// the report to report our messages to
    pub report: Arc<Mutex<ReportBuilder>>,
}

impl ValidatedVisitor for StorageVisitor {
    type Error = StorageError<ValidationError>;
    type Context = ();

    async fn visit_context(
        &self,
        _context: &ValidationContext<'_>,
    ) -> Result<Self::Context, Self::Error> {
        Ok(())
    }

    async fn visit_sbom(
        &self,
        _context: &Self::Context,
        result: Result<ValidatedSbom, ValidationError>,
    ) -> Result<(), Self::Error> {
        let doc = result?;

        let (data, _compressed) = match decompress_opt(&doc.data, doc.url.path())
            .transpose()
            .map_err(StorageError::Processing)?
        {
            Some(data) => (data, true),
            None => (doc.data.clone(), false),
        };

        let file = doc.possibly_relative_url();

        let fmt = Format::sbom_from_bytes(&data).map_err(|e| StorageError::Processing(e.into()))?;

        let result = self
            .ingestor
            .ingest(
                Labels::new()
                    .add("source", &self.source)
                    .add("importer", self.context.name())
                    .add("file", &file)
                    .extend(&self.labels.0),
                None,
                fmt,
                ReaderStream::new(data.as_ref()),
            )
            .await
            .map_err(StorageError::Storage)?;

        self.report.lock().extend_messages(
            Phase::Upload,
            file,
            result.warnings.into_iter().map(Message::warning),
        );

        self.context.check_canceled(|| StorageError::Canceled).await
    }
}
