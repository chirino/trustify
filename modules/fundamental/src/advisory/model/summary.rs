use sea_orm::{ColumnTrait, EntityTrait, LoaderTrait, QueryFilter, QuerySelect};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use trustify_common::db::ConnectionOrTransaction;
use trustify_common::paginated;
use trustify_cvss::cvss3::score::Score;
use trustify_entity::cvss3::Severity;
use trustify_entity::{advisory, advisory_vulnerability, organization, vulnerability};

use crate::advisory::model::{AdvisoryHead, AdvisoryVulnerabilityHead};
use crate::Error;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct AdvisorySummary {
    #[serde(flatten)]
    pub head: AdvisoryHead,

    /// Average (arithmetic mean) severity of the advisory aggregated from *all* related vulnerability assertions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_severity: Option<String>,

    /// Average (arithmetic mean) score of the advisory aggregated from *all* related vulnerability assertions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_score: Option<f64>,

    /// Vulnerabilities addressed within this advisory.
    pub vulnerabilities: Vec<AdvisoryVulnerabilityHead>,
}

paginated!(AdvisorySummary);

impl AdvisorySummary {
    pub async fn from_entities(
        entities: &[advisory::Model],
        averages: &[(Option<f64>, Option<Severity>)],
        tx: &ConnectionOrTransaction<'_>,
    ) -> Result<Vec<Self>, Error> {
        let issuers = entities.load_one(organization::Entity, tx).await?;

        let mut summaries = Vec::with_capacity(issuers.len());

        for ((advisory, issuer), (average_score, average_severity)) in
            entities.iter().zip(issuers.into_iter()).zip(averages)
        {
            let vulnerabilities = vulnerability::Entity::find()
                .right_join(advisory_vulnerability::Entity)
                .column_as(
                    advisory_vulnerability::Column::VulnerabilityId,
                    vulnerability::Column::Id,
                )
                .filter(advisory_vulnerability::Column::AdvisoryId.eq(advisory.id))
                .all(tx)
                .await?;

            let vulnerabilities =
                AdvisoryVulnerabilityHead::from_entities(advisory, &vulnerabilities, tx).await?;

            let average_score = average_score.map(|score| Score::new(score).roundup());

            summaries.push(AdvisorySummary {
                head: AdvisoryHead::from_advisory(advisory, issuer, tx).await?,
                average_severity: average_severity
                    .as_ref()
                    .map(|severity| severity.to_string()),
                average_score: average_score.map(|score| score.value()),
                vulnerabilities,
            })
        }

        Ok(summaries)
    }
}
