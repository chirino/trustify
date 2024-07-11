#![allow(clippy::expect_used)]

use anyhow::bail;
use test_context::test_context;
use test_log::test;
use trustify_common::id::Id;
use trustify_module_ingestor::model::IngestResult;
use trustify_test_context::TrustifyContext;

#[test_context(TrustifyContext)]
#[test(tokio::test)]
async fn reingest(ctx: &TrustifyContext) -> Result<(), anyhow::Error> {
    async fn assert(ctx: &TrustifyContext, result: IngestResult) -> anyhow::Result<()> {
        let Id::Uuid(id) = result.id else {
            bail!("must be an id")
        };
        let adv = ctx
            .graph
            .get_advisory_by_id(id, ())
            .await?
            .expect("must be found");

        assert_eq!(adv.vulnerabilities(()).await?.len(), 1);

        let all = adv.vulnerabilities(&()).await?;
        assert_eq!(all.len(), 1);
        assert_eq!(
            all[0].advisory_vulnerability.vulnerability_id,
            "CVE-2021-32714"
        );

        let all = ctx.graph.get_vulnerabilities(()).await?;
        assert_eq!(all.len(), 1);

        let vuln = ctx
            .graph
            .get_vulnerability("CVE-2021-32714", ())
            .await?
            .expect("Must be found");

        assert_eq!(vuln.vulnerability.id, "CVE-2021-32714");

        let descriptions = vuln.descriptions("en", ()).await?;
        assert_eq!(descriptions.len(), 0);

        Ok(())
    }

    // ingest once

    let result = ctx.ingest_document("osv/RUSTSEC-2021-0079.json").await?;
    assert(ctx, result).await?;

    // ingest second time

    let result = ctx.ingest_document("osv/RUSTSEC-2021-0079.json").await?;
    assert(ctx, result).await?;

    // done

    Ok(())
}
