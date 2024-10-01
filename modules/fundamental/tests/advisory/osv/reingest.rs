use super::{twice, update_mark_fixed_again, update_unmark_fixed};
use test_context::test_context;
use test_log::test;
use trustify_module_fundamental::vulnerability::service::VulnerabilityService;
use trustify_module_ingestor::common::Deprecation;
use trustify_test_context::TrustifyContext;

/// Ensure that ingesting the same document twice, leads to the same ID.
#[test_context(TrustifyContext)]
#[test(tokio::test)]
async fn equal(ctx: &TrustifyContext) -> anyhow::Result<()> {
    let (r1, r2) = twice(ctx, |cve| cve, |cve| cve).await?;

    // no change, same result

    assert_eq!(r1.id, r2.id);

    // check info

    let vuln = VulnerabilityService::new(ctx.db.clone());
    let v = vuln
        .fetch_vulnerability("CVE-2020-5238", Default::default(), ())
        .await?
        .expect("must exist");

    assert_eq!(v.advisories.len(), 1);

    // done

    Ok(())
}

/// Update a document, ensure that we get one (ignoring deprecated), or two (considering deprecated).
#[test_context(TrustifyContext)]
#[test(tokio::test)]
async fn withdrawn(ctx: &TrustifyContext) -> anyhow::Result<()> {
    let (r1, r2) = twice(ctx, update_unmark_fixed, update_mark_fixed_again).await?;

    // must be changed

    assert_ne!(r1.id, r2.id);

    // check without deprecated

    let vuln = VulnerabilityService::new(ctx.db.clone());
    let v = vuln
        .fetch_vulnerability("CVE-2020-5238", Deprecation::Ignore, ())
        .await?
        .expect("must exist");

    assert_eq!(v.advisories.len(), 1);

    assert_eq!(v.advisories[0].head.head.identifier, "RSEC-2023-6");

    // check with deprecated

    let vuln = VulnerabilityService::new(ctx.db.clone());
    let v = vuln
        .fetch_vulnerability("CVE-2020-5238", Deprecation::Consider, ())
        .await?
        .expect("must exist");

    assert_eq!(v.advisories.len(), 2);

    println!("{v:#?}");

    assert_eq!(v.advisories[0].head.head.identifier, "RSEC-2023-6");
    assert_eq!(v.advisories[1].head.head.identifier, "RSEC-2023-6");

    // done

    Ok(())
}
