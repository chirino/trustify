use std::collections::HashSet;
use std::time::Duration;
use trustify_common::config::Database;
use trustify_module_importer::model::{
    CommonImporter, CsafImporter, CveImporter, ImporterConfiguration, OsvImporter, SbomImporter,
    DEFAULT_SOURCE_CVEPROJECT,
};
use trustify_module_importer::service::{Error, ImporterService};
use url::Url;

async fn add(
    importer: &ImporterService,
    name: &str,
    config: ImporterConfiguration,
) -> anyhow::Result<()> {
    Ok(importer
        .create(name.into(), config)
        .await
        .or_else(|err| match err {
            Error::AlreadyExists(_) => Ok(()),
            err => Err(err),
        })?)
}

async fn add_osv(
    importer: &ImporterService,
    name: &str,
    source: &str,
    base: Option<&str>,
    description: &str,
) -> anyhow::Result<()> {
    add(
        importer,
        name,
        ImporterConfiguration::Osv(OsvImporter {
            common: CommonImporter {
                disabled: true,
                period: Duration::from_secs(300),
                description: Some(description.into()),
                labels: Default::default(),
            },
            source: source.to_string(),
            path: base.map(|s| s.into()),
        }),
    )
    .await
}

async fn add_cve(
    importer: &ImporterService,
    name: &str,
    start_year: Option<u16>,
    description: &str,
) -> anyhow::Result<()> {
    add(
        importer,
        name,
        ImporterConfiguration::Cve(CveImporter {
            common: CommonImporter {
                disabled: true,
                period: Duration::from_secs(300),
                description: Some(description.into()),
                labels: Default::default(),
            },
            source: DEFAULT_SOURCE_CVEPROJECT.into(),
            years: HashSet::default(),
            start_year,
        }),
    )
    .await
}

pub async fn sample_data(db: trustify_common::db::Database) -> anyhow::Result<()> {
    let importer = ImporterService::new(db);

    add(&importer, "redhat-sbom",  ImporterConfiguration::Sbom(SbomImporter {
        common: CommonImporter {
            disabled: true,
            period: Duration::from_secs(300),
            description: Some("All Red Hat SBOMs".into()),
            labels: Default::default(),
        },
        source: "https://access.redhat.com/security/data/sbom/beta/".to_string(),
        keys: vec![
            Url::parse("https://access.redhat.com/security/data/97f5eac4.txt#77E79ABE93673533ED09EBE2DCE3823597F5EAC4")?
        ],
        v3_signatures: true,
        only_patterns: vec![],
    })).await?;

    add(
        &importer,
        "redhat-csaf",
        ImporterConfiguration::Csaf(CsafImporter {
            common: CommonImporter {
                disabled: true,
                period: Duration::from_secs(300),
                description: Some("All Red Hat CSAF data".into()),
                labels: Default::default(),
            },
            source: "redhat.com".to_string(),
            v3_signatures: true,
            only_patterns: vec![],
        }),
    )
    .await?;

    add(
        &importer,
        "redhat-csaf-vex-2024",
        ImporterConfiguration::Csaf(CsafImporter {
            common: CommonImporter {
                disabled: true,
                period: Duration::from_secs(300),
                description: Some("Red Hat VEX files from 2024".into()),
                labels: Default::default(),
            },
            source: "redhat.com".to_string(),
            v3_signatures: true,
            only_patterns: vec!["^cve-2024-".into()],
        }),
    )
    .await?;

    add_cve(&importer, "cve", None, "CVE List V5").await?;
    add_cve(
        &importer,
        "cve-from-2024",
        Some(2024),
        "CVE List V5 (starting 2024)",
    )
    .await?;

    add_osv(
        &importer,
        "osv-pypa",
        "https://github.com/pypa/advisory-database",
        Some("vulns"),
        "Python Packaging Advisory Database",
    )
    .await?;

    add_osv(
        &importer,
        "osv-psf",
        "https://github.com/psf/advisory-database",
        Some("advisories"),
        "Python Software Foundation Advisory Database",
    )
    .await?;

    add_osv(
        &importer,
        "osv-r",
        "https://github.com/RConsortium/r-advisory-database",
        Some("vulns"),
        "RConsortium Advisory Database",
    )
    .await?;

    add_osv(
        &importer,
        "osv-oss-fuzz",
        "https://github.com/google/oss-fuzz-vulns",
        Some("vulns"),
        "OSS-Fuzz vulnerabilities",
    )
    .await?;

    Ok(())
}
