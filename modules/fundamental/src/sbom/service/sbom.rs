use super::SbomService;
use crate::purl::model::summary::purl::PurlSummary;
use crate::sbom::model::details::{SbomAdvisory, SbomDetails};
use crate::sbom::model::SbomHead;
use crate::{
    sbom::model::{SbomPackage, SbomPackageReference, SbomPackageRelation, SbomSummary, Which},
    Error,
};
use futures_util::{stream, StreamExt, TryStreamExt};
use sea_orm::{
    prelude::Uuid, ColumnTrait, DbErr, EntityTrait, FromQueryResult, IntoSimpleExpr, ModelTrait,
    QueryFilter, QueryOrder, QueryResult, QuerySelect, RelationTrait, Select, SelectColumns,
};
use sea_query::{extension::postgres::PgExpr, Expr, Func, JoinType, SimpleExpr};
use serde::Deserialize;
use serde_json::Value;
use std::fmt::Debug;
use tracing::instrument;
use trustify_common::db::multi_model::{FromQueryResultMultiModel, SelectIntoMultiModel};
use trustify_common::db::ConnectionOrTransaction;
use trustify_common::{
    cpe::Cpe,
    db::{
        limiter::{limit_selector, LimiterTrait},
        query::{Filtering, IntoColumns, Query},
        ArrayAgg, JsonBuildObject, ToJson, Transactional,
    },
    id::{Id, TrySelectForId},
    model::{Paginated, PaginatedResults},
};
use trustify_entity::{
    advisory, base_purl,
    cpe::{self, CpeDto},
    labels::Labels,
    package_relates_to_package, purl_status,
    qualified_purl::{self, Qualifiers},
    relationship::Relationship,
    sbom::{self, SbomNodeLink},
    sbom_node, sbom_package, sbom_package_cpe_ref, sbom_package_purl_ref, status, versioned_purl,
    vulnerability,
};

impl SbomService {
    /// fetch one sbom
    pub async fn fetch_sbom<TX: AsRef<Transactional>>(
        &self,
        id: Id,
        tx: TX,
    ) -> Result<Option<SbomDetails>, Error> {
        let connection = self.db.connection(&tx);

        let select = sbom::Entity::find().try_filter(id)?;

        Ok(
            match select
                .find_also_linked(SbomNodeLink)
                .one(&connection)
                .await?
            {
                Some(row) => self.build_details(row, &tx).await?,
                None => None,
            },
        )
    }

    /// fetch all SBOMs
    pub async fn fetch_sboms<TX: AsRef<Transactional>>(
        &self,
        search: Query,
        paginated: Paginated,
        labels: impl Into<Labels>,

        tx: TX,
    ) -> Result<PaginatedResults<SbomSummary>, Error> {
        let connection = self.db.connection(&tx);
        let labels = labels.into();

        let mut query = sbom::Entity::find().filtering(search)?;

        if !labels.is_empty() {
            query = query.filter(Expr::col(sbom::Column::Labels).contains(labels));
        }

        let limiter = query.find_also_linked(SbomNodeLink).limiting(
            &connection,
            paginated.offset,
            paginated.limit,
        );

        let total = limiter.total().await?;
        let sboms = limiter.fetch().await?;

        let tx = tx.as_ref();
        let items = stream::iter(sboms.into_iter())
            .then(|row| async move { self.build_summary(row, &tx).await })
            .try_filter_map(futures_util::future::ok)
            .try_collect()
            .await?;

        Ok(PaginatedResults { total, items })
    }

    /// turn an (sbom, sbom_node) row into an [`SbomSummary`], if possible
    async fn build_summary(
        &self,
        (sbom, node): (sbom::Model, Option<sbom_node::Model>),
        tx: impl AsRef<Transactional>,
    ) -> Result<Option<SbomSummary>, Error> {
        // TODO: consider improving the n-select issue here
        let described_by = self
            .describes_packages(sbom.sbom_id, Paginated::default(), tx)
            .await?
            .items;

        Ok(match node {
            Some(node) => Some(SbomSummary {
                head: SbomHead {
                    id: sbom.sbom_id,
                    hashes: vec![Id::Sha256(sbom.sha256)],
                    document_id: sbom.document_id,
                    name: node.name,
                    labels: sbom.labels,
                },

                published: sbom.published,
                authors: sbom.authors,

                described_by,
            }),
            None => None,
        })
    }

    /// turn an (sbom, sbom_node) row into an [`SbomDetails`], if possible
    async fn build_details(
        &self,
        (sbom, node): (sbom::Model, Option<sbom_node::Model>),
        tx: impl AsRef<Transactional>,
    ) -> Result<Option<SbomDetails>, Error> {
        let connection = self.db.connection(&tx);

        let described_by = self
            .describes_packages(sbom.sbom_id, Paginated::default(), tx.as_ref())
            .await?
            .items;

        let relevant_advisory_info = sbom
            .find_related(sbom_package::Entity)
            .join(JoinType::Join, sbom_package::Relation::Node.def())
            .join(JoinType::LeftJoin, sbom_package::Relation::Purl.def())
            .join(
                JoinType::LeftJoin,
                sbom_package_purl_ref::Relation::Purl.def(),
            )
            .join(
                JoinType::LeftJoin,
                qualified_purl::Relation::VersionedPurl.def(),
            )
            .join(JoinType::LeftJoin, versioned_purl::Relation::BasePurl.def())
            .join(JoinType::Join, base_purl::Relation::PurlStatus.def())
            .join(JoinType::Join, purl_status::Relation::Status.def())
            .join(
                JoinType::LeftJoin,
                purl_status::Relation::VersionRange.def(),
            )
            .join(JoinType::LeftJoin, purl_status::Relation::ContextCpe.def())
            .join(JoinType::Join, purl_status::Relation::Advisory.def())
            .join(JoinType::Join, purl_status::Relation::Vulnerability.def())
            .select_only()
            .try_into_multi_model::<QueryCatcher>()?
            //.into_model::<QueryCatcher>()
            .all(&connection)
            .await?;

        Ok(match node {
            Some(node) => Some(SbomDetails {
                head: SbomHead {
                    id: sbom.sbom_id,
                    hashes: vec![Id::Sha256(sbom.sha256)],
                    document_id: sbom.document_id,
                    name: node.name,
                    labels: sbom.labels,
                },

                published: sbom.published,
                authors: sbom.authors,

                described_by: described_by.clone(),
                advisories: SbomAdvisory::from_models(
                    &described_by,
                    &relevant_advisory_info,
                    &connection,
                )
                .await?,
            }),
            None => None,
        })
    }

    /// Fetch all packages from an SBOM.
    ///
    /// If you need to find packages based on their relationship, even in the relationship to
    /// SBOM itself, use [`Self::fetch_related_packages`].
    #[instrument(skip(self, tx), err)]
    pub async fn fetch_sbom_packages<TX: AsRef<Transactional>>(
        &self,
        sbom_id: Uuid,
        search: Query,
        paginated: Paginated,
        tx: TX,
    ) -> Result<PaginatedResults<SbomPackage>, Error> {
        let db = self.db.connection(&tx);

        let mut query = sbom_package::Entity::find()
            .filter(sbom_package::Column::SbomId.eq(sbom_id))
            .join(JoinType::Join, sbom_package::Relation::Node.def())
            .select_only()
            .column_as(sbom_package::Column::NodeId, "id")
            .group_by(sbom_package::Column::NodeId)
            .column_as(sbom_package::Column::Version, "version")
            .group_by(sbom_package::Column::Version)
            .column_as(sbom_node::Column::Name, "name")
            .group_by(sbom_node::Column::Name)
            .join(JoinType::LeftJoin, sbom_package::Relation::Purl.def())
            .join(JoinType::LeftJoin, sbom_package::Relation::Cpe.def());

        query = join_purls_and_cpes(query)
            .filtering_with(
                search,
                sbom_package::Entity
                    .columns()
                    .add_columns(sbom_node::Entity)
                    .add_columns(base_purl::Entity)
                    .add_columns(sbom_package_cpe_ref::Entity)
                    .add_columns(sbom_package_purl_ref::Entity),
            )?
            .order_by_asc(sbom_package::Column::NodeId); // default order

        // limit and execute

        let limiter = limit_selector::<'_, _, _, _, PackageCatcher>(
            &db,
            query,
            paginated.offset,
            paginated.limit,
        );

        let total = limiter.total().await?;
        let packages = limiter.fetch().await?;

        // collect results

        let mut items = Vec::new();

        for row in packages {
            items.push(package_from_row(row, &self.db.connection(&tx)).await?);
        }

        Ok(PaginatedResults { items, total })
    }

    /// Get all packages describing the SBOM.
    #[instrument(skip(self, tx), err)]
    pub async fn describes_packages<TX: AsRef<Transactional>>(
        &self,
        sbom_id: Uuid,
        paginated: Paginated,
        tx: TX,
    ) -> Result<PaginatedResults<SbomPackage>, Error> {
        self.fetch_related_packages(
            sbom_id,
            Default::default(),
            paginated,
            Which::Right,
            SbomPackageReference::Root,
            Some(Relationship::DescribedBy),
            tx,
        )
        .await
        .map(|r| r.map(|rel| rel.package))
    }

    #[instrument(skip(self, tx), err)]
    pub async fn find_related_sboms(
        &self,
        qualified_package_id: Uuid,
        paginated: Paginated,
        query: Query,
        tx: impl AsRef<Transactional>,
    ) -> Result<PaginatedResults<SbomSummary>, Error> {
        let db = self.db.connection(&tx);

        let query = sbom::Entity::find()
            .join(JoinType::Join, sbom::Relation::Packages.def())
            .join(JoinType::Join, sbom_package::Relation::Purl.def())
            .filter(sbom_package_purl_ref::Column::QualifiedPurlId.eq(qualified_package_id))
            .filtering(query)?
            .find_also_linked(SbomNodeLink);

        // limit and execute

        let limiter = query.limiting(&db, paginated.offset, paginated.limit);

        let total = limiter.total().await?;
        let sboms = limiter.fetch().await?;

        // collect results

        let tx = tx.as_ref();
        let items = stream::iter(sboms.into_iter())
            .then(|row| async move { self.build_summary(row, &tx).await })
            .try_filter_map(futures_util::future::ok)
            .try_collect()
            .await?;

        Ok(PaginatedResults { items, total })
    }

    /// Fetch all related packages in the context of an SBOM.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, tx), err)]
    pub async fn fetch_related_packages<TX: AsRef<Transactional>>(
        &self,
        sbom_id: Uuid,
        search: Query,
        paginated: Paginated,
        which: Which,
        reference: impl Into<SbomPackageReference<'_>> + Debug,
        relationship: Option<Relationship>,
        tx: TX,
    ) -> Result<PaginatedResults<SbomPackageRelation>, Error> {
        let db = self.db.connection(&tx);

        // which way

        log::debug!("Which: {which:?}");

        // select all qualified packages for which we have relationships

        let (filter, join) = match which {
            Which::Left => (
                package_relates_to_package::Column::LeftNodeId,
                package_relates_to_package::Relation::Right,
            ),
            Which::Right => (
                package_relates_to_package::Column::RightNodeId,
                package_relates_to_package::Relation::Left,
            ),
        };

        let mut query = package_relates_to_package::Entity::find()
            .filter(package_relates_to_package::Column::SbomId.eq(sbom_id))
            .select_only()
            .select_column_as(sbom_node::Column::NodeId, "id")
            .group_by(sbom_node::Column::NodeId)
            .select_column_as(sbom_node::Column::Name, "name")
            .group_by(sbom_node::Column::Name)
            .select_column_as(
                package_relates_to_package::Column::Relationship,
                "relationship",
            )
            .group_by(package_relates_to_package::Column::Relationship)
            .select_column_as(sbom_package::Column::Version, "version")
            .group_by(sbom_package::Column::Version)
            // join the other side
            .join(JoinType::Join, join.def())
            .join(JoinType::Join, sbom_node::Relation::Package.def())
            .join(JoinType::LeftJoin, sbom_package::Relation::Purl.def())
            .join(JoinType::LeftJoin, sbom_package::Relation::Cpe.def());

        // collect PURLs and CPEs

        query = join_purls_and_cpes(query);

        // filter for reference

        query = match reference.into() {
            SbomPackageReference::Root => {
                // sbom - add join to sbom table
                query.join(JoinType::Join, sbom_node::Relation::Sbom.def())
            }
            SbomPackageReference::Package(node_id) => {
                // package - set node id filter
                query.filter(filter.eq(node_id))
            }
        };

        // apply filter conditions

        query = query.filtering(search)?;

        // add relationship type filter

        if let Some(relationship) = relationship {
            query = query.filter(package_relates_to_package::Column::Relationship.eq(relationship));
        }

        // limit and execute

        let limiter = limit_selector::<'_, _, _, _, PackageCatcher>(
            &db,
            query,
            paginated.offset,
            paginated.limit,
        );

        let total = limiter.total().await?;
        let packages = limiter.fetch().await?;

        // collect results

        let mut items = Vec::new();

        for row in packages {
            if let Some(relationship) = row.relationship {
                items.push(SbomPackageRelation {
                    relationship,
                    package: package_from_row(row, &self.db.connection(&tx)).await?,
                });
            }
        }

        Ok(PaginatedResults { items, total })
    }

    /// A simplified version of [`Self::fetch_related_packages`].
    ///
    /// It uses [`Which::Right`] and the provided reference, [`Default::default`] for the rest.
    pub async fn related_packages<TX: AsRef<Transactional>>(
        &self,
        sbom_id: Uuid,
        relationship: impl Into<Option<Relationship>>,
        pkg: impl Into<SbomPackageReference<'_>> + Debug,
        tx: TX,
    ) -> Result<Vec<SbomPackage>, Error> {
        let result = self
            .fetch_related_packages(
                sbom_id,
                Default::default(),
                Default::default(),
                Which::Right,
                pkg,
                relationship.into(),
                tx,
            )
            .await?;

        Ok(result.items.into_iter().map(|r| r.package).collect())
    }
}

/// Join CPE and PURL information.
///
/// Given a select over something which already joins sbom_package_purl_ref and
/// sbom_package_cpe_ref, this adds joins to fetch the data for PURLs and CPEs so that it can be
/// built using [`package_from_row`].
///
/// This will add the columns `purls` and `cpes` to the selected output.
fn join_purls_and_cpes<E>(query: Select<E>) -> Select<E>
where
    E: EntityTrait,
{
    query
        .join(
            JoinType::LeftJoin,
            sbom_package_purl_ref::Relation::Purl.def(),
        )
        .join(
            JoinType::LeftJoin,
            qualified_purl::Relation::VersionedPurl.def(),
        )
        .join(JoinType::LeftJoin, versioned_purl::Relation::BasePurl.def())
        // aggregate the q -> v -> p hierarchy into an array of json objects
        .select_column_as(
            Expr::cust_with_exprs(
                "coalesce($1 filter (where $2), '{}')",
                [
                    SimpleExpr::from(
                        Func::cust(ArrayAgg).arg(
                            Func::cust(JsonBuildObject)
                                // must match with PurlDto struct
                                .arg("base_purl_id")
                                .arg(base_purl::Column::Id.into_expr())
                                .arg("type")
                                .arg(base_purl::Column::Type.into_expr())
                                .arg("name")
                                .arg(base_purl::Column::Name.into_expr())
                                .arg("namespace")
                                .arg(base_purl::Column::Namespace.into_expr())
                                .arg("versioned_purl_id")
                                .arg(versioned_purl::Column::Id.into_expr())
                                .arg("version")
                                .arg(versioned_purl::Column::Version.into_expr())
                                .arg("qualified_purl_id")
                                .arg(qualified_purl::Column::Id.into_expr())
                                .arg("qualifiers")
                                .arg(qualified_purl::Column::Qualifiers.into_expr()),
                        ),
                    ),
                    sbom_package_purl_ref::Column::QualifiedPurlId
                        .is_not_null()
                        .into_simple_expr(),
                ],
            ),
            "purls",
        )
        .join(
            JoinType::LeftJoin,
            sbom_package_cpe_ref::Relation::Cpe.def(),
        )
        // aggregate the cpe rows into an array of json objects
        .select_column_as(
            Expr::cust_with_exprs(
                "coalesce($1 filter (where $2), '{}')",
                [
                    SimpleExpr::from(
                        Func::cust(ArrayAgg).arg(Func::cust(ToJson).arg(Expr::col(cpe::Entity))),
                    ),
                    sbom_package_cpe_ref::Column::CpeId.is_not_null(),
                ],
            ),
            "cpes",
        )
}

#[derive(FromQueryResult)]
struct PackageCatcher {
    id: String,
    name: String,
    version: Option<String>,
    purls: Vec<Value>,
    cpes: Vec<Value>,
    relationship: Option<Relationship>,
}

/// Convert values from a "package row" into an SBOM package
async fn package_from_row(
    row: PackageCatcher,
    tx: &ConnectionOrTransaction<'_>,
) -> Result<SbomPackage, Error> {
    let mut purls = Vec::new();

    for purl in row.purls {
        if let Ok(dto) = serde_json::from_value::<PurlDto>(purl) {
            purls.push(
                PurlSummary::from_entity(
                    &base_purl::Model {
                        id: dto.base_purl_id,
                        r#type: dto.r#type,
                        namespace: dto.namespace,
                        name: dto.name,
                    },
                    &versioned_purl::Model {
                        id: dto.versioned_purl_id,
                        base_purl_id: dto.base_purl_id,
                        version: dto.version,
                    },
                    &qualified_purl::Model {
                        id: dto.qualified_purl_id,
                        versioned_purl_id: dto.versioned_purl_id,
                        qualifiers: dto.qualifiers,
                    },
                    tx,
                )
                .await?,
            );
        }
    }

    Ok(SbomPackage {
        id: row.id,
        name: row.name,
        version: row.version,
        purl: purls,
        cpe: row
            .cpes
            .into_iter()
            .flat_map(|cpe| {
                serde_json::from_value::<CpeDto>(cpe)
                    .inspect_err(|err| {
                        log::warn!("Failed to deserialize CPE: {err}");
                    })
                    .ok()
            })
            .flat_map(|cpe| {
                log::debug!("CPE: {cpe:?}");
                Cpe::try_from(cpe)
                    .inspect_err(|err| {
                        log::warn!("Failed to build CPE: {err}");
                    })
                    .ok()
            })
            .map(|cpe| cpe.to_string())
            .collect(),
    })
}

#[derive(Clone, Debug, Deserialize)]
struct PurlDto {
    base_purl_id: Uuid,
    r#type: String,
    name: String,
    #[serde(default)]
    namespace: Option<String>,
    versioned_purl_id: Uuid,
    version: String,
    qualified_purl_id: Uuid,
    qualifiers: Qualifiers,
}

/*
impl From<PurlDto> for Purl {
    fn from(value: PurlDto) -> Self {
        let PurlDto {
            r#type,
            name,
            namespace,
            version,
            qualifiers,
        } = value;
        Self {
            ty: r#type,
            name,
            namespace,
            version: if version.is_empty() {
                None
            } else {
                Some(version)
            },
            qualifiers: qualifiers.0,
        }
    }
}

 */

#[derive(Debug)]
pub struct QueryCatcher {
    pub advisory: advisory::Model,
    pub base_purl: base_purl::Model,
    pub versioned_purl: versioned_purl::Model,
    pub qualified_purl: qualified_purl::Model,
    pub sbom_package: sbom_package::Model,
    pub sbom_node: sbom_node::Model,
    pub vulnerability: vulnerability::Model,
    pub context_cpe: Option<cpe::Model>,
    pub status: status::Model,
}

impl FromQueryResult for QueryCatcher {
    fn from_query_result(res: &QueryResult, _pre: &str) -> Result<Self, DbErr> {
        Ok(Self {
            advisory: Self::from_query_result_multi_model(res, advisory::Entity)?,
            vulnerability: Self::from_query_result_multi_model(res, vulnerability::Entity)?,
            base_purl: Self::from_query_result_multi_model(res, base_purl::Entity)?,
            versioned_purl: Self::from_query_result_multi_model(res, versioned_purl::Entity)?,
            qualified_purl: Self::from_query_result_multi_model(res, qualified_purl::Entity)?,
            sbom_package: Self::from_query_result_multi_model(res, sbom_package::Entity)?,
            sbom_node: Self::from_query_result_multi_model(res, sbom_node::Entity)?,
            context_cpe: Self::from_query_result_multi_model_optional(res, cpe::Entity)?,
            status: Self::from_query_result_multi_model(res, status::Entity)?,
        })
    }
}

impl FromQueryResultMultiModel for QueryCatcher {
    fn try_into_multi_model<E: EntityTrait>(select: Select<E>) -> Result<Select<E>, DbErr> {
        select
            .try_model_columns(advisory::Entity)?
            .try_model_columns(vulnerability::Entity)?
            .try_model_columns(base_purl::Entity)?
            .try_model_columns(versioned_purl::Entity)?
            .try_model_columns(qualified_purl::Entity)?
            .try_model_columns(sbom_package::Entity)?
            .try_model_columns(sbom_node::Entity)?
            .try_model_columns(status::Entity)?
            .try_model_columns(cpe::Entity)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use test_context::test_context;
    use test_log::test;
    use trustify_common::db::query::q;
    use trustify_common::hashing::Digests;
    use trustify_entity::labels::Labels;
    use trustify_test_context::TrustifyContext;

    #[test_context(TrustifyContext)]
    #[test(tokio::test)]
    async fn all_sboms(ctx: &TrustifyContext) -> Result<(), anyhow::Error> {
        let sbom_v1 = ctx
            .graph
            .ingest_sbom(
                Labels::default(),
                &Digests::digest("RHSA-1"),
                "http://redhat.com/test.json",
                (),
                Transactional::None,
            )
            .await?;
        let sbom_v1_again = ctx
            .graph
            .ingest_sbom(
                Labels::default(),
                &Digests::digest("RHSA-1"),
                "http://redhat.com/test.json",
                (),
                Transactional::None,
            )
            .await?;
        let sbom_v2 = ctx
            .graph
            .ingest_sbom(
                Labels::default(),
                &Digests::digest("RHSA-2"),
                "http://myspace.com/test.json",
                (),
                Transactional::None,
            )
            .await?;

        let _other_sbom = ctx
            .graph
            .ingest_sbom(
                Labels::default(),
                &Digests::digest("RHSA-3"),
                "http://geocities.com/other.json",
                (),
                Transactional::None,
            )
            .await?;

        assert_eq!(sbom_v1.sbom.sbom_id, sbom_v1_again.sbom.sbom_id);
        assert_ne!(sbom_v1.sbom.sbom_id, sbom_v2.sbom.sbom_id);

        let fetch = SbomService::new(ctx.db.clone());

        let fetched = fetch
            .fetch_sboms(q("MySpAcE"), Paginated::default(), (), ())
            .await?;

        log::debug!("{:#?}", fetched.items);
        assert_eq!(1, fetched.total);

        Ok(())
    }

    #[test_context(TrustifyContext)]
    #[test(tokio::test)]
    async fn labels(ctx: &TrustifyContext) -> Result<(), anyhow::Error> {
        let _sbom1 = ctx
            .graph
            .ingest_sbom(
                Labels::new()
                    .add("source", "test")
                    .add("ci", "job1")
                    .add("team", "a"),
                &Digests::digest("RHSA-1"),
                "http://redhat.com/test1.json",
                (),
                Transactional::None,
            )
            .await?;

        let _sbom2 = ctx
            .graph
            .ingest_sbom(
                Labels::new()
                    .add("source", "test")
                    .add("ci", "job2")
                    .add("team", "b"),
                &Digests::digest("RHSA-2"),
                "http://redhat.com/test2.json",
                (),
                Transactional::None,
            )
            .await?;

        let _sbom3 = ctx
            .graph
            .ingest_sbom(
                Labels::new()
                    .add("source", "test")
                    .add("ci", "job2")
                    .add("team", "a"),
                &Digests::digest("RHSA-3"),
                "http://redhat.com/test3.json",
                (),
                Transactional::None,
            )
            .await?;

        let service = SbomService::new(ctx.db.clone());

        let fetched = service
            .fetch_sboms(Query::default(), Paginated::default(), ("ci", "job1"), ())
            .await?;
        assert_eq!(1, fetched.total);

        let fetched = service
            .fetch_sboms(Query::default(), Paginated::default(), ("ci", "job2"), ())
            .await?;
        assert_eq!(2, fetched.total);

        let fetched = service
            .fetch_sboms(Query::default(), Paginated::default(), ("ci", "job3"), ())
            .await?;
        assert_eq!(0, fetched.total);

        let fetched = service
            .fetch_sboms(Query::default(), Paginated::default(), ("foo", "bar"), ())
            .await?;
        assert_eq!(0, fetched.total);

        let fetched = service
            .fetch_sboms(Query::default(), Paginated::default(), (), ())
            .await?;
        assert_eq!(3, fetched.total);

        let fetched = service
            .fetch_sboms(
                Query::default(),
                Paginated::default(),
                [("ci", "job2"), ("team", "a")],
                (),
            )
            .await?;
        assert_eq!(1, fetched.total);

        Ok(())
    }
}
