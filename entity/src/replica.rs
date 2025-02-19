use async_graphql::SimpleObject;
use sea_orm::entity::prelude::*;
use time::OffsetDateTime;

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, SimpleObject)]
#[sea_orm(table_name = "replica")]
#[graphql(concrete(name = "Replica", params()))]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: Uuid,
    pub created_at: OffsetDateTime,
    pub last_seen: OffsetDateTime,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
