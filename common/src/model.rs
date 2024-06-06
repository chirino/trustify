use crate::db::limiter::Limiter;
use sea_orm::{ConnectionTrait, DbErr, ItemsAndPagesNumber, SelectorTrait};
use serde::{Serialize, Serializer};
use std::num::NonZeroU64;
use utoipa::{IntoParams, ToSchema};

pub use concat_idents::concat_idents;

/// A struct wrapping an item with a revision.
///
/// If the revision should not be part of the payload, but e.g. an HTTP header (like `ETag`), this
/// struct can help carrying both pieces.
// NOTE: This struct must be synced with the version in the [`revisioned`] macro below.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, ToSchema)]
pub struct Revisioned<T> {
    /// The actual value
    pub value: T,
    /// The revision.
    ///
    /// An opaque string that should have no meaning to the user, only to the backend.
    pub revision: String,
}

/// Creates a revisioned newtype for the provided type.
#[macro_export]
macro_rules! revisioned {
    ($n:ident) => {
        $crate::model::concat_idents!(RevisionedType = Revisioned, $n {
            #[derive(Clone, std::fmt::Debug, serde::Deserialize, serde::Serialize)]
            pub struct RevisionedType(pub trustify_common::model::Revisioned<$n>);

            impl<'s> utoipa::ToSchema<'s> for RevisionedType {
                fn schema() -> (&'s str, utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>) {
                    /// A struct wrapping an item with a revision.
                    ///
                    /// If the revision should not be part of the payload, but e.g. an HTTP header (like `ETag`), this
                    /// struct can help carrying both pieces.
                    #[derive(Clone, std::fmt::Debug, serde::Deserialize, serde::Serialize, utoipa::ToSchema)]
                    #[serde(rename_all = "camelCase")]
                    struct __SchemaType {
                        /// The actual value
                        pub value: $n,
                        /// The revision.
                        ///
                        /// An opaque string that should have no meaning to the user, only to the backend.
                        pub revision: String,
                    }

                    let (_, schema) = __SchemaType::schema();
                    (concat!("Revisioned", stringify!($n)), schema)
                }
            }
        });
    };
}

#[derive(
    IntoParams, Copy, Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub struct Paginated {
    /// The first item to return, skipping all that come before it.
    ///
    /// NOTE: The order of items is defined by the API being called.
    #[serde(default)]
    pub offset: u64,
    /// The maximum number of entries to return.
    ///
    /// Zero means: no limit
    #[serde(default = "default::limit")]
    pub limit: u64,
}

mod default {
    pub(super) const fn limit() -> u64 {
        25
    }
}

// NOTE: This struct must be aligned with the struct in the [`paginated`] macro below.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct PaginatedResults<R> {
    pub items: Vec<R>,
    pub total: u64,
}

impl<R> PaginatedResults<R> {
    /// Create a new paginated result
    pub async fn new<C, S1, S2>(
        limiter: Limiter<'_, C, S1, S2>,
    ) -> Result<PaginatedResults<S1::Item>, DbErr>
    where
        C: ConnectionTrait,
        S1: SelectorTrait,
        S2: SelectorTrait,
    {
        let total = limiter.total().await?;
        let results = limiter.fetch().await?;

        Ok(PaginatedResults {
            items: results,
            total,
        })
    }

    pub fn map<O, F: Fn(R) -> O>(mut self, f: F) -> PaginatedResults<O> {
        PaginatedResults {
            items: self.items.drain(..).map(f).collect(),
            total: self.total,
        }
    }
}

/// Creates an explicit ad-hoc [`PaginatedResults<T>`] type which can be used for `utoipa`. The
/// name of the type will be `PaginatedFoo` if the type is `Foo`.
#[macro_export]
macro_rules! paginated {
    ($n:ident) => {
        $crate::model::concat_idents!(PaginatedType = Paginated, $n {
            /// Paginated returned items
            #[derive(Clone, Debug, serde::Deserialize, serde::Serialize, ToSchema)]
            #[serde(rename_all = "camelCase")]
            pub struct PaginatedType {
                /// Returned items
                pub items: Vec<$n>,
                /// Total number of items found
                pub total: u64,
            }
        });
    };
}
