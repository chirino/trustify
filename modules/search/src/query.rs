use crate::service::Error;
use regex::Regex;
use sea_orm::sea_query::IntoCondition;
use sea_orm::{ColumnTrait, ColumnType, Condition, EntityTrait, Iterable, Order};
use std::str::FromStr;
use std::sync::OnceLock;

pub struct Filter<T: EntityTrait> {
    operands: Operand<T>,
    operator: Operator,
}

pub struct Sort<T: EntityTrait> {
    pub field: T::Column,
    pub order: Order,
}

enum Operand<T: EntityTrait> {
    Simple(T::Column, String),
    Composite(Vec<Filter<T>>),
}

#[derive(Copy, Clone)]
pub enum Operator {
    Equal,
    NotEqual,
    Like,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    And,
    Or,
}

impl<T: EntityTrait> Filter<T> {
    pub fn into_condition(&self) -> Condition {
        match &self.operands {
            Operand::Simple(col, v) => match self.operator {
                Operator::Equal => col.eq(v).into_condition(),
                Operator::NotEqual => col.ne(v).into_condition(),
                Operator::Like => col
                    .contains(v.replace('%', r"\%").replace('_', r"\_"))
                    .into_condition(),
                Operator::GreaterThan => col.gt(v).into_condition(),
                Operator::GreaterThanOrEqual => col.gte(v).into_condition(),
                Operator::LessThan => col.lt(v).into_condition(),
                Operator::LessThanOrEqual => col.lte(v).into_condition(),
                _ => unreachable!(),
            },
            Operand::Composite(v) => match self.operator {
                Operator::And => v
                    .iter()
                    .fold(Condition::all(), |and, t| and.add(t.into_condition())),
                Operator::Or => v
                    .iter()
                    .fold(Condition::any(), |or, t| or.add(t.into_condition())),
                _ => unreachable!(),
            },
        }
    }
}

/////////////////////////////////////////////////////////////////////////
// FromStr impls
/////////////////////////////////////////////////////////////////////////

// Form expected: "full text search({field}{op}{value})*"
impl<T: EntityTrait> FromStr for Filter<T> {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        const RE: &str = r"^(?<field>[[:word:]]+)(?<op>=|!=|~|>=|>|<=|<)(?<value>.*)$";
        static LOCK: OnceLock<Regex> = OnceLock::new();
        #[allow(clippy::unwrap_used)]
        let filter = LOCK.get_or_init(|| (Regex::new(RE).unwrap()));

        let encoded = encode(s);
        if encoded.contains('&') {
            Ok(Filter {
                operator: Operator::And,
                operands: Operand::Composite(
                    encoded
                        .split('&')
                        .map(Self::from_str)
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            })
        } else if let Some(caps) = filter.captures(&encoded) {
            let field = &caps["field"];
            let col = T::Column::from_str(field).map_err(|_| {
                Error::SearchSyntax(format!("Invalid field name for filter: '{field}'"))
            })?;
            let operator = Operator::from_str(&caps["op"])?;
            Ok(Filter {
                operator: Operator::Or,
                operands: Operand::Composite(
                    caps["value"]
                        .split('|')
                        .map(|s| Filter {
                            operands: Operand::Simple(col, decode(s)),
                            operator,
                        })
                        .collect(),
                ),
            })
        } else {
            Ok(Filter {
                operator: Operator::Or,
                operands: Operand::Composite(
                    encoded
                        .split('|')
                        .flat_map(|s| {
                            T::Column::iter().filter_map(|col| match col.def().get_column_type() {
                                ColumnType::String(_) | ColumnType::Text => Some(Filter {
                                    operands: Operand::<T>::Simple(col, decode(s)),
                                    operator: Operator::Like,
                                }),
                                _ => None,
                            })
                        })
                        .collect(),
                ),
            })
        }
    }
}

impl<T: EntityTrait> FromStr for Sort<T> {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        let (field, order) = match s.split(':').collect::<Vec<_>>()[..] {
            [f, "asc"] | [f] => (f, Order::Asc),
            [f, "desc"] => (f, Order::Desc),
            _ => {
                return Err(Error::SearchSyntax(format!("Invalid sort: '{s}'")));
            }
        };
        Ok(Self {
            field: T::Column::from_str(field).map_err(|_| {
                Error::SearchSyntax(format!("Invalid field name for sort: '{field}'"))
            })?,
            order,
        })
    }
}

impl FromStr for Operator {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "=" => Ok(Operator::Equal),
            "!=" => Ok(Operator::NotEqual),
            "~" => Ok(Operator::Like),
            ">" => Ok(Operator::GreaterThan),
            ">=" => Ok(Operator::GreaterThanOrEqual),
            "<" => Ok(Operator::LessThan),
            "<=" => Ok(Operator::LessThanOrEqual),
            "|" => Ok(Operator::Or),
            "&" => Ok(Operator::And),
            _ => Err(Error::SearchSyntax(format!("Invalid operator: '{s}'"))),
        }
    }
}

/////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////

fn encode(s: &str) -> String {
    s.replace(r"\&", "\x07").replace(r"\|", "\x08")
}

fn decode(s: &str) -> String {
    s.replace('\x07', "&").replace('\x08', "|")
}

/////////////////////////////////////////////////////////////////////////
// Tests
/////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use sea_orm::{QueryFilter, QuerySelect, QueryTrait};
    use test_log::test;
    use trustify_entity::advisory;

    #[test(tokio::test)]
    async fn filters() -> Result<(), anyhow::Error> {
        // Good filters
        assert!(Filter::<advisory::Entity>::from_str("location=foo").is_ok());
        assert!(Filter::<advisory::Entity>::from_str("location!=foo").is_ok());
        assert!(Filter::<advisory::Entity>::from_str("location~foo").is_ok());
        assert!(Filter::<advisory::Entity>::from_str("location>foo").is_ok());
        assert!(Filter::<advisory::Entity>::from_str("location>=foo").is_ok());
        assert!(Filter::<advisory::Entity>::from_str("location<foo").is_ok());
        assert!(Filter::<advisory::Entity>::from_str("location<=foo").is_ok());
        assert!(Filter::<advisory::Entity>::from_str("something").is_ok());
        // Bad filters
        assert!(Filter::<advisory::Entity>::from_str("foo=bar").is_err());

        // There aren't many "bad filters" since random text is
        // considered a "full-text search" in which an OR clause is
        // constructed from a LIKE clause for all string fields in the
        // entity.

        Ok(())
    }

    #[test(tokio::test)]
    async fn sorts() -> Result<(), anyhow::Error> {
        // Good sorts
        assert!(Sort::<advisory::Entity>::from_str("location").is_ok());
        assert!(Sort::<advisory::Entity>::from_str("location:asc").is_ok());
        assert!(Sort::<advisory::Entity>::from_str("location:desc").is_ok());
        assert!(Sort::<advisory::Entity>::from_str("Location").is_ok());
        assert!(Sort::<advisory::Entity>::from_str("Location:Asc").is_ok());
        assert!(Sort::<advisory::Entity>::from_str("Location:Desc").is_ok());
        // Bad sorts
        assert!(Sort::<advisory::Entity>::from_str("foo").is_err());
        assert!(Sort::<advisory::Entity>::from_str("foo:").is_err());
        assert!(Sort::<advisory::Entity>::from_str(":foo").is_err());
        assert!(Sort::<advisory::Entity>::from_str("location:foo").is_err());
        assert!(Sort::<advisory::Entity>::from_str("location:asc:foo").is_err());

        Ok(())
    }

    fn where_clause(query: &str) -> Result<String, anyhow::Error> {
        Ok(advisory::Entity::find()
            .select_only()
            .column(advisory::Column::Id)
            .filter(Filter::<advisory::Entity>::from_str(query)?.into_condition())
            .build(sea_orm::DatabaseBackend::Postgres)
            .to_string()[45..]
            .to_string())
    }

    #[test(tokio::test)]
    async fn conditions() -> Result<(), anyhow::Error> {
        assert_eq!(
            where_clause("location=foo")?,
            r#""advisory"."location" = 'foo'"#
        );
        assert_eq!(
            where_clause("location!=foo")?,
            r#""advisory"."location" <> 'foo'"#
        );
        assert_eq!(
            where_clause("location~foo")?,
            r#""advisory"."location" LIKE '%foo%'"#
        );
        assert_eq!(
            where_clause("location~f_o%o")?,
            r#""advisory"."location" LIKE E'%f\\_o\\%o%'"#
        );
        assert_eq!(
            where_clause("location>foo")?,
            r#""advisory"."location" > 'foo'"#
        );
        assert_eq!(
            where_clause("location>=foo")?,
            r#""advisory"."location" >= 'foo'"#
        );
        assert_eq!(
            where_clause("location<foo")?,
            r#""advisory"."location" < 'foo'"#
        );
        assert_eq!(
            where_clause("location<=foo")?,
            r#""advisory"."location" <= 'foo'"#
        );
        assert_eq!(
            where_clause("foo")?,
            r#""advisory"."identifier" LIKE '%foo%' OR "advisory"."location" LIKE '%foo%' OR "advisory"."sha256" LIKE '%foo%' OR "advisory"."title" LIKE '%foo%'"#
        );
        assert_eq!(
            where_clause("foo&location=bar")?,
            r#"("advisory"."identifier" LIKE '%foo%' OR "advisory"."location" LIKE '%foo%' OR "advisory"."sha256" LIKE '%foo%' OR "advisory"."title" LIKE '%foo%') AND "advisory"."location" = 'bar'"#
        );
        assert_eq!(
            where_clause(r"m\&m's&location=f\&oo&id=ba\&r")?,
            r#"("advisory"."identifier" LIKE E'%m&m\'s%' OR "advisory"."location" LIKE E'%m&m\'s%' OR "advisory"."sha256" LIKE E'%m&m\'s%' OR "advisory"."title" LIKE E'%m&m\'s%') AND "advisory"."location" = 'f&oo' AND "advisory"."id" = 'ba&r'"#
        );
        assert_eq!(
            where_clause("location=a|b|c")?,
            r#""advisory"."location" = 'a' OR "advisory"."location" = 'b' OR "advisory"."location" = 'c'"#
        );
        assert_eq!(
            where_clause(r"location=foo|\&\|")?,
            r#""advisory"."location" = 'foo' OR "advisory"."location" = '&|'"#
        );
        assert_eq!(
            where_clause("a|b|c")?,
            r#""advisory"."identifier" LIKE '%a%' OR "advisory"."location" LIKE '%a%' OR "advisory"."sha256" LIKE '%a%' OR "advisory"."title" LIKE '%a%' OR "advisory"."identifier" LIKE '%b%' OR "advisory"."location" LIKE '%b%' OR "advisory"."sha256" LIKE '%b%' OR "advisory"."title" LIKE '%b%' OR "advisory"."identifier" LIKE '%c%' OR "advisory"."location" LIKE '%c%' OR "advisory"."sha256" LIKE '%c%' OR "advisory"."title" LIKE '%c%'"#
        );
        assert_eq!(
            where_clause("a|b&id=1")?,
            r#"("advisory"."identifier" LIKE '%a%' OR "advisory"."location" LIKE '%a%' OR "advisory"."sha256" LIKE '%a%' OR "advisory"."title" LIKE '%a%' OR "advisory"."identifier" LIKE '%b%' OR "advisory"."location" LIKE '%b%' OR "advisory"."sha256" LIKE '%b%' OR "advisory"."title" LIKE '%b%') AND "advisory"."id" = '1'"#
        );

        Ok(())
    }
}
