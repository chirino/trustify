use human_date_parser::{from_human_time, ParseResult};
use regex::Regex;
use sea_orm::entity::ColumnDef;
use sea_orm::sea_query::{extension::postgres::PgExpr, ConditionExpression, IntoCondition};
use sea_orm::{
    sea_query, ColumnTrait, ColumnType, Condition, EntityName, EntityTrait, Iden, IntoIdentity,
    Iterable, Order, PrimaryKeyToColumn, QueryFilter, QueryOrder, QuerySelect, QueryTrait, Select,
    Value,
};
use sea_query::{BinOper, ColumnRef, DynIden, Expr, IntoColumnRef, SimpleExpr};
use std::collections::HashMap;
use std::fmt::Display;
use std::str::FromStr;
use std::sync::OnceLock;
use time::format_description::well_known::Rfc3339;
use time::macros::format_description;
use time::{Date, OffsetDateTime};
use utoipa::IntoParams;

/////////////////////////////////////////////////////////////////////////
// Public interface
/////////////////////////////////////////////////////////////////////////

/// Convenience function for creating a search query
///
/// ```
/// use trustify_common::db::query::q;
///
/// let query = q("foo&bar>100").sort("bar:desc,baz");
///
/// ```
pub fn q(s: &str) -> Query {
    Query::q(s)
}

impl Query {
    /// Form expected: `{query}*{filter}*`
    ///
    /// where `{filter}` is of the form `{field}{op}{value}`
    ///
    /// Multiple queries and/or filters should be `&`-delimited
    ///
    /// The `{query}` text will result in an OR clause of LIKE clauses
    /// for every [String] field in the associated
    /// [Entity](sea_orm::EntityTrait). Optional filters of the form
    /// `{field}{op}{value}` may further constrain the results. Each
    /// `{field}` must name an actual
    /// [Column](sea_orm::EntityTrait::Column) variant.
    ///
    /// Both `{query}` and `{value}` may contain `|`-delimited
    /// alternate values that will result in an OR clause. Any `|` or
    /// `&` within a query/value should be escaped with a backslash,
    /// e.g. `\|` or `\&`.
    ///
    /// `{op}` should be one of `=`, `!=`, `~`, `!~, `>=`, `>`, `<=`,
    /// or `<`.
    pub fn q(s: &str) -> Self {
        Self {
            q: s.into(),
            sort: String::default(),
        }
    }

    /// Form expected: `{sort}*`
    ///
    /// where `{sort}` is of the form `{field}[:order]` and the
    /// optional `order` should be one of `asc` or `desc`. If omitted,
    /// the order defaults to `asc`.
    ///
    /// Multiple sorts should be `,`-delimited
    ///
    /// Each `{field}` must name an actual
    /// [Column](sea_orm::EntityTrait::Column) variant.
    ///
    pub fn sort(self, s: &str) -> Self {
        Self {
            q: self.q,
            sort: s.into(),
        }
    }
}

pub trait Filtering<T: EntityTrait> {
    fn filtering(self, search: Query) -> Result<Self, Error>
    where
        Self: Sized,
    {
        self.filtering_with(search, Columns::from_entity::<T>())
    }

    fn filtering_with<C: IntoColumns>(self, search: Query, context: C) -> Result<Self, Error>
    where
        Self: Sized;
}

impl<T: EntityTrait> Filtering<T> for Select<T> {
    fn filtering_with<C: IntoColumns>(self, search: Query, context: C) -> Result<Self, Error> {
        let Query { q, sort } = &search;
        let columns = context.columns();

        let mut result = if q.is_empty() {
            self
        } else {
            self.filter(Filter::parse(q, &columns)?)
        };

        if !sort.is_empty() {
            result = sort
                .split(',')
                .map(|s| Sort::parse(s, &columns))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .fold(result, |select, s| {
                    select.order_by(SimpleExpr::Column(s.field), s.order)
                });
        };

        Ok(result)
    }
}

#[derive(
    Clone,
    Default,
    Debug,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
    utoipa::IntoParams,
)]
#[serde(rename_all = "camelCase")]
pub struct Query {
    #[serde(default)]
    pub q: String,
    #[serde(default)]
    pub sort: String,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("query syntax error: {0}")]
    SearchSyntax(String),
}

pub trait IntoColumns {
    fn columns(self) -> Columns;
}

impl IntoColumns for Columns {
    fn columns(self) -> Columns {
        self
    }
}

impl<E: EntityTrait> IntoColumns for E {
    fn columns(self) -> Columns {
        Columns::from_entity::<E>()
    }
}

/// Context of columns which can be used for filtering and sorting.
#[derive(Default, Debug, Clone)]
pub struct Columns {
    columns: Vec<(ColumnRef, ColumnDef)>,
}

impl Columns {
    /// Construct a new columns context from an entity type.
    pub fn from_entity<E: EntityTrait>() -> Self {
        let columns = E::Column::iter()
            .map(|c| {
                let (t, u) = c.as_column_ref();
                let column_ref = ColumnRef::TableColumn(t, u);
                let column_def = c.def();
                (column_ref, column_def)
            })
            .collect();
        Self { columns }
    }

    /// Add an arbitrary column into the context.
    pub fn add_column<I: IntoIdentity>(mut self, name: I, def: ColumnDef) -> Self {
        self.columns
            .push((name.into_identity().into_column_ref(), def));
        self
    }

    pub fn iter(&self) -> impl Iterator<Item = &(ColumnRef, ColumnDef)> {
        self.columns.iter()
    }

    /// Look up the column context for a given simple field name.
    fn for_field(&self, field: &str) -> Option<(ColumnRef, ColumnDef)> {
        self.columns
            .iter()
            .find(|(col_ref, _)| {
                matches!( col_ref,
                   ColumnRef::Column(name)
                    | ColumnRef::TableColumn(_, name)
                    | ColumnRef::SchemaTableColumn(_, _, name)
                        if name.to_string() == field)
            })
            .cloned()
    }
}

/////////////////////////////////////////////////////////////////////////
// Internal types
/////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
struct Filter {
    operands: Operand,
    operator: Operator,
}

impl Filter {
    fn parse(s: &str, columns: &Columns) -> Result<Self, Error> {
        const RE: &str = r"^(?<field>[[:word:]]+)(?<op>=|!=|~|!~|>=|>|<=|<)(?<value>.*)$";
        static LOCK: OnceLock<Regex> = OnceLock::new();
        #[allow(clippy::unwrap_used)]
        let filter = LOCK.get_or_init(|| (Regex::new(RE).unwrap()));

        let encoded = encode(s);
        if encoded.contains('&') {
            // We have a collection of filters and/or queries
            Ok(Filter {
                operator: Operator::And,
                operands: Operand::Composite(
                    encoded
                        .split('&')
                        .map(|e| Filter::parse(e, columns))
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            })
        } else if let Some(caps) = filter.captures(&encoded) {
            // We have a filter: {field}{op}{value}
            let field = &caps["field"];
            let (col_ref, col_def) = columns.for_field(field).ok_or(Error::SearchSyntax(
                format!("Invalid field name for filter: '{field}'"),
            ))?;
            let operator = Operator::from_str(&caps["op"])?;
            Ok(Filter {
                operator: match operator {
                    Operator::NotLike | Operator::NotEqual => Operator::And,
                    _ => Operator::Or,
                },
                operands: Operand::Composite(
                    caps["value"]
                        .split('|')
                        .map(decode)
                        .map(|s| envalue(&s, col_def.get_column_type()))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .map(|v| Filter {
                            operands: Operand::Simple(col_ref.clone(), v),
                            operator,
                        })
                        .collect(),
                ),
            })
        } else {
            // We have a full-text search query
            Ok(Filter {
                operator: Operator::Or,
                operands: Operand::Composite(
                    encoded
                        .split('|')
                        .flat_map(|s| {
                            columns.iter().filter_map(|(col_ref, col_def)| {
                                match col_def.get_column_type() {
                                    ColumnType::String(_) | ColumnType::Text => Some(Filter {
                                        operands: Operand::Simple(
                                            col_ref.clone(),
                                            decode(s).into(),
                                        ),
                                        operator: Operator::Like,
                                    }),
                                    _ => None,
                                }
                            })
                        })
                        .collect(),
                ),
            })
        }
    }
}

struct Sort {
    field: ColumnRef,
    order: Order,
}

impl Sort {
    fn parse(s: &str, columns: &Columns) -> Result<Self, Error> {
        let s = s.to_lowercase();
        let (field, order) = match s.split(':').collect::<Vec<_>>()[..] {
            [f, "asc"] | [f] => (f, Order::Asc),
            [f, "desc"] => (f, Order::Desc),
            _ => {
                return Err(Error::SearchSyntax(format!("Invalid sort: '{s}'")));
            }
        };
        Ok(Self {
            field: columns
                .for_field(field)
                .ok_or(Error::SearchSyntax(format!(
                    "Invalid field name for sort: '{field}'"
                )))?
                .0,
            order,
        })
    }
}

/////////////////////////////////////////////////////////////////////////
// SeaORM impls
/////////////////////////////////////////////////////////////////////////

impl IntoCondition for Filter {
    fn into_condition(self) -> Condition {
        match self.operands {
            Operand::Simple(col, v) => match self.operator {
                Operator::Equal => {
                    let expr = Expr::val(v);
                    Expr::col(col).binary(BinOper::Equal, expr)
                }
                Operator::NotEqual => {
                    let expr = Expr::val(v);
                    Expr::col(col).binary(BinOper::NotEqual, expr)
                }

                Operator::GreaterThan => {
                    let expr = Expr::val(v);
                    Expr::col(col).binary(BinOper::GreaterThan, expr)
                }
                Operator::GreaterThanOrEqual => {
                    let expr = Expr::val(v);
                    Expr::col(col).binary(BinOper::GreaterThanOrEqual, expr)
                }
                Operator::LessThan => {
                    let expr = Expr::val(v);
                    Expr::col(col).binary(BinOper::SmallerThan, expr)
                }
                Operator::LessThanOrEqual => {
                    let expr = Expr::val(v);
                    Expr::col(col).binary(BinOper::SmallerThanOrEqual, expr)
                }
                op @ (Operator::Like | Operator::NotLike) => {
                    let v = format!(
                        "%{}%",
                        v.unwrap::<String>().replace('%', r"\%").replace('_', r"\_")
                    );
                    if op == Operator::Like {
                        SimpleExpr::Column(col).ilike(v)
                    } else {
                        SimpleExpr::Column(col).not_ilike(v)
                    }
                }
                _ => unreachable!(),
            }
            .into_condition(),
            Operand::Composite(v) => match self.operator {
                Operator::And => v.into_iter().fold(Condition::all(), |and, f| and.add(f)),
                Operator::Or => v.into_iter().fold(Condition::any(), |or, f| or.add(f)),
                _ => unreachable!(),
            },
        }
    }
}

impl From<Filter> for ConditionExpression {
    fn from(f: Filter) -> Self {
        ConditionExpression::Condition(f.into_condition())
    }
}

/////////////////////////////////////////////////////////////////////////
// FromStr impls
/////////////////////////////////////////////////////////////////////////

impl FromStr for Operator {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "=" => Ok(Operator::Equal),
            "!=" => Ok(Operator::NotEqual),
            "~" => Ok(Operator::Like),
            "!~" => Ok(Operator::NotLike),
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
// Internal helpers
/////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
enum Operand {
    Simple(ColumnRef, Value),
    Composite(Vec<Filter>),
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Operator {
    Equal,
    NotEqual,
    Like,
    NotLike,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    And,
    Or,
}

fn encode(s: &str) -> String {
    s.replace(r"\&", "\x07").replace(r"\|", "\x08")
}

fn decode(s: &str) -> String {
    s.replace('\x07', "&").replace('\x08', "|")
}

fn envalue(s: &str, ct: &ColumnType) -> Result<Value, Error> {
    fn err(e: impl Display) -> Error {
        Error::SearchSyntax(format!(r#"conversion error: "{e}""#))
    }
    Ok(match ct {
        ColumnType::Integer => s.parse::<i32>().map_err(err)?.into(),
        ColumnType::Decimal(_) => s.parse::<f64>().map_err(err)?.into(),
        ColumnType::TimestampWithTimeZone => {
            if let Ok(odt) = OffsetDateTime::parse(s, &Rfc3339) {
                odt.into()
            } else if let Ok(d) = Date::parse(s, &format_description!("[year]-[month]-[day]")) {
                d.into()
            } else if let Ok(human) = from_human_time(s) {
                match human {
                    ParseResult::DateTime(dt) => dt.into(),
                    ParseResult::Date(d) => d.into(),
                    ParseResult::Time(t) => t.into(),
                }
            } else {
                s.into()
            }
        }
        _ => s.into(),
    })
}

/////////////////////////////////////////////////////////////////////////
// Tests
/////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Local, TimeDelta};
    use sea_orm::{ColumnTypeTrait, QueryFilter, QuerySelect, QueryTrait};
    use sea_query::{Func, Function, IntoIden};
    use test_log::test;

    #[test(tokio::test)]
    async fn happy_path() -> Result<(), anyhow::Error> {
        let stmt = advisory::Entity::find()
            .filtering(q("foo&published>2024-04-20").sort("location,title:desc"))?
            .order_by_desc(advisory::Column::Id)
            .build(sea_orm::DatabaseBackend::Postgres)
            .to_string()[106..]
            .to_string();
        assert_eq!(
            stmt,
            r#"WHERE (("advisory"."location" ILIKE '%foo%') OR ("advisory"."title" ILIKE '%foo%')) AND "advisory"."published" > '2024-04-20' ORDER BY "advisory"."location" ASC, "advisory"."title" DESC, "advisory"."id" DESC"#
        );
        Ok(())
    }

    #[test(tokio::test)]
    async fn filters() -> Result<(), anyhow::Error> {
        let columns = advisory::Entity.columns();
        let test = |s: &str, expected: Operator| match Filter::parse(s, &columns) {
            Ok(Filter {
                operands: Operand::Composite(v),
                ..
            }) => assert_eq!(
                v[0].operator, expected,
                "The query '{s}' didn't resolve to {expected:?}"
            ),
            _ => panic!("The query '{s}' didn't resolve to {expected:?}"),
        };

        // Good filters
        test("location=foo", Operator::Equal);
        test("location!=foo", Operator::NotEqual);
        test("location~foo", Operator::Like);
        test("location!~foo", Operator::NotLike);
        test("location>foo", Operator::GreaterThan);
        test("location>=foo", Operator::GreaterThanOrEqual);
        test("location<foo", Operator::LessThan);
        test("location<=foo", Operator::LessThanOrEqual);

        // If a query matches the '{field}{op}{value}' regex, then the
        // first operand must resolve to a field on the Entity
        assert!(Filter::parse("foo=bar", &columns).is_err());

        // There aren't many bad queries since random text is
        // considered a "full-text search" in which an OR clause is
        // constructed from a LIKE clause for all string fields in the
        // entity.
        test("search the entity", Operator::Like);

        Ok(())
    }

    #[test(tokio::test)]
    async fn filters_extra_columns() -> Result<(), anyhow::Error> {
        let test = |s: &str, expected: Operator| {
            let columns = advisory::Entity
                .columns()
                .add_column("len", ColumnType::Integer.def());
            match Filter::parse(s, &columns) {
                Ok(Filter {
                    operands: Operand::Composite(v),
                    ..
                }) => assert_eq!(
                    v[0].operator, expected,
                    "The query '{s}' didn't resolve to {expected:?}"
                ),
                _ => panic!("The query '{s}' didn't resolve to {expected:?}"),
            }
        };

        test("len=42", Operator::Equal);
        test("len!=42", Operator::NotEqual);
        test("len~42", Operator::Like);
        test("len!~42", Operator::NotLike);
        test("len>42", Operator::GreaterThan);
        test("len>=42", Operator::GreaterThanOrEqual);
        test("len<42", Operator::LessThan);
        test("len<=42", Operator::LessThanOrEqual);

        Ok(())
    }

    #[test(tokio::test)]
    async fn sorts() -> Result<(), anyhow::Error> {
        let columns = advisory::Entity.columns();
        // Good sorts
        assert!(Sort::parse("location", &columns).is_ok());
        assert!(Sort::parse("location:asc", &columns).is_ok());
        assert!(Sort::parse("location:desc", &columns).is_ok());
        assert!(Sort::parse("Location", &columns).is_ok());
        assert!(Sort::parse("Location:Asc", &columns).is_ok());
        assert!(Sort::parse("Location:Desc", &columns).is_ok());
        // Bad sorts
        assert!(Sort::parse("foo", &columns).is_err());
        assert!(Sort::parse("foo:", &columns).is_err());
        assert!(Sort::parse(":foo", &columns).is_err());
        assert!(Sort::parse("location:foo", &columns).is_err());
        assert!(Sort::parse("location:asc:foo", &columns).is_err());

        // Good sorts with other columns
        assert!(Sort::parse(
            "foo",
            &advisory::Entity
                .columns()
                .add_column("foo", ColumnType::String(None).def())
        )
        .is_ok());

        // Bad sorts with other columns
        assert!(Sort::parse(
            "bar",
            &advisory::Entity
                .columns()
                .add_column("foo", ColumnType::String(None).def())
        )
        .is_err());

        Ok(())
    }

    #[test(tokio::test)]
    async fn conditions_on_extra_columns() -> Result<(), anyhow::Error> {
        let query = advisory::Entity::find()
            .column(advisory::Column::Id)
            .expr_as_(
                Func::char_length(Expr::col("location".into_identity())),
                "location_len",
            );

        let sql = query
            .filtering_with(
                q("location_len>10"),
                advisory::Entity
                    .columns()
                    .add_column("location_len", ColumnType::Integer.def()),
            )?
            .build(sea_orm::DatabaseBackend::Postgres)
            .to_string();

        assert_eq!(
            sql,
            r#"SELECT "advisory"."id", "advisory"."location", "advisory"."title", "advisory"."published", "advisory"."id", CHAR_LENGTH("location") AS "location_len" FROM "advisory" WHERE "location_len" > 10"#
        );

        Ok(())
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
            r#""advisory"."location" ILIKE '%foo%'"#
        );
        assert_eq!(
            where_clause("location~f_o%o")?,
            r#""advisory"."location" ILIKE E'%f\\_o\\%o%'"#
        );
        assert_eq!(
            where_clause("location!~foo")?,
            r#""advisory"."location" NOT ILIKE '%foo%'"#
        );
        assert_eq!(
            where_clause("location!~f_o%o")?,
            r#""advisory"."location" NOT ILIKE E'%f\\_o\\%o%'"#
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
            where_clause("location=a|b|c")?,
            r#""advisory"."location" = 'a' OR "advisory"."location" = 'b' OR "advisory"."location" = 'c'"#
        );
        assert_eq!(
            where_clause("location!=a|b|c")?,
            r#""advisory"."location" <> 'a' AND "advisory"."location" <> 'b' AND "advisory"."location" <> 'c'"#
        );
        assert_eq!(
            where_clause(r"location=foo|\&\|")?,
            r#""advisory"."location" = 'foo' OR "advisory"."location" = '&|'"#
        );
        assert_eq!(
            where_clause("published>2023-11-03T23:20:50.52Z")?,
            r#""advisory"."published" > '2023-11-03 23:20:50.520000 +00:00'"#
        );
        assert_eq!(
            where_clause("published>2023-11-03T23:20:51-04:00")?,
            r#""advisory"."published" > '2023-11-03 23:20:51.000000 -04:00'"#
        );
        assert_eq!(
            where_clause("published>2023-11-03")?,
            r#""advisory"."published" > '2023-11-03'"#
        );

        Ok(())
    }

    #[test(tokio::test)]
    async fn complex_ilikes() -> Result<(), anyhow::Error> {
        //
        // I broke these assertions out into their own test as they
        // resulted in very conservative parentheses when moving from
        // LIKE to ILIKE. I think the extra parens are harmless, but I
        // suspect it may be a bug that LIKE and ILIKE operators are
        // treated differently, as their precedence should be the same
        // on PostgreSQL.
        //
        // Upstream issue: https://github.com/SeaQL/sea-query/issues/776
        // See also https://github.com/SeaQL/sea-query/pull/675

        assert_eq!(
            where_clause("foo")?,
            r#"("advisory"."location" ILIKE '%foo%') OR ("advisory"."title" ILIKE '%foo%')"#
        );
        assert_eq!(
            where_clause("foo&location=bar")?,
            r#"(("advisory"."location" ILIKE '%foo%') OR ("advisory"."title" ILIKE '%foo%')) AND "advisory"."location" = 'bar'"#
        );
        assert_eq!(
            where_clause(r"m\&m's&location=f\&oo&id=13")?,
            r#"(("advisory"."location" ILIKE E'%m&m\'s%') OR ("advisory"."title" ILIKE E'%m&m\'s%')) AND "advisory"."location" = 'f&oo' AND "advisory"."id" = 13"#
        );
        assert_eq!(
            where_clause("a|b|c")?,
            r#"("advisory"."location" ILIKE '%a%') OR ("advisory"."title" ILIKE '%a%') OR ("advisory"."location" ILIKE '%b%') OR ("advisory"."title" ILIKE '%b%') OR ("advisory"."location" ILIKE '%c%') OR ("advisory"."title" ILIKE '%c%')"#
        );
        assert_eq!(
            where_clause("a|b&id=1")?,
            r#"(("advisory"."location" ILIKE '%a%') OR ("advisory"."title" ILIKE '%a%') OR ("advisory"."location" ILIKE '%b%') OR ("advisory"."title" ILIKE '%b%')) AND "advisory"."id" = 1"#
        );
        assert_eq!(
            where_clause("a&b")?,
            r#"(("advisory"."location" ILIKE '%a%') OR ("advisory"."title" ILIKE '%a%')) AND (("advisory"."location" ILIKE '%b%') OR ("advisory"."title" ILIKE '%b%'))"#
        );
        assert_eq!(
            where_clause("here&location!~there|hereford")?,
            r#"(("advisory"."location" ILIKE '%here%') OR ("advisory"."title" ILIKE '%here%')) AND (("advisory"."location" NOT ILIKE '%there%') AND ("advisory"."location" NOT ILIKE '%hereford%'))"#
        );

        Ok(())
    }

    #[test(tokio::test)]
    async fn human_time() -> Result<(), anyhow::Error> {
        let now = Local::now();
        let yesterday = (now - TimeDelta::try_days(1).unwrap()).format("%Y-%m-%d");
        let last_week = (now - TimeDelta::try_days(7).unwrap()).format("%Y-%m-%d");
        let three_days_ago = (now - TimeDelta::try_days(3).unwrap()).format("%Y-%m-%d");
        assert_eq!(
            where_clause("published<yesterday")?,
            format!(r#""advisory"."published" < '{yesterday}'"#)
        );
        assert_eq!(
            where_clause("published>last week")?,
            format!(r#""advisory"."published" > '{last_week}'"#)
        );
        let wc = where_clause("published=3 days ago")?;
        let expected = &format!(r#""advisory"."published" = '{three_days_ago} "#);
        assert!(
            wc.starts_with(expected),
            "expected '{wc}' to start with '{expected}'"
        );

        // Other possibilities, assuming it's New Year's day, 2010
        //
        // "Today 18:30" = "2010-01-01 18:30:00",
        // "Yesterday 18:30" = "2009-12-31 18:30:00",
        // "Tomorrow 18:30" = "2010-01-02 18:30:00",
        // "Overmorrow 18:30" = "2010-01-03 18:30:00",
        // "2022-11-07 13:25:30" = "2022-11-07 13:25:30",
        // "15:20 Friday" = "2010-01-08 15:20:00",
        // "This Friday 17:00" = "2010-01-08 17:00:00",
        // "13:25, Next Tuesday" = "2010-01-12 13:25:00",
        // "Last Friday at 19:45" = "2009-12-25 19:45:00",
        // "Next week" = "2010-01-08 00:00:00",
        // "This week" = "2010-01-01 00:00:00",
        // "Last week" = "2009-12-25 00:00:00",
        // "Next week Monday" = "2010-01-04 00:00:00",
        // "This week Friday" = "2010-01-01 00:00:00",
        // "This week Monday" = "2009-12-28 00:00:00",
        // "Last week Tuesday" = "2009-12-22 00:00:00",
        // "In 3 days" = "2010-01-04 00:00:00",
        // "In 2 hours" = "2010-01-01 02:00:00",
        // "In 5 minutes and 30 seconds" = "2010-01-01 00:05:30",
        // "10 seconds ago" = "2009-12-31 23:59:50",
        // "10 hours and 5 minutes ago" = "2009-12-31 13:55:00",
        // "2 hours, 32 minutes and 7 seconds ago" = "2009-12-31 21:27:53",
        // "1 years, 2 months, 3 weeks, 5 days, 8 hours, 17 minutes and 45 seconds ago" =
        //     "2008-10-07 16:42:15",
        // "1 year, 1 month, 1 week, 1 day, 1 hour, 1 minute and 1 second ago" = "2008-11-23 22:58:59",
        // "A year ago" = "2009-01-01 00:00:00",
        // "A month ago" = "2009-12-01 00:00:00",
        // "A week ago" = "2009-12-25 00:00:00",
        // "A day ago" = "2009-12-31 00:00:00",
        // "An hour ago" = "2009-12-31 23:00:00",
        // "A minute ago" = "2009-12-31 23:59:00",
        // "A second ago" = "2009-12-31 23:59:59",
        // "now" = "2010-01-01 00:00:00",
        // "Overmorrow" = "2010-01-03 00:00:00"

        Ok(())
    }

    /////////////////////////////////////////////////////////////////////////
    // Test helpers
    /////////////////////////////////////////////////////////////////////////

    fn where_clause(query: &str) -> Result<String, anyhow::Error> {
        Ok(advisory::Entity::find()
            .select_only()
            .column(advisory::Column::Id)
            .filtering(q(query))?
            .build(sea_orm::DatabaseBackend::Postgres)
            .to_string()[45..]
            .to_string())
    }

    mod advisory {
        use sea_orm::entity::prelude::*;
        use time::OffsetDateTime;

        #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
        #[sea_orm(table_name = "advisory")]
        pub struct Model {
            #[sea_orm(primary_key)]
            pub id: i32,
            pub location: String,
            pub title: String,
            pub published: Option<OffsetDateTime>,
        }
        #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
        pub enum Relation {}
        impl ActiveModelBehavior for ActiveModel {}
    }
}
