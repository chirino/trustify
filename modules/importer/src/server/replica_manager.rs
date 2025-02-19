use std::hash::{Hasher, Hash, DefaultHasher};
use sea_orm::{entity::*, query::*, DbErr};
use sea_query::OnConflict;
use std::time::Duration;
use time::OffsetDateTime;
use tokio::sync::broadcast;
use trustify_common::db::Database;
use trustify_entity::replica;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ReplicaManager {
    db: Database,
    replica_id: Uuid,
    interval: Duration,
    sender: broadcast::Sender<MembershipEvent>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MembershipEvent {
    pub replica_id: Uuid,
    pub members: Vec<Uuid>,
}

fn calculate_hash<T: Hash>(t: &str) -> usize {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish() as usize
}

impl MembershipEvent {
    pub fn is_leader(&self) -> bool {
        // return true if self.replica_id is the first element in self.members
        self.members.first().map_or(false, |id| *id == self.replica_id)
    }
    pub fn is_leader_for(&self, key : &str) -> bool {
        if let Some(index) = self.members.iter().position(|x| x == &self.replica_id) {
            calculate_hash(key) % self.members.len() == index
        } else {
            false
        }
    }
}

impl ReplicaManager {
    pub fn new(db: Database, interval: Duration) -> Self {
        let (sender, _) = broadcast::channel(10);
        ReplicaManager {
            db,
            replica_id: Uuid::now_v7(),
            interval,
            sender,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<MembershipEvent> {
        self.sender.subscribe()
    }

    async fn register_replica(&self) -> Result<(), DbErr> {
        let now = OffsetDateTime::now_utc();
        let model = replica::ActiveModel {
            id: Set(self.replica_id),
            created_at: Set(now),
            last_seen: Set(now),
        };
        replica::Entity::insert(model)
            .on_conflict(
                OnConflict::column(replica::Column::Id)
                    .update_column(replica::Column::LastSeen)
                    .to_owned(),
            )
            .exec(&self.db)
            .await?;
        Ok(())
    }

    pub async fn get_active_replicas(&self) -> Result<Vec<Uuid>, DbErr> {
        let deadline = OffsetDateTime::now_utc() - (self.interval * 3);
        let replicas = replica::Entity::find()
            .filter(replica::Column::LastSeen.gt(deadline))
            .order_by_asc(replica::Column::CreatedAt)
            .order_by_asc(replica::Column::Id)
            .all(&self.db)
            .await?;
        Ok(replicas.into_iter().map(|r| r.id).collect())
    }

    pub async fn prune_stale_replicas(&self) -> Result<(), DbErr> {
        let deadline = OffsetDateTime::now_utc() - (self.interval * 10);
        replica::Entity::delete_many()
            .filter(replica::Column::LastSeen.lt(deadline))
            .exec(&self.db)
            .await?;
        Ok(())
    }

    async fn interval_work(&self) -> Result<Vec<Uuid>, DbErr> {
        self.register_replica().await?;
        let members = self.get_active_replicas().await?;
        if !members.contains(&self.replica_id) {
            return Err(DbErr::Custom("This replica not found in active replicas".to_string()));
        }
        self.prune_stale_replicas().await?;
        Ok(members)
    }

    pub async fn start(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(manager.interval);
            let mut last_members = None;

            loop {
                interval.tick().await;
                let new_members = manager.interval_work().await;
                match new_members {
                    Ok(members) => {
                        if last_members.as_ref() != Some(&members) {
                            last_members = Some(members.clone());
                            let _ = manager.sender.send(MembershipEvent {
                                replica_id: manager.replica_id,
                                members,
                            });
                        }
                    }
                    Err(e)=> {
                        log::error!("Failed to register replica: {:?}", e);
                        if last_members.is_some() {
                            last_members = None;
                            let _ = manager.sender.send(MembershipEvent {
                                replica_id: manager.replica_id,
                                members: vec![],
                            });
                        }
                    }
                }
            }
        });
    }

}
#[cfg(test)]
mod tests {
    use std::time::Duration;
    use test_context::test_context;
    use test_log::test;
    use trustify_test_context::TrustifyContext;

    use crate::server::replica_manager::{MembershipEvent, ReplicaManager};

    #[test_context(TrustifyContext, skip_teardown)]
    #[test(actix_web::test)]
    async fn default(ctx: TrustifyContext) {
        let interval = Duration::from_secs(1);
        let replica_manager_1 = ReplicaManager::new(ctx.db.clone(), interval);
        let mut sub1 = replica_manager_1.subscribe();
        replica_manager_1.start().await;

        println!("waiting for first event from replica_manager_1");
        let members = sub1.recv().await.unwrap();
        assert_eq!(members, MembershipEvent{
            replica_id: replica_manager_1.replica_id,
            members: vec![replica_manager_1.replica_id],
        });


        let replica_manager_2 = ReplicaManager::new(ctx.db.clone(), interval);
        let mut sub2 = replica_manager_2.subscribe();
        replica_manager_2.start().await;

        println!("waiting for first event from replica_manager_2");
        let members = sub2.recv().await.unwrap();
        assert_eq!(members, MembershipEvent{
            replica_id: replica_manager_2.replica_id,
            members: vec![replica_manager_1.replica_id, replica_manager_2.replica_id],
        });


        println!("waiting for 2nd event from replica_manager_1");
        let members = sub1.recv().await.unwrap();
        assert_eq!(members, MembershipEvent{
            replica_id: replica_manager_1.replica_id,
            members: vec![replica_manager_1.replica_id, replica_manager_2.replica_id],
        });

    }

}