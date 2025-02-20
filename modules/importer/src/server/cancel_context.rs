use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Notify;

/// A context for managing cancellation in asynchronous operations.
///
/// This struct provides a mechanism to signal cancellation to multiple tasks.
/// It should be wrapped in an Arc to same cancellation state across different parts
/// of an application.
///
/// # Example
///
/// ```rust
/// use tokio::time::{sleep, Duration};
/// use std::sync::Arc;
/// use trustify_module_importer::server::cancel_context::CancelContext;
///
/// #[tokio::main]
/// async fn main() {
///
///     let cancel_ctx = Arc::new(CancelContext::new());
///     let cancel_ctx_clone = cancel_ctx.clone();
///
///     // Spawn a task to cancel after 1 second
///     tokio::spawn(async move {
///         sleep(Duration::from_secs(1)).await;
///         cancel_ctx_clone.cancel();
///     });
///
///     // Main task loops until canceled
///     loop {
///         tokio::select! {
///             _ = cancel_ctx.done() => {
///                 println!("Canceled");
///                 break;
///             }
///             _ = sleep(Duration::from_millis(100)) => {
///                 println!("Working...");
///             }
///         }
///     }
/// }
/// ```
#[derive(Debug)]
pub struct CancelContext {
    canceled: AtomicBool,
    done: Notify,
}

impl CancelContext {
    /// Creates a new `CancelContext` with cancellation initially unset.
    pub fn new() -> Self {
        Self {
            canceled: AtomicBool::new(false),
            done: Notify::new(),
        }
    }

    /// Checks if the context has been canceled.
    ///
    /// Returns `true` if `cancel()` has been called, `false` otherwise.
    pub fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::Acquire)
    }

    /// Cancels the context and notifies all waiting tasks.
    ///
    /// This sets the cancellation flag and wakes up all tasks awaiting `done()`.
    /// Calling this multiple times is safe and idempotent in terms of effect.
    pub fn cancel(&self) {
        self.canceled.store(true, Ordering::Release);
        self.done.notify_waiters();
    }

    /// Returns a future that completes when the context is canceled and forever thereafter.
    ///
    /// Tasks can await this future to be notified when `cancel()` is called.
    pub async fn done(&self) {
        // queue up the done notification
        let notified = self.done.notified();
        if !self.is_canceled() {
            // if we queued up the done notification on the await line, and
            // the cancel occurred here between if check and the await line, then we would
            // end up waiting forever since we were not in the done notification queue
            notified.await
        }
    }
}

impl Default for CancelContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::server::cancel_context::CancelContext;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_initial_state() {
        let ctx = CancelContext::new();
        assert!(!ctx.is_canceled());
    }

    #[tokio::test]
    async fn test_cancellation() {
        let ctx = CancelContext::new();
        ctx.cancel();
        assert!(ctx.is_canceled());
    }

    #[tokio::test]
    async fn test_notification() {
        let cancel = Arc::new(CancelContext::new());
        let cancel_clone = cancel.clone();

        let handle = tokio::spawn(async move {
            tokio::select! {
                _ = cancel_clone.done() => {
                    log::debug!("Canceled");
                }
            }
        });

        // Allow the task to start awaiting
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        cancel.cancel();

        // Ensure the task completes
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_immediate_trigger_after_cancellation() {
        let ctx = CancelContext::new();
        ctx.cancel();

        // Should complete immediately
        ctx.done().await;
    }
}
