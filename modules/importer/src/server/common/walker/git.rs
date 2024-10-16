use crate::server::common::walker::WorkingDirectory;
use anyhow::anyhow;
use git2::{
    build::RepoBuilder, ErrorClass, ErrorCode, FetchOptions, RemoteCallbacks, Repository, ResetType,
};
use std::{
    borrow::Cow,
    collections::HashSet,
    convert::Infallible,
    fmt::{Debug, Display},
    path::{Path, PathBuf},
};
use tokio::task::JoinError;
use tracing::{info_span, instrument};
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to await the task: {0}")]
    Join(#[from] JoinError),
    #[error("failed to create the working directory: {0}")]
    WorkingDir(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error(transparent)]
    Git(#[from] git2::Error),
    #[error("failed to walk files: {0}")]
    Walk(#[from] walkdir::Error),
    #[error("critical processing error: {0}")]
    Processing(#[source] anyhow::Error),
    #[error("{0} is not a relative subdirectory of the repository")]
    Path(String),
    #[error("operation canceled")]
    Canceled,
}

#[derive(Debug, thiserror::Error)]
pub enum HandlerError<T> {
    #[error(transparent)]
    Processing(T),
    #[error("operation canceled")]
    Canceled,
}

pub trait Handler: Send + 'static {
    type Error: Display + Debug;

    fn process(
        &mut self,
        path: &Path,
        relative_path: &Path,
    ) -> Result<(), HandlerError<Self::Error>>;
}

impl Handler for () {
    type Error = Infallible;

    fn process(&mut self, _: &Path, _: &Path) -> Result<(), HandlerError<Self::Error>> {
        Ok(())
    }
}

pub struct GitWalker<H, T>
where
    T: WorkingDirectory + Send + 'static,
    H: Handler,
{
    /// The git source to clone from
    pub source: String,

    /// A path inside the cloned repository to start searching for files
    pub path: Option<String>,

    /// Continuation token
    pub continuation: Continuation,

    /// A working directory
    pub working_dir: T,

    /// The handler
    pub handler: H,
}

impl<H> GitWalker<H, ()>
where
    H: Handler,
{
    pub fn new(source: impl Into<String>, handler: H) -> Self {
        Self {
            source: source.into(),
            path: None,
            continuation: Default::default(),
            working_dir: (),
            handler,
        }
    }
}

impl<H, T> GitWalker<H, T>
where
    H: Handler,
    T: WorkingDirectory + Send + 'static,
{
    pub fn handler<U: Handler>(self, handler: U) -> GitWalker<U, T> {
        GitWalker {
            source: self.source,
            path: self.path,
            continuation: self.continuation,
            working_dir: self.working_dir,
            handler,
        }
    }

    /// Set a working directory.
    ///
    /// The data in this working directory will be re-used. However, it must be specific to the
    /// source used. It is not possible to re-use the same working-directory for multiple different
    /// sources.
    ///
    /// It may also be `()`, which uses a temporary working directory. However, this will result in
    /// the walker cloning the full repository with ever run, which might be quite expensive.
    pub fn working_dir<U: WorkingDirectory + Send + 'static>(
        self,
        working_dir: U,
    ) -> GitWalker<H, U> {
        GitWalker {
            source: self.source,
            path: self.path,
            continuation: self.continuation,
            working_dir,
            handler: self.handler,
        }
    }

    pub fn path(mut self, path: Option<impl Into<String>>) -> Self {
        self.path = path.map(|s| s.into());
        self
    }

    /// Set a continuation token from a previous run.
    pub fn continuation(mut self, continuation: Continuation) -> Self {
        self.continuation = continuation;
        self
    }

    /// Run the walker
    #[instrument(skip(self), ret)]
    pub async fn run(self) -> Result<Continuation, Error> {
        tokio::task::spawn_blocking(move || self.run_sync()).await?
    }

    /// Sync version, as all git functions are sync
    #[instrument(skip(self), ret)]
    fn run_sync(mut self) -> Result<Continuation, Error> {
        log::debug!("Starting run for: {}", self.source);

        let working_dir = self
            .working_dir
            .create()
            .map_err(|err| Error::WorkingDir(Box::new(err)))?;

        let path = working_dir.as_ref();

        log::info!("Cloning {} into {}", self.source, path.display());

        let mut cb = RemoteCallbacks::new();
        cb.transfer_progress(|progress| {
            let received = progress.received_objects();
            let total = progress.total_objects();
            let bytes = progress.received_bytes();

            log::trace!("Progress - objects: {received} of {total}, bytes: {bytes}");

            true
        });
        cb.update_tips(|refname, a, b| {
            if a.is_zero() {
                log::debug!("[new]     {:20} {}", b, refname);
            } else {
                log::debug!("[updated] {:10}..{:10} {}", a, b, refname);
            }
            true
        });

        let mut fo = FetchOptions::new();
        fo.remote_callbacks(cb);

        // clone or open repository

        let result = info_span!("clone repository").in_scope(|| {
            RepoBuilder::new()
                .fetch_options(fo)
                .clone(&self.source, path)
        });

        let repo = match result {
            Ok(repo) => repo,
            Err(err) if err.code() == ErrorCode::Exists && err.class() == ErrorClass::Invalid => {
                log::info!("Already exists, opening ...");
                let repo = info_span!("open repository").in_scope(|| Repository::open(path))?;

                info_span!("fetching updates").in_scope(|| {
                    log::debug!("Fetching updates");
                    let mut remote = repo.find_remote("origin")?;
                    remote.fetch(&[] as &[&str], None, None)?;
                    remote.disconnect()?;

                    let head = repo.find_reference("FETCH_HEAD")?;
                    let head = head.peel_to_commit()?;

                    // reset to the most recent commit
                    repo.reset(head.as_object(), ResetType::Hard, None)?;

                    Ok::<_, Error>(())
                })?;

                repo
            }
            Err(err) => {
                log::info!(
                    "Clone failed - code: {:?}, class: {:?}",
                    err.code(),
                    err.class()
                );
                return Err(err.into());
            }
        };

        log::debug!("Repository cloned or updated");

        // discover files between "then" and now

        let changes = match &self.continuation.0 {
            Some(commit) => {
                log::info!("Continuing from: {commit}");

                let files = info_span!("continue from", commit).in_scope(|| {
                    let start = repo.find_commit(repo.revparse_single(commit)?.id())?;
                    let end = repo.head()?.peel_to_commit()?;

                    let start = start.tree()?;
                    let end = end.tree()?;

                    let diff = repo.diff_tree_to_tree(Some(&start), Some(&end), None)?;

                    let mut files = HashSet::with_capacity(diff.deltas().len());

                    for delta in diff.deltas() {
                        if let Some(path) = delta.new_file().path() {
                            let path = match &self.path {
                                // files are relative to the base dir
                                Some(base) => match path.strip_prefix(base) {
                                    Ok(path) => Some(path.to_path_buf()),
                                    Err(..) => None,
                                },
                                // files are relative to the repo
                                None => Some(path.to_path_buf()),
                            };

                            if let Some(path) = path {
                                log::debug!("Record {} as changed file", path.display());
                                files.insert(path);
                            }
                        }
                    }

                    Ok::<_, Error>(files)
                })?;

                log::info!("Detected {} changed files", files.len());

                Some(files)
            }
            _ => {
                log::debug!("Ingesting all files");
                None
            }
        };

        // discover and process files

        let mut path = Cow::Borrowed(path);
        if let Some(base) = &self.path {
            let new_path = path.join(base);

            log::debug!("  Base: {}", path.display());
            log::debug!("Target: {}", new_path.display());

            // ensure that self.path was a relative sub-directory of the repository
            let _ = new_path
                .strip_prefix(path)
                .map_err(|_| Error::Path(base.into()))?;

            path = new_path.into();
        }

        self.walk(&path, &changes)?;

        let head = repo.head()?;
        let commit = head.peel_to_commit()?.id();
        log::info!("Most recent commit: {commit}");

        // only drop when we are done, as this might delete the working directory

        drop(working_dir);

        // return result

        Ok(Continuation(Some(commit.to_string())))
    }

    #[instrument(skip(self, changes), err)]
    fn walk(&mut self, base: &Path, changes: &Option<HashSet<PathBuf>>) -> Result<(), Error> {
        for entry in WalkDir::new(base)
            .into_iter()
            .filter_entry(|entry| !is_hidden(entry))
        {
            let entry = entry?;

            log::trace!("Checking: {entry:?}");

            if !entry.file_type().is_file() {
                continue;
            }

            // the path in the filesystem
            let path = entry.path();
            // the path, relative to the base (plus repo) dir
            let path = path.strip_prefix(base).unwrap_or(path);

            if let Some(changes) = changes {
                if !changes.contains(path) {
                    log::trace!("Skipping {}, as file did not change", path.display());
                    continue;
                }
            }

            self.handler
                .process(entry.path(), path)
                .map_err(|err| match err {
                    HandlerError::Canceled => Error::Canceled,
                    HandlerError::Processing(err) => Error::Processing(anyhow!("{err}")),
                })?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Continuation(Option<String>);

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with('.'))
        .unwrap_or(false)
}

#[cfg(test)]
pub(crate) use test::git_reset;

#[cfg(test)]
mod test {
    use super::Continuation;
    use git2::{Repository, ResetType};
    use std::path::Path;

    /// reset a git repository to the spec and return the commit as continuation
    pub(crate) fn git_reset(path: &Path, spec: &str) -> anyhow::Result<Continuation> {
        let repo = Repository::open(path)?;

        let r#ref = repo.revparse_single(spec)?;
        repo.reset(&r#ref, ResetType::Hard, None)?;

        let commit = r#ref.peel_to_commit()?.id().to_string();

        Ok(Continuation(Some(commit)))
    }
}
