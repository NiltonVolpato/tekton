//! The Tekton `type` tool — sends keystrokes to a persistent PTY session.
//!
//! This is the single LLM-facing tool in the Tekton framework. The model sends
//! keystrokes to a terminal the same way a human would: by typing. Everything
//! else — file I/O, web search, code execution, sub-agents — is a shell command.
//!
//! # Harness wiring
//!
//! After constructing the tool and before passing it to the agent builder,
//! the harness calls [`TypeTool::spawn_watcher`] to start a background task
//! that polls `sysinfo` for tracked PIDs. When a PID disappears from the
//! process table, [`JobManager::retain`] fires the completion callback.
//!
//! [`TypeTool::call`] handles **discovery** of new background jobs (via
//! `jobs -p`) and also detects completions. The watcher handles completions
//! while the tool is idle. Both paths are safe to run concurrently: `retain()`
//! never adds PIDs, so there is no risk of re-introducing a reaped PID.
//!
//! ```no_run
//! use std::sync::{Arc, Mutex};
//! use tekton_type_tool::TypeTool;
//! use rig::{
//!     agent::AgentBuilder,
//!     client::{CompletionClient, ProviderClient},
//!     providers::anthropic::{self, completion::CLAUDE_4_SONNET},
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     let anthropic = anthropic::Client::from_env();
//!     let model = anthropic.completion_model(CLAUDE_4_SONNET);
//!
//!     let (tool, _cwd) = TypeTool::spawn().await.unwrap();
//!     let tool = tool.with_job_callback(|notif| {
//!         // Queue a proactive message to the agent, wake it up if idle, etc.
//!         eprintln!("job {} done (exit {:?})", notif.pid, notif.exit_code);
//!     });
//!
//!     // Start the sysinfo watcher before moving `tool` into the agent builder.
//!     let watcher = tool.spawn_watcher();
//!
//!     let agent = AgentBuilder::new(model)
//!         .preamble(
//!             "You're looking at a terminal. The only tool you have is `type`, \
//!              which sends keystrokes to the terminal.\n\n\
//!              The terminal says:\n\n\
//!              Welcome back Claude! For help run the command `help`.\n\
//!              [0] claude@alpha:/Users/nilton/src/tekton $",
//!         )
//!         .tool(tool)
//!         .build();
//!
//!     // On shutdown: watcher.abort();
//! }
//! ```

mod session;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

/// How often the sysinfo watcher polls for completed background jobs.
const WATCHER_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Non-interactive timeout in seconds: how long to wait before killing the foreground process.
const DEFAULT_TIMEOUT_SECS: f64 = 300.0;

/// Interactive timeout in seconds: how long to wait before returning partial output to the model.
const DEFAULT_INTERACTIVE_TIMEOUT_SECS: f64 = 5.0;

/// Maximum allowed timeout in seconds (~24 hours).
///
/// Values above this are rejected with [`TypeError::InvalidTimeout`].
/// This prevents panics in [`Duration::from_secs_f64`] for very large finite
/// values (e.g., `f64::MAX` ≈ 1.8e308 overflows `Duration`'s internal `u64`).
const MAX_TIMEOUT_SECS: f64 = 86_400.0;

// ── Arguments & output ──────────────────────────────────────────────────────

/// Arguments for the `type` tool.
#[derive(Debug, Deserialize)]
pub struct TypeArgs {
    /// Keystrokes to send to the terminal.
    ///
    /// Supports escape sequences: `\n` for Enter, `\x03` for Ctrl-C, etc.
    pub keys: String,

    /// Timeout behavior on expiry.
    ///
    /// - `false` (default): kill the foreground process group and return output + error.
    /// - `true`: return partial output to the model so it can decide what to type next.
    #[serde(default)]
    pub interactive: bool,

    /// Seconds to wait before triggering the timeout behavior.
    ///
    /// Defaults to 300s when `interactive = false`, 5s when `interactive = true`.
    /// Can be overridden per call.
    pub timeout: Option<f64>,
}

impl TypeArgs {
    /// Resolved timeout: caller's value, or the mode-appropriate default.
    pub fn resolved_timeout(&self) -> f64 {
        self.timeout.unwrap_or(if self.interactive {
            DEFAULT_INTERACTIVE_TIMEOUT_SECS
        } else {
            DEFAULT_TIMEOUT_SECS
        })
    }
}

/// Output returned to the model after a `type` call.
#[derive(Debug, Serialize)]
pub struct TypeOutput {
    /// Terminal output captured between the keystroke and the next prompt (or timeout).
    pub output: String,

    /// What happened after the keystrokes were sent.
    pub outcome: Outcome,
}

/// Describes how a `type` call concluded.
#[derive(Debug, Serialize)]
pub enum Outcome {
    /// The command ran to completion and the shell emitted a prompt sentinel.
    Completed {
        exit_code: i32,
        working_directory: String,
    },

    /// The command timed out (non-interactive) and was killed.
    /// The shell is back at a prompt and ready for the next command.
    Killed {
        working_directory: String,
        timeout_secs: f64,
    },

    /// The command timed out (interactive) and is still running.
    /// The foreground process was not killed — the caller can send more input.
    Waiting,

    /// The shell process exited (e.g. `exec`, `exit`, crash).
    /// A new session has been spawned and is ready.
    ShellExited {
        working_directory: String,
    },
}

// ── Job tracking ─────────────────────────────────────────────────────────────

/// A tracked background job.
#[derive(Debug, Clone)]
pub struct Job {
    /// PID of the background process.
    pub pid: u32,
    /// The command string. Empty when the job was discovered via `jobs -p` alone;
    /// populated when richer output (e.g., `jobs -l`) is available.
    pub command: String,
}

/// A background job completion event, delivered to the [`JobManager`] callback.
#[derive(Debug, Clone)]
pub struct JobNotification {
    /// PID of the completed background process.
    pub pid: u32,
    /// Exit code returned by the process (`None` when not determinable from `jobs -p`).
    pub exit_code: Option<i32>,
    /// The command string supplied when the job was first tracked.
    pub command: String,
}

/// Tracks running background jobs and detects completions.
///
/// Two update paths exist:
///
/// - [`JobManager::sync`] — called by [`TypeTool::call`] after each command.
///   Discovers new PIDs (from `jobs -p`) and removes completed ones.
/// - [`JobManager::retain`] — called by the sysinfo watcher task. Only removes
///   completed PIDs; never adds new ones. Safe to call concurrently with `sync`
///   (both are serialized by the `Mutex<JobManager>`).
pub struct JobManager {
    pub(crate) jobs: HashMap<u32, Job>,
    on_complete: Option<Box<dyn Fn(JobNotification) + Send + Sync>>,
}

impl JobManager {
    /// Create a new, empty `JobManager` with no callback.
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            on_complete: None,
        }
    }

    /// Register a callback invoked for each completed background job.
    pub fn with_callback(
        mut self,
        callback: impl Fn(JobNotification) + Send + Sync + 'static,
    ) -> Self {
        self.on_complete = Some(Box::new(callback));
        self
    }

    /// Sync tracked jobs against the current set of active background PIDs.
    ///
    /// - **PIDs in `active_pids` but not yet tracked**: added as new jobs
    ///   (command is unknown until richer shell output is available).
    /// - **PIDs tracked but absent from `active_pids`**: they have completed;
    ///   the callback is fired for each one and they are removed.
    ///
    /// This is the **discovery** path: use it from [`TypeTool::call`] where
    /// `jobs -p` provides the authoritative set of background PIDs, including
    /// newly started ones.
    ///
    /// For completion-only detection (e.g., the sysinfo watcher), use
    /// [`JobManager::retain`] instead — it never adds new PIDs, avoiding races
    /// where a stale active set could re-introduce a PID that was already reaped.
    pub fn sync(&mut self, active_pids: &HashSet<u32>) {
        self.remove_completed(active_pids);

        // Track newly appeared PIDs.
        for &pid in active_pids {
            self.jobs.entry(pid).or_insert_with(|| Job {
                pid,
                // TODO: populate from `jobs -l` or `ps` output for richer info.
                command: String::new(),
            });
        }
    }

    /// Retain only the tracked PIDs that are still alive.
    ///
    /// `checked` is the set of PIDs that were actually queried in sysinfo.
    /// `alive` is the subset of `checked` that sysinfo confirmed are still running.
    ///
    /// Only PIDs in `checked` are considered: a tracked PID that was never checked
    /// (e.g., added by `sync()` between the snapshot and this call) is left alone.
    ///
    /// This is the **completion-detection-only** path: use it from the sysinfo
    /// watcher task, which can determine whether a PID is alive but cannot
    /// discover new background jobs (that's `jobs -p`'s job inside `call()`).
    pub fn retain(&mut self, checked: &HashSet<u32>, alive: &HashSet<u32>) {
        let completed: Vec<u32> = self
            .jobs
            .keys()
            .filter(|pid| checked.contains(pid) && !alive.contains(pid))
            .copied()
            .collect();

        for pid in completed {
            if let Some(job) = self.jobs.remove(&pid)
                && let Some(cb) = &self.on_complete
            {
                cb(JobNotification {
                    pid,
                    // TODO: use waitpid(WNOHANG) to get the actual exit code.
                    exit_code: None,
                    command: job.command,
                });
            }
        }
    }

    /// Remove tracked PIDs that are absent from `active_pids` and fire the
    /// callback for each one.
    fn remove_completed(&mut self, active_pids: &HashSet<u32>) {
        let completed: Vec<u32> = self
            .jobs
            .keys()
            .filter(|pid| !active_pids.contains(pid))
            .copied()
            .collect();

        for pid in completed {
            if let Some(job) = self.jobs.remove(&pid)
                && let Some(cb) = &self.on_complete
            {
                cb(JobNotification {
                    pid,
                    // TODO: use waitpid(WNOHANG) to get the actual exit code.
                    exit_code: None,
                    command: job.command,
                });
            }
        }
    }
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Parse the output of `jobs -p` into a set of PIDs.
///
/// `jobs -p` prints one PID per line. Lines that cannot be parsed as integers
/// are silently skipped.
pub fn parse_jobs_p(output: &str) -> HashSet<u32> {
    output
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect()
}

// ── Tool ─────────────────────────────────────────────────────────────────────

/// Error type for `TypeTool` failures.
#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("PTY write failed: {0}")]
    PtyWrite(String),

    #[error("PTY read failed: {0}")]
    PtyRead(String),

    #[error("Session not initialized")]
    SessionNotInitialized,

    #[error("PTY spawn failed: {0}")]
    PtySpawn(String),

    #[error("Invalid timeout: {0}")]
    InvalidTimeout(f64),
}

/// The Tekton `type` tool.
///
/// Wraps a persistent PTY session. The model calls this tool to send keystrokes
/// to the terminal; the harness captures output until the shell emits a sentinel
/// (command complete) or the timeout fires.
///
/// After each command, `call` runs `jobs -p` internally and syncs the
/// [`JobManager`]. The sysinfo watcher task (see [`TypeTool::spawn_watcher`])
/// detects completions while the tool is idle using [`JobManager::retain`].
pub struct TypeTool {
    /// The persistent PTY session. `None` when constructed via [`TypeTool::new`].
    session: Arc<tokio::sync::Mutex<Option<session::PtySession>>>,

    /// Shared with the watcher task for background job lifecycle tracking.
    job_manager: Arc<Mutex<JobManager>>,
}

impl TypeTool {
    /// Create a new `TypeTool` with no session, an empty [`JobManager`], and no callback.
    ///
    /// Calling `call()` on a tool created this way returns
    /// [`TypeError::SessionNotInitialized`]. Use [`TypeTool::spawn`] to get a
    /// fully initialized tool with an active PTY session.
    pub fn new() -> Self {
        Self {
            session: Arc::new(tokio::sync::Mutex::new(None)),
            job_manager: Arc::new(Mutex::new(JobManager::new())),
        }
    }

    /// Spawn a bash PTY session and return a fully initialized `TypeTool`
    /// along with the shell's initial working directory.
    pub async fn spawn() -> Result<(Self, String), TypeError> {
        let session = session::PtySession::spawn().await?;
        let working_directory = session.working_directory.clone();
        let tool = Self {
            session: Arc::new(tokio::sync::Mutex::new(Some(session))),
            job_manager: Arc::new(Mutex::new(JobManager::new())),
        };
        Ok((tool, working_directory))
    }

    /// Register a callback invoked whenever a background job completes.
    ///
    /// Must be called **before** [`TypeTool::job_manager`] is cloned.
    ///
    /// Delegates to [`JobManager::with_callback`].
    pub fn with_job_callback(
        self,
        callback: impl Fn(JobNotification) + Send + Sync + 'static,
    ) -> Self {
        let manager = Arc::try_unwrap(self.job_manager)
            .unwrap_or_else(|_| panic!("job_manager Arc should have exactly one owner at construction time"))
            .into_inner()
            .unwrap_or_else(|_| panic!("job_manager Mutex should not be poisoned at construction time"))
            .with_callback(callback);
        Self {
            session: self.session,
            job_manager: Arc::new(Mutex::new(manager)),
        }
    }

    /// Return a shared handle to the [`JobManager`].
    ///
    /// The harness task uses this to call [`JobManager::sync`] on the idle path.
    pub fn job_manager(&self) -> Arc<Mutex<JobManager>> {
        Arc::clone(&self.job_manager)
    }

    /// Spawn an async watcher task that polls `sysinfo` for tracked PIDs.
    ///
    /// Every [`WATCHER_POLL_INTERVAL`], the task refreshes only the tracked PIDs
    /// in the system process table. PIDs that have disappeared are passed to
    /// [`JobManager::retain`], which fires the completion callback and removes
    /// them — without ever adding new PIDs (that's `call()`'s job via `jobs -p`).
    ///
    /// Returns a [`JoinHandle`](tokio::task::JoinHandle) the harness can
    /// `.abort()` on shutdown.
    pub fn spawn_watcher(&self) -> tokio::task::JoinHandle<()> {
        let job_manager = Arc::clone(&self.job_manager);
        tokio::spawn(async move {
            let mut sys = sysinfo::System::new();
            loop {
                tokio::time::sleep(WATCHER_POLL_INTERVAL).await;

                let tracked: Vec<sysinfo::Pid> = {
                    let jm = job_manager.lock().unwrap();
                    if jm.jobs.is_empty() {
                        continue;
                    }
                    jm.jobs
                        .keys()
                        .map(|&p| sysinfo::Pid::from_u32(p))
                        .collect()
                };

                // Refresh only the specific PIDs — no full system scan.
                // remove_dead_processes=true so dead PIDs are pruned from
                // sysinfo's internal map, making sys.process() return None.
                sys.refresh_processes_specifics(
                    sysinfo::ProcessesToUpdate::Some(&tracked),
                    true,
                    sysinfo::ProcessRefreshKind::nothing(),
                );

                let checked: HashSet<u32> = tracked.iter().map(|pid| pid.as_u32()).collect();
                let alive: HashSet<u32> = tracked
                    .iter()
                    .filter(|pid| sys.process(**pid).is_some())
                    .map(|pid| pid.as_u32())
                    .collect();

                job_manager.lock().unwrap().retain(&checked, &alive);
            }
        })
    }
}

impl Default for TypeTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for TypeTool {
    const NAME: &'static str = "type";

    type Error = TypeError;
    type Args = TypeArgs;
    type Output = TypeOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Send keystrokes to the terminal. Supports escape sequences: \\n for Enter, \
                 \\x03 for Ctrl-C, etc. Returns the terminal output captured until the next \
                 shell prompt or timeout."
                    .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Keystrokes to send. Supports escape sequences: \\n for Enter, \\x03 for Ctrl-C, \\x04 for Ctrl-D, etc."
                    },
                    "interactive": {
                        "type": "boolean",
                        "description": "Timeout behavior. false (default): kill the foreground process on timeout. true: return partial output so you can decide what to type next.",
                        "default": false
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Seconds to wait before triggering the timeout behavior. Defaults to 300 when interactive=false, 5 when interactive=true."
                    }
                },
                "required": ["keys"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let raw_timeout = args.resolved_timeout();
        if raw_timeout.is_nan() || raw_timeout.is_infinite() || raw_timeout < 0.0
            || raw_timeout > MAX_TIMEOUT_SECS
        {
            return Err(TypeError::InvalidTimeout(raw_timeout));
        }
        let timeout = Duration::from_secs_f64(raw_timeout);
        let keys = args.keys.clone();
        let interactive = args.interactive;
        let background_pids: HashSet<u32> = self
            .job_manager
            .lock()
            .unwrap()
            .jobs
            .keys()
            .copied()
            .collect();

        // Acquire session as an OwnedMutexGuard so it can be sent to spawn_blocking.
        let guard = Arc::clone(&self.session).lock_owned().await;

        // Fast-path: no session — return the error without spawning a thread.
        if guard.is_none() {
            return Err(TypeError::SessionNotInitialized);
        }

        // Run the blocking PTY I/O in a dedicated thread-pool thread.
        //
        // The active_pids slot is `Option<HashSet<u32>>`:
        //   - `Some(pids)` on a normal return or non-interactive timeout — sync the manager.
        //   - `None` on an interactive timeout or shell exit — the foreground process is
        //     still alive (Waiting) or gone (ShellExited), so we have no fresh `jobs -p`
        //     snapshot.
        let timeout_secs = timeout.as_secs_f64();
        let result = tokio::task::spawn_blocking(move || -> Result<_, TypeError> {
            let mut guard = guard;
            let pty = guard.as_mut().ok_or(TypeError::SessionNotInitialized)?;
            let shell_pid = pty.shell_pid;
            let s = &mut pty.session;

            // 1. Send keystrokes to the PTY.
            use expectrl::Expect;
            s.send(keys.as_bytes())
                .map_err(|e| TypeError::PtyWrite(e.to_string()))?;

            // 2. Wait for the prompt sentinel.
            match session::wait_for_sentinel(s, timeout) {
                Ok(captures) => {
                    let output = captures.before().to_vec();
                    let (exit_code, cwd) = session::parse_sentinel(&captures);
                    let active_pids = session::run_jobs_p_sync(s);
                    let outcome = Outcome::Completed {
                        exit_code,
                        working_directory: cwd,
                    };
                    Ok((output, outcome, Some(active_pids)))
                }

                Err(expectrl::Error::ExpectTimeout) if !interactive => {
                    // Drain partial output buffered by expect() before the timeout.
                    use std::io::BufRead;
                    let output = s.fill_buf()
                        .map(|b| b.to_vec())
                        .unwrap_or_default();
                    s.consume(output.len());

                    // Kill the foreground process (SIGTERM → SIGKILL).
                    session::kill_foreground(shell_pid, &background_pids);

                    // Drain PTY to restore a clean prompt; extract post-kill $PWD.
                    let cwd = session::drain_after_kill(s)
                        .unwrap_or_default();

                    let active_pids = session::run_jobs_p_sync(s);
                    let outcome = Outcome::Killed {
                        working_directory: cwd,
                        timeout_secs,
                    };
                    Ok((output, outcome, Some(active_pids)))
                }

                Err(expectrl::Error::ExpectTimeout) => {
                    // Interactive timeout: return partial output; process still alive.
                    // active_pids is None to suppress the post-call sync — we have no
                    // fresh jobs-p snapshot and syncing with an empty set would fire
                    // spurious completion callbacks for all tracked background jobs.
                    //
                    // expect() already read data into its internal buffer before
                    // timing out. Drain it via BufRead::fill_buf + consume.
                    use std::io::BufRead;
                    let output = s.fill_buf()
                        .map(|b| b.to_vec())
                        .unwrap_or_default();
                    s.consume(output.len());
                    Ok((output, Outcome::Waiting, None))
                }

                Err(expectrl::Error::Eof) => {
                    // Shell process exited (exec, exit, crash, etc.).
                    // Drain any buffered output before EOF.
                    use std::io::BufRead;
                    let output = s.fill_buf()
                        .map(|b| b.to_vec())
                        .unwrap_or_default();
                    s.consume(output.len());

                    // Respawn a fresh session so the next call() works.
                    let new_pty = session::PtySession::spawn_blocking_inner()?;
                    let cwd = new_pty.working_directory.clone();
                    *guard = Some(new_pty);

                    let outcome = Outcome::ShellExited {
                        working_directory: cwd,
                    };
                    Ok((output, outcome, None))
                }

                Err(e) => Err(TypeError::PtyRead(e.to_string())),
            }
        })
        .await
        .map_err(|e| TypeError::PtyRead(e.to_string()))?;

        let (output_bytes, outcome, active_pids) = result?;

        if let Some(active_pids) = active_pids {
            self.job_manager.lock().unwrap().sync(&active_pids);
        }

        Ok(TypeOutput {
            output: String::from_utf8_lossy(&output_bytes).into_owned(),
            outcome,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // TypeArgs / timeout resolution

    #[test]
    fn resolved_timeout_uses_default_for_non_interactive() {
        let args = TypeArgs {
            keys: "ls\n".to_string(),
            interactive: false,
            timeout: None,
        };
        assert_eq!(args.resolved_timeout(), DEFAULT_TIMEOUT_SECS);
    }

    #[test]
    fn resolved_timeout_uses_default_for_interactive() {
        let args = TypeArgs {
            keys: "y\n".to_string(),
            interactive: true,
            timeout: None,
        };
        assert_eq!(args.resolved_timeout(), DEFAULT_INTERACTIVE_TIMEOUT_SECS);
    }

    #[test]
    fn resolved_timeout_respects_caller_override() {
        let args = TypeArgs {
            keys: "analyze dataset.csv\n".to_string(),
            interactive: true,
            timeout: Some(30.0),
        };
        assert_eq!(args.resolved_timeout(), 30.0);
    }

    // parse_jobs_p

    #[test]
    fn parse_jobs_p_handles_typical_output() {
        let output = "12345\n67890\n";
        let pids = parse_jobs_p(output);
        assert!(pids.contains(&12345));
        assert!(pids.contains(&67890));
        assert_eq!(pids.len(), 2);
    }

    #[test]
    fn parse_jobs_p_ignores_empty_lines_and_garbage() {
        let output = "\n12345\n  \nbad\n67890\n";
        let pids = parse_jobs_p(output);
        assert_eq!(pids, HashSet::from([12345, 67890]));
    }

    #[test]
    fn parse_jobs_p_returns_empty_for_no_jobs() {
        assert!(parse_jobs_p("").is_empty());
    }

    // JobManager::sync

    #[test]
    fn sync_adds_new_pids() {
        let mut manager = JobManager::new();
        manager.sync(&HashSet::from([1, 2]));
        assert!(manager.jobs.contains_key(&1));
        assert!(manager.jobs.contains_key(&2));
    }

    #[test]
    fn sync_fires_callback_for_completed_jobs() {
        let completed = Arc::new(Mutex::new(Vec::new()));
        let completed_clone = Arc::clone(&completed);

        let mut manager = JobManager::new().with_callback(move |notif| {
            completed_clone.lock().unwrap().push(notif.pid);
        });

        // Job 1 is running.
        manager.sync(&HashSet::from([1]));
        // Job 1 has finished; job 2 has started.
        manager.sync(&HashSet::from([2]));

        let fired = completed.lock().unwrap();
        assert_eq!(*fired, vec![1]);
        assert!(manager.jobs.contains_key(&2));
        assert!(!manager.jobs.contains_key(&1));
    }

    #[test]
    fn sync_with_unchanged_pids_fires_no_callbacks() {
        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);

        let mut manager = JobManager::new().with_callback(move |_| {
            *fired_clone.lock().unwrap() = true;
        });

        manager.sync(&HashSet::from([42]));
        manager.sync(&HashSet::from([42])); // still running

        assert!(!*fired.lock().unwrap());
    }

    // Tool definition

    #[tokio::test]
    async fn tool_definition_has_correct_name() {
        let tool = TypeTool::new();
        assert_eq!(tool.definition(String::new()).await.name, "type");
    }

    #[tokio::test]
    async fn tool_definition_requires_keys() {
        let tool = TypeTool::new();
        let def = tool.definition(String::new()).await;
        let required = def.parameters["required"]
            .as_array()
            .expect("required should be an array");
        assert!(required.iter().any(|v| v == "keys"));
    }

    #[tokio::test]
    async fn call_returns_error_when_session_not_initialized() {
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs {
                keys: "ls\n".to_string(),
                interactive: false,
                timeout: None,
            })
            .await;
        assert!(matches!(result, Err(TypeError::SessionNotInitialized)));
    }

    // Shared handles

    #[test]
    fn job_manager_arc_is_shared() {
        let tool = TypeTool::new();
        assert!(Arc::ptr_eq(&tool.job_manager(), &tool.job_manager()));
    }

    // ── TypeArgs: additional boundary-value tests ──────────────────────────

    #[test]
    fn resolved_timeout_explicit_zero_is_honored() {
        // Zero is a valid (degenerate) timeout. The resolver must not substitute
        // the default — a bug might treat Some(0.0) like None via `unwrap_or`.
        let args = TypeArgs { keys: "ls\n".into(), interactive: false, timeout: Some(0.0) };
        assert_eq!(args.resolved_timeout(), 0.0);
    }

    #[test]
    fn resolved_timeout_negative_value_passed_through() {
        // Validation of the timeout range is the harness's job, not the resolver's.
        // The resolver must faithfully return whatever the caller supplied, even
        // nonsensical values like -1.
        let args = TypeArgs { keys: "ls\n".into(), interactive: false, timeout: Some(-1.0) };
        assert_eq!(args.resolved_timeout(), -1.0);
    }

    #[test]
    fn resolved_timeout_infinity_passed_through() {
        // Infinite timeout is useful for commands with no known upper bound.
        let args = TypeArgs {
            keys: "ls\n".into(),
            interactive: false,
            timeout: Some(f64::INFINITY),
        };
        assert_eq!(args.resolved_timeout(), f64::INFINITY);
    }

    #[test]
    fn resolved_timeout_nan_is_passed_through_not_swapped_for_default() {
        // NaN is pathological. The resolver must not silently swap it for the
        // default — that would hide a caller bug. Validation is the harness's job.
        let args = TypeArgs {
            keys: "ls\n".into(),
            interactive: false,
            timeout: Some(f64::NAN),
        };
        assert!(args.resolved_timeout().is_nan());
    }

    #[test]
    fn resolved_timeout_override_applies_in_both_interactive_modes() {
        // The caller-supplied value wins regardless of the `interactive` flag.
        // A bug might only apply the override in one branch of the if/else.
        let non_interactive = TypeArgs { keys: String::new(), interactive: false, timeout: Some(42.0) };
        let interactive = TypeArgs { keys: String::new(), interactive: true, timeout: Some(42.0) };
        assert_eq!(non_interactive.resolved_timeout(), 42.0);
        assert_eq!(interactive.resolved_timeout(), 42.0);
    }

    // ── parse_jobs_p: adversarial inputs ──────────────────────────────────

    #[test]
    fn parse_jobs_p_skips_negative_numbers() {
        // Negative integers fall outside u32's range; parse::<u32>() must reject
        // them. A bug would use parse::<i32>() then as u32, wrapping the value.
        let pids = parse_jobs_p("-1\n12345\n");
        assert!(pids.contains(&12345));
        assert_eq!(pids.len(), 1, "negative number must be silently skipped");
    }

    #[test]
    fn parse_jobs_p_skips_u32_overflow() {
        // 2^32 = 4294967296 overflows u32. A bug using i64→u32 truncating cast
        // would let it through as 0.
        let pids = parse_jobs_p("4294967296\n1234\n");
        assert!(pids.contains(&1234));
        assert_eq!(pids.len(), 1, "overflowing number must be silently skipped");
    }

    #[test]
    fn parse_jobs_p_accepts_u32_max() {
        // u32::MAX = 4294967295 is a valid u32 and must be accepted.
        // (Not a realistic PID, but the parser should not impose arbitrary limits.)
        let output = format!("{}\n", u32::MAX);
        let pids = parse_jobs_p(&output);
        assert!(pids.contains(&u32::MAX));
    }

    #[test]
    fn parse_jobs_p_skips_floating_point_numbers() {
        // "123.45" looks numeric but parse::<u32>() must reject it.
        // A bug might parse via f64 then truncate to u32.
        let pids = parse_jobs_p("123.45\n999\n");
        assert!(pids.contains(&999));
        assert_eq!(pids.len(), 1, "floating-point string must be silently skipped");
    }

    #[test]
    fn parse_jobs_p_skips_hex_strings() {
        // Default parse::<u32>() uses base 10 only; "0x…" must be skipped.
        // A bug might call u32::from_str_radix or accept hex implicitly.
        let pids = parse_jobs_p("0xDEAD\n5678\n");
        assert!(pids.contains(&5678));
        assert_eq!(pids.len(), 1, "hex string must be silently skipped");
    }

    #[test]
    fn parse_jobs_p_deduplicates_repeated_pids() {
        // `jobs -p` should never repeat a PID, but if it does the result is
        // still a set with exactly one entry for that PID.
        let pids = parse_jobs_p("1234\n1234\n");
        assert_eq!(pids.len(), 1);
        assert!(pids.contains(&1234));
    }

    #[test]
    fn parse_jobs_p_handles_crlf_line_endings() {
        // A misconfigured PTY may emit CRLF. str::lines() splits on \r\n and
        // trim() removes stray \r. Both PIDs should be parsed correctly.
        let pids = parse_jobs_p("12345\r\n67890\r\n");
        assert!(pids.contains(&12345));
        assert!(pids.contains(&67890));
        assert_eq!(pids.len(), 2);
    }

    #[test]
    fn parse_jobs_p_handles_no_trailing_newline() {
        // `jobs -p` may omit the trailing newline on the last line.
        // The final PID must still be parsed.
        let pids = parse_jobs_p("12345");
        assert!(pids.contains(&12345));
        assert_eq!(pids.len(), 1);
    }

    #[test]
    fn parse_jobs_p_returns_empty_for_whitespace_only_input() {
        // Pure whitespace contains no valid PIDs.
        assert!(parse_jobs_p("   \n\t\n  ").is_empty());
    }

    #[test]
    fn parse_jobs_p_includes_pid_zero_when_present() {
        // PID 0 is a valid u32 and parse::<u32>() accepts it.
        // Whether it is a meaningful job PID is a higher-level concern.
        let pids = parse_jobs_p("0\n1\n");
        assert!(pids.contains(&0));
        assert!(pids.contains(&1));
    }

    // ── JobManager: additional state-machine tests ─────────────────────────

    #[test]
    fn sync_with_empty_set_when_no_jobs_does_not_fire_callback() {
        // Syncing against an empty active set when nothing is tracked is a no-op.
        // A bug might fire spurious callbacks or panic on an empty map.
        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);
        let mut manager = JobManager::new().with_callback(move |_| {
            *fired_clone.lock().unwrap() = true;
        });
        manager.sync(&HashSet::new());
        assert!(!*fired.lock().unwrap(), "no callback should fire for empty→empty transition");
    }

    #[test]
    fn sync_fires_callback_for_every_simultaneously_completed_job() {
        // When multiple jobs vanish between two syncs, each must produce exactly
        // one notification. A bug might only fire for the first or last PID found.
        let completed = Arc::new(Mutex::new(Vec::new()));
        let cc = Arc::clone(&completed);
        let mut manager = JobManager::new().with_callback(move |n| {
            cc.lock().unwrap().push(n.pid);
        });
        manager.sync(&HashSet::from([10, 20, 30]));
        manager.sync(&HashSet::new()); // all three complete simultaneously
        let mut fired = completed.lock().unwrap().clone();
        fired.sort(); // HashMap iteration order is non-deterministic
        assert_eq!(fired, vec![10, 20, 30]);
    }

    #[test]
    fn sync_does_not_fire_callback_for_still_running_jobs() {
        // Only departed PIDs trigger notifications. Surviving jobs must not.
        let completed = Arc::new(Mutex::new(Vec::new()));
        let cc = Arc::clone(&completed);
        let mut manager = JobManager::new().with_callback(move |n| {
            cc.lock().unwrap().push(n.pid);
        });
        manager.sync(&HashSet::from([1, 2, 3]));
        manager.sync(&HashSet::from([2, 3])); // only PID 1 completed
        let fired = completed.lock().unwrap().clone();
        assert_eq!(fired, vec![1]);
        assert!(manager.jobs.contains_key(&2), "PID 2 must remain tracked");
        assert!(manager.jobs.contains_key(&3), "PID 3 must remain tracked");
    }

    #[test]
    fn sync_without_callback_silently_removes_completed_jobs() {
        // No callback registered: a completed job must be quietly removed, not
        // kept in the map indefinitely or cause a panic.
        let mut manager = JobManager::new();
        manager.sync(&HashSet::from([99]));
        manager.sync(&HashSet::new()); // 99 completes
        assert!(
            manager.jobs.is_empty(),
            "completed job must be removed from the map even when no callback is registered"
        );
    }

    #[test]
    fn sync_handles_pid_reuse_fires_callback_each_lifecycle() {
        // The OS reuses PIDs. A PID that completed and was removed can
        // reappear for an entirely new process. Each appearance+disappearance
        // must produce exactly one notification — not zero, not two.
        let completed = Arc::new(Mutex::new(Vec::new()));
        let cc = Arc::clone(&completed);
        let mut manager = JobManager::new().with_callback(move |n| {
            cc.lock().unwrap().push(n.pid);
        });
        manager.sync(&HashSet::from([42])); // first process with PID 42 starts
        manager.sync(&HashSet::new());      // first process completes → callback
        manager.sync(&HashSet::from([42])); // PID 42 reused by a new process
        manager.sync(&HashSet::new());      // second process completes → callback
        let fired = completed.lock().unwrap().clone();
        assert_eq!(
            fired,
            vec![42, 42],
            "callback must fire once per lifecycle, including across PID reuse"
        );
    }

    #[test]
    fn sync_notification_preserves_backfilled_command_field() {
        // The harness may backfill the `command` field after the initial sync
        // (e.g., from `jobs -l` or `ps` output). The notification must carry
        // that value, not an empty string from the initial or_insert_with.
        let received = Arc::new(Mutex::new(String::new()));
        let rc = Arc::clone(&received);
        let mut manager = JobManager::new().with_callback(move |n| {
            *rc.lock().unwrap() = n.command.clone();
        });
        manager.sync(&HashSet::from([7]));
        // Simulate the harness backfilling richer info (design §"Background Jobs").
        manager.jobs.get_mut(&7).unwrap().command = "long_task".to_string();
        manager.sync(&HashSet::new()); // job 7 completes
        assert_eq!(*received.lock().unwrap(), "long_task");
    }

    #[test]
    fn sync_notification_exit_code_is_none_for_jobs_p_detection() {
        // Per the design doc: exit_code is always None when completion is
        // detected by diffing `jobs -p` output (the harness has not yet called
        // waitpid). A bug would default it to Some(0).
        let received_exit_code = Arc::new(Mutex::new(Some(-999_i32)));
        let rc = Arc::clone(&received_exit_code);
        let mut manager = JobManager::new().with_callback(move |n| {
            *rc.lock().unwrap() = n.exit_code;
        });
        manager.sync(&HashSet::from([5]));
        manager.sync(&HashSet::new()); // completes
        assert_eq!(
            *received_exit_code.lock().unwrap(),
            None,
            "exit_code must be None when completion is detected via jobs-p diff, not waitpid"
        );
    }

    // ── JobManager::retain ─────────────────────────────────────────────────

    #[test]
    fn retain_removes_completed_jobs() {
        let completed = Arc::new(Mutex::new(Vec::new()));
        let cc = Arc::clone(&completed);
        let mut manager = JobManager::new().with_callback(move |n| {
            cc.lock().unwrap().push(n.pid);
        });
        manager.sync(&HashSet::from([1, 2, 3])); // discover three jobs
        manager.retain(&HashSet::from([1, 2, 3]), &HashSet::from([1, 3])); // checked all three; PID 2 is gone
        let fired = completed.lock().unwrap().clone();
        assert_eq!(fired, vec![2]);
        assert!(manager.jobs.contains_key(&1));
        assert!(!manager.jobs.contains_key(&2));
        assert!(manager.jobs.contains_key(&3));
    }

    #[test]
    fn retain_does_not_add_new_pids() {
        let mut manager = JobManager::new();
        manager.sync(&HashSet::from([1])); // discover PID 1
        manager.retain(&HashSet::from([1]), &HashSet::from([1, 2])); // checked PID 1; PID 2 was never tracked
        assert!(manager.jobs.contains_key(&1));
        assert!(
            !manager.jobs.contains_key(&2),
            "retain must not add new PIDs; that's sync's job"
        );
    }

    #[test]
    fn retain_fires_callback_with_correct_data() {
        let received = Arc::new(Mutex::new(None));
        let rc = Arc::clone(&received);
        let mut manager = JobManager::new().with_callback(move |n| {
            *rc.lock().unwrap() = Some(n);
        });
        manager.sync(&HashSet::from([42]));
        manager.jobs.get_mut(&42).unwrap().command = "make build".to_string();
        manager.retain(&HashSet::from([42]), &HashSet::new()); // checked PID 42; it's gone
        let notif = received.lock().unwrap().take().expect("callback must fire");
        assert_eq!(notif.pid, 42);
        assert_eq!(notif.command, "make build");
        assert_eq!(notif.exit_code, None);
    }

    #[test]
    fn retain_no_op_when_all_alive() {
        let fired = Arc::new(Mutex::new(false));
        let fc = Arc::clone(&fired);
        let mut manager = JobManager::new().with_callback(move |_| {
            *fc.lock().unwrap() = true;
        });
        manager.sync(&HashSet::from([1, 2]));
        manager.retain(&HashSet::from([1, 2]), &HashSet::from([1, 2])); // checked both; both alive
        assert!(!*fired.lock().unwrap(), "no callback should fire");
        assert_eq!(manager.jobs.len(), 2);
    }

    #[test]
    fn retain_empty_jobs_is_no_op() {
        let fired = Arc::new(Mutex::new(false));
        let fc = Arc::clone(&fired);
        let mut manager = JobManager::new().with_callback(move |_| {
            *fc.lock().unwrap() = true;
        });
        manager.retain(&HashSet::from([1, 2, 3]), &HashSet::from([1, 2, 3])); // nothing tracked
        assert!(!*fired.lock().unwrap());
        assert!(manager.jobs.is_empty());
    }

    // ── TypeTool: construction and shared-handle contract ──────────────────

    #[test]
    #[should_panic(expected = "job_manager Arc should have exactly one owner at construction time")]
    fn with_job_callback_panics_when_job_manager_arc_already_cloned() {
        // API contract: register the callback BEFORE calling job_manager().
        // If the harness already holds a clone of the Arc, with_job_callback
        // cannot take exclusive ownership and must panic to surface the misuse.
        let tool = TypeTool::new();
        let _shared = tool.job_manager(); // second owner created here
        let _tool = tool.with_job_callback(|_| {}); // must panic
    }

    #[test]
    fn with_job_callback_fires_when_registered_before_job_manager_clone() {
        // Happy-path ordering: callback first, then Arc shared with the harness.
        // Completions detected via the shared handle must reach the callback.
        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);
        let tool = TypeTool::new().with_job_callback(move |_| {
            *fired_clone.lock().unwrap() = true;
        });
        let jm = tool.job_manager();
        jm.lock().unwrap().sync(&HashSet::from([1]));
        jm.lock().unwrap().sync(&HashSet::new()); // triggers completion
        assert!(*fired.lock().unwrap());
    }

    // ── PTY integration: state-persistence tests ───────────────────────────

    #[tokio::test]
    async fn env_var_exported_in_one_call_is_visible_in_next_call() {
        // Core persistence contract: exported env vars must survive across tool
        // calls. A stateless fresh-shell implementation would fail this test.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        tool.call(TypeArgs {
            keys: "export TEKTON_PERSIST_TEST=hello\n".into(),
            interactive: false,
            timeout: None,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "echo $TEKTON_PERSIST_TEST\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        assert!(
            result.output.contains("hello"),
            "exported env var must persist across calls; got: {:?}", result.output
        );
    }

    #[tokio::test]
    async fn working_directory_persists_after_cd() {
        // `cd` must permanently change $PWD for all subsequent calls.
        // Regression guard: fresh-shell execution resets $PWD on every call.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        tool.call(TypeArgs {
            keys: "cd /tmp\n".into(),
            interactive: false,
            timeout: None,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "pwd\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        assert!(
            result.output.contains("/tmp"),
            "$PWD must persist across calls; got: {:?}", result.output
        );
        assert!(
            matches!(result.outcome, Outcome::Completed { ref working_directory, .. } if working_directory == "/tmp"),
            "working directory must be /tmp; got: {:?}", result.outcome
        );
    }

    #[tokio::test]
    async fn shell_function_defined_in_one_call_is_callable_in_next() {
        // Shell functions live in the session's environment. A function defined
        // in call N must be invocable in call N+1.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        tool.call(TypeArgs {
            keys: "tekton_greet() { echo \"greetings $1\"; }\n".into(),
            interactive: false,
            timeout: None,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "tekton_greet world\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        assert!(
            result.output.contains("greetings world"),
            "shell function must persist across calls; got: {:?}", result.output
        );
    }

    #[tokio::test]
    async fn non_exported_shell_variable_does_not_leak_to_subprocesses() {
        // Standard bash scoping: non-exported variables must not be visible to
        // child processes, even across calls within the same session.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        tool.call(TypeArgs {
            keys: "TEKTON_LOCAL=secret\n".into(),
            interactive: false,
            timeout: None,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "bash -c 'echo ${TEKTON_LOCAL:-UNSET}'\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        assert!(
            result.output.contains("UNSET"),
            "non-exported var must not be visible to subprocesses; got: {:?}", result.output
        );
    }

    // ── PTY integration: OSC sentinel and exit-code tests ─────────────────

    #[tokio::test]
    async fn exit_code_zero_for_successful_command() {
        // `true` always exits 0. The OSC sentinel must carry that value.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs { keys: "true\n".into(), interactive: false, timeout: None })
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 0, .. }));
    }

    #[tokio::test]
    async fn exit_code_one_for_failing_command() {
        // `false` always exits 1. Tests that non-zero codes are relayed.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs { keys: "false\n".into(), interactive: false, timeout: None })
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 1, .. }));
    }

    #[tokio::test]
    async fn exit_code_arbitrary_value_faithfully_captured_from_sentinel() {
        // The OSC sentinel must relay any exit code, not just 0/1.
        // A bug might mask the code by always returning 0 or 1.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "sh -c 'exit 42'\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 42, .. }));
    }

    #[tokio::test]
    async fn exit_code_127_for_unknown_command() {
        // Bash exits 127 on "command not found". Verifies the sentinel works
        // for error paths that never reach the program's own exit() call.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "tekton_nonexistent_command_xyz_abc\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 127, .. }));
    }

    #[tokio::test]
    async fn osc_sentinel_bytes_are_stripped_from_output() {
        // The sentinel `\e]999;EXIT\a` is machine metadata. It must not appear
        // in the output string returned to the model — only clean terminal text.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs { keys: "echo hello\n".into(), interactive: false, timeout: None })
            .await
            .unwrap();
        assert!(
            !result.output.contains("\x1b]999;"),
            "OSC sentinel bytes must be stripped from output; got: {:?}", result.output
        );
    }

    #[tokio::test]
    async fn keystroke_echo_disabled_so_typed_input_does_not_appear_in_output() {
        // Design: echo is disabled on the PTY. The model's keystrokes must not
        // be reflected back in the output it receives.
        //
        // If echo is ON, the input "echo marker_xyz\n" appears in the output
        // stream, making "marker_xyz" occur twice (once as echoed input, once
        // as command output). With echo OFF it occurs exactly once.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "echo tekton_echo_marker_xyz\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        let count = result.output.matches("tekton_echo_marker_xyz").count();
        assert_eq!(
            count, 1,
            "marker must appear exactly once (command output only, not echoed input); got: {:?}",
            result.output
        );
    }

    // ── PTY integration: timeout and process-control tests ─────────────────

    #[tokio::test]
    async fn non_interactive_timeout_kills_foreground_and_sets_timeout_message() {
        // Non-interactive timeout: kill the foreground process group, return
        // captured output. The outcome is Killed.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: false,
                timeout: Some(0.1), // 100 ms — much shorter than sleep
            })
            .await
            .unwrap();
        assert!(
            matches!(result.outcome, Outcome::Killed { .. }),
            "non-interactive timeout must result in Outcome::Killed; got: {:?}", result.outcome
        );
    }

    #[ignore = "BUG: Outcome::Killed discards post-kill exit code from drain_after_kill"]
    #[tokio::test]
    async fn non_interactive_timeout_killed_outcome_includes_exit_code() {
        // After kill_foreground sends SIGTERM/SIGKILL, bash regains control and
        // emits a sentinel whose exit code reflects the killed process (typically
        // 143 for SIGTERM or 137 for SIGKILL). drain_after_kill reads that
        // sentinel but currently discards the exit code, returning only the cwd.
        //
        // Correct behavior: Outcome::Killed should include the post-kill exit
        // code so the model can distinguish "timed out and killed" from other
        // failure modes. The exit code is already available in the sentinel —
        // drain_after_kill just needs to return it, and Outcome::Killed needs
        // an exit_code field.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: false,
                timeout: Some(0.1),
            })
            .await
            .unwrap();
        match result.outcome {
            // TODO: Change to `Outcome::Killed { exit_code, .. }` once the field exists.
            Outcome::Killed { .. } => {
                // SIGTERM = 143, SIGKILL = 137. Either is acceptable.
                // assert!(
                //     exit_code == 143 || exit_code == 137,
                //     "killed process exit code must be 143 (SIGTERM) or 137 (SIGKILL); got: {exit_code}"
                // );
                panic!(
                    "Outcome::Killed does not yet have an exit_code field. \
                     Add it, then uncomment the assertion above."
                );
            }
            other => panic!("expected Outcome::Killed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn interactive_timeout_returns_partial_output_without_killing_process() {
        // Interactive timeout: return partial output so the model can decide
        // what to type next. The process must remain alive.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        // `cat` with no args blocks on stdin — ideal for testing interactive mode.
        let partial = tool
            .call(TypeArgs {
                keys: "cat\n".into(),
                interactive: true,
                timeout: Some(0.1),
            })
            .await
            .unwrap();
        // No sentinel yet — cat is still running (Waiting state).
        assert!(
            matches!(partial.outcome, Outcome::Waiting),
            "interactive timeout must return Outcome::Waiting (process is still running)"
        );
        // Clean up: Ctrl-D (EOF) so cat exits and the shell returns to prompt.
        let cleanup = tool
            .call(TypeArgs {
                keys: "\x04".into(), // Ctrl-D
                interactive: false,
                timeout: Some(2.0),
            })
            .await
            .unwrap();
        assert!(
            matches!(cleanup.outcome, Outcome::Completed { exit_code: 0, .. }),
            "cat must exit cleanly after receiving EOF via Ctrl-D"
        );
    }

    #[tokio::test]
    async fn ctrl_c_sends_sigint_terminating_foreground_process_group() {
        // Ctrl-C (\x03) must terminate the foreground process so the shell
        // returns to the prompt and emits an OSC sentinel.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        // Start a long-running process in interactive mode, let it time out so
        // we get partial output quickly.
        let _ = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: true,
                timeout: Some(0.1),
            })
            .await;
        // Send Ctrl-C to kill the still-running sleep.
        let result = tool
            .call(TypeArgs {
                keys: "\x03".into(),
                interactive: false,
                timeout: Some(2.0),
            })
            .await
            .unwrap();
        assert!(
            matches!(result.outcome, Outcome::Completed { .. }),
            "Ctrl-C must terminate the foreground process and produce a sentinel; got: {:?}", result.outcome
        );
    }

    // ── Invalid timeout returns Err(InvalidTimeout) ───────────────────────

    #[tokio::test]
    async fn call_with_nan_timeout_returns_error() {
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs { keys: "echo hi\n".into(), interactive: false, timeout: Some(f64::NAN) })
            .await;
        assert!(matches!(result, Err(TypeError::InvalidTimeout(t)) if t.is_nan()));
    }

    #[tokio::test]
    async fn call_with_negative_timeout_returns_error() {
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs { keys: "echo hi\n".into(), interactive: false, timeout: Some(-1.0) })
            .await;
        assert!(matches!(result, Err(TypeError::InvalidTimeout(t)) if t == -1.0));
    }

    #[tokio::test]
    async fn call_with_infinite_timeout_returns_error() {
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs {
                keys: "echo hi\n".into(),
                interactive: false,
                timeout: Some(f64::INFINITY),
            })
            .await;
        assert!(matches!(result, Err(TypeError::InvalidTimeout(t)) if t.is_infinite()));
    }

    #[tokio::test]
    async fn call_with_huge_finite_timeout_returns_error_instead_of_panicking() {
        // f64::MAX is finite but overflows Duration::from_secs_f64, which panics.
        // The validation must reject values above MAX_TIMEOUT_SECS.
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs {
                keys: "echo hi\n".into(),
                interactive: false,
                timeout: Some(f64::MAX),
            })
            .await;
        assert!(
            matches!(result, Err(TypeError::InvalidTimeout(_))),
            "f64::MAX must be rejected as InvalidTimeout, not panic; got: {result:?}"
        );
    }

    #[tokio::test]
    async fn call_with_timeout_just_above_max_returns_error() {
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs {
                keys: "echo hi\n".into(),
                interactive: false,
                timeout: Some(MAX_TIMEOUT_SECS + 1.0),
            })
            .await;
        assert!(matches!(result, Err(TypeError::InvalidTimeout(_))));
    }

    #[tokio::test]
    async fn call_with_timeout_at_max_is_accepted() {
        // Exactly MAX_TIMEOUT_SECS must be valid.
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs {
                keys: "echo hi\n".into(),
                interactive: false,
                timeout: Some(MAX_TIMEOUT_SECS),
            })
            .await;
        // Should fail with SessionNotInitialized (timeout is valid, no session).
        assert!(
            matches!(result, Err(TypeError::SessionNotInitialized)),
            "MAX_TIMEOUT_SECS must be accepted; got: {result:?}"
        );
    }

    #[tokio::test]
    async fn interactive_timeout_returns_buffered_partial_output_not_empty() {
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();

        // `printf` flushes immediately (unlike `echo` which may buffer).
        // The marker appears in the PTY stream before `cat` blocks on stdin,
        // so it must be present in the partial output returned on timeout.
        let result = tool
            .call(TypeArgs {
                keys: "printf 'partial_output_marker'; cat\n".into(),
                interactive: true,
                timeout: Some(0.3),
            })
            .await
            .unwrap();

        // Clean up: Ctrl-C kills the still-running cat.
        let _ = tool
            .call(TypeArgs { keys: "\x03".into(), interactive: false, timeout: Some(2.0) })
            .await;

        assert!(
            result.output.contains("partial_output_marker"),
            "interactive timeout must return output buffered before the timeout; \
             got {:?}. Bug: interactive-timeout arm returns Ok((vec![], ...)).",
            result.output
        );
    }

    #[test]
    fn retain_does_not_fire_for_pids_absent_from_sysinfo_snapshot() {
        // Regression test for the spawn_watcher TOCTOU race.
        //
        // The watcher snapshots `tracked` PIDs under the lock, drops the lock,
        // calls sysinfo for only those PIDs, then calls retain(checked, alive).
        // If `call()` adds a new PID between the snapshot and retain, that PID
        // must not be treated as completed — it was never checked by sysinfo.
        //
        // Timeline:
        //   T0: watcher snapshots tracked = {10}, drops lock
        //   T1: call() runs sync({10, 20}) — PID 20 newly discovered
        //   T2: watcher refreshes sysinfo for {10} only; checked = {10}, alive = {10}
        //   T3: watcher calls retain({10}, {10}) — PID 20 is in jobs but not checked
        //   Fix: retain leaves PID 20 alone; only checked+dead PIDs are removed.
        let completed = Arc::new(Mutex::new(Vec::<u32>::new()));
        let cc = Arc::clone(&completed);
        let mut manager = JobManager::new().with_callback(move |n| {
            cc.lock().unwrap().push(n.pid);
        });

        manager.sync(&HashSet::from([10]));     // T0: snapshot — tracked = {10}
        manager.sync(&HashSet::from([10, 20])); // T1: call() adds PID 20 mid-flight

        // T3: watcher calls retain with only the PIDs it actually asked sysinfo
        // about. PID 20 is alive; it simply wasn't in the snapshot.
        manager.retain(&HashSet::from([10]), &HashSet::from([10]));

        let fired = completed.lock().unwrap().clone();
        assert!(
            !fired.contains(&20),
            "retain must not fire completion for PID 20 — it was not in the \
             sysinfo snapshot so its liveness was never checked; \
             got spurious completions: {:?}",
            fired
        );
    }

    // ── spawn_watcher: sysinfo polling integration tests ─────────────────

    #[tokio::test]
    async fn watcher_detects_background_job_completion() {
        // Start a short-lived background job, discover it via call(), then let
        // the watcher detect its completion via sysinfo polling.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let tool = tool.with_job_callback(move |notif| {
            let _ = tx.send(notif.pid);
        });

        // Start the watcher before launching the job so it's already polling.
        let watcher = tool.spawn_watcher();

        // Launch `sleep 2 &` and discover the background PID via call().
        let result = tool
            .call(TypeArgs {
                keys: "sleep 2 &\n".into(),
                interactive: false,
                timeout: None,
            })
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 0, .. }));

        // Verify the job was discovered.
        let tracked: Vec<u32> = tool
            .job_manager
            .lock()
            .unwrap()
            .jobs
            .keys()
            .copied()
            .collect();
        assert!(!tracked.is_empty(), "background job must be tracked after call()");

        // Wait for the callback to fire (sleep 2 + up to 3s poll margin).
        let completed_pid = tokio::time::timeout(
            Duration::from_secs(8),
            rx.recv(),
        )
        .await
        .expect("watcher must detect background job completion within 8s")
        .expect("channel must not be closed");

        assert!(
            tracked.contains(&completed_pid),
            "completed PID must match a tracked job"
        );

        watcher.abort();
    }

    // ── Shell death / exec: session destruction tests ──────────────────

    #[tokio::test]
    async fn exec_replaces_shell_first_call_returns_buffered_output() {
        // `exec /bin/echo` replaces bash with echo. Echo prints "Hello Mom!"
        // and exits, closing the PTY slave side. expectrl hits EOF.
        //
        // Correct behavior: the output printed before EOF ("Hello Mom!") must
        // be returned to the caller. The shell is dead so there's no sentinel
        // — the outcome should be ShellExited.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();

        let result = tool
            .call(TypeArgs {
                keys: "exec /bin/echo 'Hello Mom!'\n".into(),
                interactive: false,
                timeout: Some(2.0),
            })
            .await
            .expect("call must return Ok with buffered output on shell exit");

        assert!(
            result.output.contains("Hello Mom!"),
            "output printed before EOF must be returned to the caller; got: {:?}",
            result.output
        );
        assert!(
            matches!(result.outcome, Outcome::ShellExited { .. }),
            "shell is dead, so outcome must be ShellExited; got: {:?}", result.outcome
        );
    }

    #[tokio::test]
    async fn exec_replaces_shell_subsequent_call_recovers_with_new_session() {
        // After `exec` kills the shell, the PTY slave side is closed.
        //
        // Correct behavior: When the shell dies, call() detects EOF and
        // respawns a new bash session transparently. The first call returns
        // ShellExited, and the second call should work normally.
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();

        // Kill the shell.
        let first_result = tool
            .call(TypeArgs {
                keys: "exec /bin/echo 'Hello Mom!'\n".into(),
                interactive: false,
                timeout: Some(2.0),
            })
            .await
            .expect("call must return Ok with ShellExited on exec");

        // First call should result in ShellExited.
        assert!(
            matches!(first_result.outcome, Outcome::ShellExited { .. }),
            "exec must kill the shell, resulting in ShellExited; got: {:?}", first_result.outcome
        );

        // Next call should work — the shell has been respawned.
        let result = tool
            .call(TypeArgs {
                keys: "echo 'still alive!'\n".into(),
                interactive: false,
                timeout: Some(5.0),
            })
            .await
            .expect("call on a respawned session must work normally");

        assert!(
            result.output.contains("still alive!"),
            "respawned shell must execute the command; got: {:?}",
            result.output
        );
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 0, .. }));
    }

    #[tokio::test]
    async fn watcher_no_op_when_no_jobs() {
        // The watcher must not panic or fire callbacks when no jobs are tracked.
        let fired = Arc::new(Mutex::new(false));
        let fc = Arc::clone(&fired);
        let (tool, _cwd) = TypeTool::spawn().await.unwrap();
        let tool = tool.with_job_callback(move |_| {
            *fc.lock().unwrap() = true;
        });

        let watcher = tool.spawn_watcher();

        // Let several poll cycles pass.
        tokio::time::sleep(Duration::from_millis(2500)).await;

        assert!(!*fired.lock().unwrap(), "no callback should fire with no tracked jobs");
        watcher.abort();
    }
}
