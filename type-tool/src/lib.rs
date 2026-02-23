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
//!     let tool = TypeTool::new()
//!         .with_name("claude")
//!         .with_job_callback(|notif| {
//!             // Queue a proactive message to the agent, wake it up if idle, etc.
//!             eprintln!("job {} done (exit {:?})", notif.pid, notif.exit_code);
//!         })
//!         .spawn()
//!         .await
//!         .unwrap();
//!     let cwd = tool.working_directory().await;
//!
//!     // Start the sysinfo watcher before moving `tool` into the agent builder.
//!     let watcher = tool.spawn_watcher();
//!
//!     let agent = AgentBuilder::new(model)
//!         .preamble(&format!(
//!             "You're looking at a terminal. The only tool you have is `type`, \
//!              which sends keystrokes to the terminal.\n\n\
//!              The terminal says:\n\n\
//!              $?=0 claude:{cwd} $ ",
//!         ))
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

/// Exit code used when `drain_after_kill` fails to extract the real exit code
/// from the post-kill sentinel. Must not be 0 (which would look like success).
const DRAIN_FAILURE_EXIT_CODE: i32 = -1;

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
    /// Supports JSON escape sequences: `\n` for Enter, `\u0003` for Ctrl-C, etc.
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

    /// Kill the current shell session and spawn a fresh one.
    ///
    /// When `true`, all other arguments are ignored. The shell is killed,
    /// a new bash session is spawned, and the tool returns with
    /// [`Outcome::Reset`]. Use this as a last resort when the shell is
    /// in an unrecoverable state.
    #[serde(default)]
    pub reset: bool,
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
#[serde(tag = "type")]
pub enum Outcome {
    /// The command ran to completion and the shell emitted a prompt sentinel.
    Completed {
        exit_code: i32,
        working_directory: String,
    },

    /// The command timed out (non-interactive) and was killed.
    /// The shell is back at a prompt and ready for the next command.
    Killed {
        exit_code: i32,
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

    /// The session was explicitly reset by the caller (`reset: true`).
    /// The old shell was killed and a fresh one has been spawned.
    Reset {
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
                // command is backfilled by the sysinfo watcher on its next poll.
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

    /// Agent name shown in the prompt (e.g. "claude"). Empty = no name prefix.
    agent_name: String,
}

impl TypeTool {
    /// Create a new `TypeTool` with no session, an empty [`JobManager`], and no callback.
    ///
    /// Use the builder methods [`with_name`](Self::with_name) and
    /// [`with_job_callback`](Self::with_job_callback) to configure the tool,
    /// then call [`spawn`](Self::spawn) to start the PTY session.
    pub fn new() -> Self {
        Self {
            session: Arc::new(tokio::sync::Mutex::new(None)),
            job_manager: Arc::new(Mutex::new(JobManager::new())),
            agent_name: String::new(),
        }
    }

    /// Set the agent name shown in the terminal prompt (e.g. "claude").
    ///
    /// When set, prompts look like `$?=0 claude:/Users/nilton/src $ `.
    /// When empty (the default), the name prefix is omitted: `$?=0 /Users/nilton/src $ `.
    pub fn with_name(mut self, name: impl AsRef<str>) -> Self {
        self.agent_name = name.as_ref().to_string();
        self
    }

    /// Spawn a bash PTY session and return a fully initialized `TypeTool`.
    ///
    /// Consumes `self` (from the builder chain) and returns an initialized tool.
    /// Use [`working_directory`](Self::working_directory) to read the initial `$PWD`.
    pub async fn spawn(self) -> Result<Self, TypeError> {
        let pty_session = session::PtySession::spawn().await?;
        Ok(Self {
            session: Arc::new(tokio::sync::Mutex::new(Some(pty_session))),
            job_manager: self.job_manager,
            agent_name: self.agent_name,
        })
    }

    /// Return the current working directory from the most recent sentinel.
    ///
    /// Returns an empty string if no session is active.
    pub async fn working_directory(&self) -> String {
        let guard = self.session.lock().await;
        guard
            .as_ref()
            .map(|pty| pty.working_directory.clone())
            .unwrap_or_default()
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
            agent_name: self.agent_name,
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
                    let jm = job_manager.lock().unwrap_or_else(|e| e.into_inner());
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
                // OnlyIfNotSet fetches the command line once per PID (cached
                // across subsequent refreshes).
                sys.refresh_processes_specifics(
                    sysinfo::ProcessesToUpdate::Some(&tracked),
                    true,
                    sysinfo::ProcessRefreshKind::nothing()
                        .with_cmd(sysinfo::UpdateKind::OnlyIfNotSet),
                );

                let checked: HashSet<u32> = tracked.iter().map(|pid| pid.as_u32()).collect();
                let alive: HashSet<u32> = tracked
                    .iter()
                    .filter(|pid| sys.process(**pid).is_some())
                    .map(|pid| pid.as_u32())
                    .collect();

                {
                    let mut jm = job_manager.lock().unwrap_or_else(|e| e.into_inner());

                    // Backfill the command field for alive jobs that don't have one yet.
                    for &pid in &alive {
                        if let Some(job) = jm.jobs.get_mut(&pid) {
                            if job.command.is_empty() {
                                if let Some(proc) = sys.process(sysinfo::Pid::from_u32(pid)) {
                                    let cmd = proc.cmd();
                                    if !cmd.is_empty() {
                                        job.command = cmd
                                            .iter()
                                            .map(|s| s.to_string_lossy())
                                            .collect::<Vec<_>>()
                                            .join(" ");
                                    } else {
                                        // cmd() can be empty (permissions, kernel threads).
                                        // Fall back to the process name (short but non-empty).
                                        let name = proc.name().to_string_lossy();
                                        if !name.is_empty() {
                                            job.command = name.into_owned();
                                        }
                                    }
                                }
                            }
                        }
                    }

                    jm.retain(&checked, &alive);
                }
            }
        })
    }
}

impl TypeTool {
    /// Build the prompt string appended to every command's output.
    ///
    /// Format with agent name: `$?=0 claude:/Users/nilton/src $ `
    /// Format without name:    `$?=0 /Users/nilton/src $ `
    fn format_prompt(&self, exit_code: i32, cwd: &str) -> String {
        if self.agent_name.is_empty() {
            format!("$?={exit_code} {cwd} $ ")
        } else {
            format!("$?={} {}:{} $ ", exit_code, self.agent_name, cwd)
        }
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
        let cwd = self.working_directory().await;
        let initial_prompt = self.format_prompt(0, &cwd);
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Send keystrokes to the terminal. Supports escape sequences: \\n for Enter, \
                 \\u0003 for Ctrl-C, \\u0004 for Ctrl-D, etc. Returns the terminal output \
                 captured until the next shell prompt or timeout.\n\n\
                 The terminal says:\n\n\
                 {initial_prompt}"
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Keystrokes to send. Supports escape sequences: \\n for Enter, \\u0003 for Ctrl-C, \\u0004 for Ctrl-D, etc."
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
        // Reset: kill the current shell and spawn a fresh one. All other args ignored.
        if args.reset {
            let mut guard = self.session.lock().await;

            // Kill the old shell if it exists.
            if let Some(pty) = guard.as_ref() {
                session::kill_shell(pty.shell_pid);
            }

            // Spawn a fresh session.
            let new_pty = session::PtySession::spawn().await?;
            let working_directory = new_pty.working_directory.clone();
            *guard = Some(new_pty);

            // Clear all tracked background jobs — they belonged to the old shell.
            self.job_manager.lock().unwrap().jobs.clear();

            let mut output = String::from("Session reset\n");
            output.push_str(&self.format_prompt(0, &working_directory));

            return Ok(TypeOutput {
                output,
                outcome: Outcome::Reset { working_directory },
            });
        }

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

            // 1. Send keystrokes to the PTY.
            use expectrl::Expect;
            pty.session.send(keys.as_bytes())
                .map_err(|e| TypeError::PtyWrite(e.to_string()))?;

            // 2. Wait for the prompt sentinel.
            match session::wait_for_sentinel(&mut pty.session, timeout) {
                Ok(captures) => {
                    let output = captures.before().to_vec();
                    let (exit_code, cwd) = session::parse_sentinel(&captures);
                    let active_pids = session::run_jobs_p_sync(&mut pty.session);
                    pty.working_directory = cwd.clone();
                    let outcome = Outcome::Completed {
                        exit_code,
                        working_directory: cwd,
                    };
                    Ok((output, outcome, active_pids))
                }

                Err(expectrl::Error::ExpectTimeout) if !interactive => {
                    // Drain partial output buffered by expect() before the timeout.
                    let output = session::drain_buf(&mut pty.session)?;

                    // Kill the foreground process (SIGTERM → SIGKILL).
                    if !session::kill_foreground(shell_pid, &background_pids) {
                        // Foreground processes survived SIGKILL — the shell is
                        // unrecoverable. Kill it entirely and respawn.
                        session::kill_shell(shell_pid);
                        let new_pty = session::PtySession::spawn_blocking_inner()?;
                        let cwd = new_pty.working_directory.clone();
                        *guard = Some(new_pty);
                        let outcome = Outcome::ShellExited {
                            working_directory: cwd,
                        };
                        return Ok((output, outcome, None));
                    }

                    // Drain PTY to restore a clean prompt; extract post-kill exit code and $PWD.
                    let (exit_code, cwd) = session::drain_after_kill(&mut pty.session)
                        .unwrap_or_else(|| (DRAIN_FAILURE_EXIT_CODE, pty.working_directory.clone()));

                    let active_pids = session::run_jobs_p_sync(&mut pty.session);
                    pty.working_directory = cwd.clone();
                    let outcome = Outcome::Killed {
                        exit_code,
                        working_directory: cwd,
                        timeout_secs,
                    };
                    Ok((output, outcome, active_pids))
                }

                Err(expectrl::Error::ExpectTimeout) => {
                    // Interactive timeout: return partial output; process still alive.
                    // active_pids is None to suppress the post-call sync — we have no
                    // fresh jobs-p snapshot and syncing with an empty set would fire
                    // spurious completion callbacks for all tracked background jobs.
                    let output = session::drain_buf(&mut pty.session)?;
                    Ok((output, Outcome::Waiting, None))
                }

                Err(expectrl::Error::Eof) => {
                    // Shell process exited (exec, exit, crash, etc.).
                    // Drain any buffered output before EOF.
                    let output = session::drain_buf(&mut pty.session)?;

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

        let mut output = String::from_utf8_lossy(&output_bytes).into_owned();

        // Append the harness-built prompt so the model sees exit code, identity,
        // and location context at the end of every command's output.
        match &outcome {
            Outcome::Completed { exit_code, working_directory } => {
                output.push_str(&self.format_prompt(*exit_code, working_directory));
            }
            Outcome::Killed { exit_code, working_directory, .. } => {
                output.push_str("Killed\n");
                output.push_str(&self.format_prompt(*exit_code, working_directory));
            }
            Outcome::ShellExited { working_directory } => {
                output.push_str("Shell exited, respawned\n");
                output.push_str(&self.format_prompt(0, working_directory));
            }
            Outcome::Reset { .. } => {
                // Reset builds its own output and returns early above.
                unreachable!("reset returns early");
            }
            Outcome::Waiting => {
                // Process still running — no prompt to append.
            }
        }

        Ok(TypeOutput { output, outcome })
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
            reset: false,
        };
        assert_eq!(args.resolved_timeout(), DEFAULT_TIMEOUT_SECS);
    }

    #[test]
    fn resolved_timeout_uses_default_for_interactive() {
        let args = TypeArgs {
            keys: "y\n".to_string(),
            interactive: true,
            timeout: None,
            reset: false,
        };
        assert_eq!(args.resolved_timeout(), DEFAULT_INTERACTIVE_TIMEOUT_SECS);
    }

    #[test]
    fn resolved_timeout_respects_caller_override() {
        let args = TypeArgs {
            keys: "analyze dataset.csv\n".to_string(),
            interactive: true,
            timeout: Some(30.0),
            reset: false,
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
                reset: false,
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
        let args = TypeArgs { keys: "ls\n".into(), interactive: false, timeout: Some(0.0) , reset: false};
        assert_eq!(args.resolved_timeout(), 0.0);
    }

    #[test]
    fn resolved_timeout_negative_value_passed_through() {
        // Validation of the timeout range is the harness's job, not the resolver's.
        // The resolver must faithfully return whatever the caller supplied, even
        // nonsensical values like -1.
        let args = TypeArgs { keys: "ls\n".into(), interactive: false, timeout: Some(-1.0) , reset: false};
        assert_eq!(args.resolved_timeout(), -1.0);
    }

    #[test]
    fn resolved_timeout_infinity_passed_through() {
        // Infinite timeout is useful for commands with no known upper bound.
        let args = TypeArgs {
            keys: "ls\n".into(),
            interactive: false,
            timeout: Some(f64::INFINITY),
            reset: false,
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
            reset: false,
        };
        assert!(args.resolved_timeout().is_nan());
    }

    #[test]
    fn resolved_timeout_override_applies_in_both_interactive_modes() {
        // The caller-supplied value wins regardless of the `interactive` flag.
        // A bug might only apply the override in one branch of the if/else.
        let non_interactive = TypeArgs { keys: String::new(), interactive: false, timeout: Some(42.0) , reset: false};
        let interactive = TypeArgs { keys: String::new(), interactive: true, timeout: Some(42.0) , reset: false};
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
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        tool.call(TypeArgs {
            keys: "export TEKTON_PERSIST_TEST=hello\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "echo $TEKTON_PERSIST_TEST\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();
        assert_eq!(result.output, format!("hello\r\n{}", expected_prompt("", 0, &cwd)));
    }

    #[tokio::test]
    async fn working_directory_persists_after_cd() {
        // `cd` must permanently change $PWD for all subsequent calls.
        // Regression guard: fresh-shell execution resets $PWD on every call.
        let tool = TypeTool::new().spawn().await.unwrap();
        tool.call(TypeArgs {
            keys: "cd /tmp\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "pwd\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();
        assert_eq!(result.output, format!("/tmp\r\n{}", expected_prompt("", 0, "/tmp")));
        assert!(
            matches!(result.outcome, Outcome::Completed { ref working_directory, .. } if working_directory == "/tmp"),
            "working directory must be /tmp; got: {:?}", result.outcome
        );
    }

    #[tokio::test]
    async fn shell_function_defined_in_one_call_is_callable_in_next() {
        // Shell functions live in the session's environment. A function defined
        // in call N must be invocable in call N+1.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        tool.call(TypeArgs {
            keys: "tekton_greet() { echo \"greetings $1\"; }\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "tekton_greet world\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();
        assert_eq!(
            result.output,
            format!("greetings world\r\n{}", expected_prompt("", 0, &cwd)),
        );
    }

    #[tokio::test]
    async fn non_exported_shell_variable_does_not_leak_to_subprocesses() {
        // Standard bash scoping: non-exported variables must not be visible to
        // child processes, even across calls within the same session.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        tool.call(TypeArgs {
            keys: "TEKTON_LOCAL=secret\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "bash -c 'echo ${TEKTON_LOCAL:-UNSET}'\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();
        assert_eq!(
            result.output,
            format!("UNSET\r\n{}", expected_prompt("", 0, &cwd)),
        );
    }

    // ── PTY integration: OSC sentinel and exit-code tests ─────────────────

    #[tokio::test]
    async fn exit_code_zero_for_successful_command() {
        // `true` always exits 0. The OSC sentinel must carry that value.
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs { keys: "true\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 0, .. }));
    }

    #[tokio::test]
    async fn exit_code_one_for_failing_command() {
        // `false` always exits 1. Tests that non-zero codes are relayed.
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs { keys: "false\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 1, .. }));
    }

    #[tokio::test]
    async fn exit_code_arbitrary_value_faithfully_captured_from_sentinel() {
        // The OSC sentinel must relay any exit code, not just 0/1.
        // A bug might mask the code by always returning 0 or 1.
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "sh -c 'exit 42'\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 42, .. }));
    }

    #[tokio::test]
    async fn exit_code_127_for_unknown_command() {
        // Bash exits 127 on "command not found". Verifies the sentinel works
        // for error paths that never reach the program's own exit() call.
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "tekton_nonexistent_command_xyz_abc\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();
        assert!(matches!(result.outcome, Outcome::Completed { exit_code: 127, .. }));
    }

    #[tokio::test]
    async fn osc_sentinel_bytes_are_stripped_from_output() {
        // The sentinel `\e]999;EXIT\a` is machine metadata. It must not appear
        // in the output string returned to the model — only clean terminal text.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs { keys: "echo hello\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(result.output, format!("hello\r\n{}", expected_prompt("", 0, &cwd)));
    }

    #[tokio::test]
    async fn keystroke_echo_disabled_so_typed_input_does_not_appear_in_output() {
        // Design: echo is disabled on the PTY. The model's keystrokes must not
        // be reflected back in the output it receives.
        //
        // If echo is ON, the input "echo marker_xyz\n" appears in the output
        // stream, making "marker_xyz" occur twice (once as echoed input, once
        // as command output). With echo OFF it occurs exactly once.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs {
                keys: "echo tekton_echo_marker_xyz\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();
        assert_eq!(
            result.output,
            format!("tekton_echo_marker_xyz\r\n{}", expected_prompt("", 0, &cwd)),
        );
    }

    // ── PTY integration: harness-built prompt in output ─────────────────────
    //
    // The harness appends a prompt to every command's output from sentinel
    // data. Format: `$?=EXIT_CODE [NAME:]CWD $ `

    /// Build the expected prompt string for test assertions.
    fn expected_prompt(name: &str, exit_code: i32, cwd: &str) -> String {
        if name.is_empty() {
            format!("$?={exit_code} {cwd} $ ")
        } else {
            format!("$?={exit_code} {name}:{cwd} $ ")
        }
    }

    #[tokio::test]
    async fn true_exact_output() {
        // `true` produces no output — just the prompt with exit code 0.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs { keys: "true\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(result.output, expected_prompt("", 0, &cwd));
    }

    #[tokio::test]
    async fn false_exact_output() {
        // `false` produces no output — just the prompt with exit code 1.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs { keys: "false\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(result.output, expected_prompt("", 1, &cwd));
    }

    #[tokio::test]
    async fn echo_hello_exact_output() {
        // `echo hello` outputs "hello\r\n" (PTY translates \n → \r\n) then prompt.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs { keys: "echo hello\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        let expected = format!("hello\r\n{}", expected_prompt("", 0, &cwd));
        assert_eq!(result.output, expected);
    }

    #[tokio::test]
    async fn echo_n_no_trailing_newline() {
        // `echo -n foo` outputs "foo" with no newline. The prompt appears on
        // the same line, just like a real terminal.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs { keys: "echo -n foo\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        let expected = format!("foo{}", expected_prompt("", 0, &cwd));
        assert_eq!(result.output, expected);
    }

    #[tokio::test]
    async fn cd_then_echo_no_stale_prompt_leakage() {
        // Two calls: cd /tmp, then echo marker. The second call's output must
        // start clean — no leftover prompt from the first call leaking in.
        let tool = TypeTool::new().spawn().await.unwrap();

        let cd_result = tool
            .call(TypeArgs { keys: "cd /tmp\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        // cd produces no output, just the prompt showing the new cwd.
        assert_eq!(cd_result.output, expected_prompt("", 0, "/tmp"));

        let echo_result = tool
            .call(TypeArgs { keys: "echo output_marker\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        let expected = format!("output_marker\r\n{}", expected_prompt("", 0, "/tmp"));
        assert_eq!(echo_result.output, expected);
    }

    #[tokio::test]
    async fn multi_command_session_exact_outputs() {
        // A multi-step session: export, cd, echo, false — verifying exact output
        // for each call and no cross-call leakage.
        let tool = TypeTool::new().spawn().await.unwrap();
        let initial_cwd = tool.working_directory().await;

        // 1. export — no visible output
        let r1 = tool
            .call(TypeArgs { keys: "export TEKTON_X=42\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(r1.output, expected_prompt("", 0, &initial_cwd));

        // 2. cd /
        let r2 = tool
            .call(TypeArgs { keys: "cd /\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(r2.output, expected_prompt("", 0, "/"));

        // 3. echo $TEKTON_X — verifies persistence of env var and cwd
        let r3 = tool
            .call(TypeArgs { keys: "echo $TEKTON_X\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(r3.output, format!("42\r\n{}", expected_prompt("", 0, "/")));

        // 4. false — exit code 1
        let r4 = tool
            .call(TypeArgs { keys: "false\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(r4.output, expected_prompt("", 1, "/"));
    }

    #[tokio::test]
    async fn prompt_with_agent_name() {
        // When agent name is set, prompts include `name:cwd`.
        let tool = TypeTool::new()
            .with_name("claude")
            .spawn()
            .await
            .unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs { keys: "true\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        assert_eq!(result.output, expected_prompt("claude", 0, &cwd));
    }

    #[tokio::test]
    async fn prompt_with_empty_agent_name() {
        // Default (no name) omits the name prefix from the prompt.
        let tool = TypeTool::new().spawn().await.unwrap();
        let cwd = tool.working_directory().await;
        let result = tool
            .call(TypeArgs { keys: "true\n".into(), interactive: false, timeout: None , reset: false})
            .await
            .unwrap();
        // Must NOT contain a colon before the cwd (no name prefix).
        assert_eq!(result.output, format!("$?=0 {cwd} $ "));
    }

    #[tokio::test]
    async fn killed_output_includes_killed_message() {
        // Non-interactive timeout kill appends "Killed\n" then the prompt.
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: false,
                timeout: Some(0.1),
                reset: false,
            })
            .await
            .unwrap();
        match &result.outcome {
            Outcome::Killed { exit_code, working_directory, .. } => {
                let expected = format!(
                    "Killed\n{}",
                    expected_prompt("", *exit_code, working_directory),
                );
                assert_eq!(result.output, expected);
            }
            other => panic!("expected Outcome::Killed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn shell_exited_output_includes_respawned_message() {
        // When the shell exits, output includes "Shell exited, respawned\n" + prompt.
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "exec command true\n".into(),
                interactive: false,
                timeout: Some(2.0),
                reset: false,
            })
            .await
            .unwrap();
        match &result.outcome {
            Outcome::ShellExited { working_directory } => {
                let expected = format!(
                    "Shell exited, respawned\n{}",
                    expected_prompt("", 0, working_directory),
                );
                assert_eq!(result.output, expected);
            }
            other => panic!("expected Outcome::ShellExited; got: {other:?}"),
        }
    }

    // ── PTY integration: timeout and process-control tests ─────────────────

    #[tokio::test]
    async fn non_interactive_timeout_kills_foreground_and_sets_timeout_message() {
        // Non-interactive timeout: kill the foreground process group, return
        // captured output. The outcome is Killed.
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: false,
                timeout: Some(0.1), // 100 ms — much shorter than sleep
                reset: false,
            })
            .await
            .unwrap();
        assert!(
            matches!(result.outcome, Outcome::Killed { .. }),
            "non-interactive timeout must result in Outcome::Killed; got: {:?}", result.outcome
        );
    }

    #[tokio::test]
    async fn non_interactive_timeout_killed_outcome_includes_exit_code() {
        // After kill_foreground sends SIGTERM/SIGKILL, bash regains control and
        // emits a sentinel whose exit code reflects the killed process (typically
        // 143 for SIGTERM or 137 for SIGKILL).
        let tool = TypeTool::new().spawn().await.unwrap();
        let result = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: false,
                timeout: Some(0.1),
                reset: false,
            })
            .await
            .unwrap();
        match result.outcome {
            Outcome::Killed { exit_code, .. } => {
                // SIGTERM = 128+15=143, SIGKILL = 128+9=137. Either is acceptable.
                assert!(
                    exit_code == 143 || exit_code == 137,
                    "killed process exit code must be 143 (SIGTERM) or 137 (SIGKILL); got: {exit_code}"
                );
            }
            other => panic!("expected Outcome::Killed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn interactive_timeout_returns_partial_output_without_killing_process() {
        // Interactive timeout: return partial output so the model can decide
        // what to type next. The process must remain alive.
        let tool = TypeTool::new().spawn().await.unwrap();
        // `cat` with no args blocks on stdin — ideal for testing interactive mode.
        let partial = tool
            .call(TypeArgs {
                keys: "cat\n".into(),
                interactive: true,
                timeout: Some(0.1),
                reset: false,
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
                reset: false,
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
        let tool = TypeTool::new().spawn().await.unwrap();
        // Start a long-running process in interactive mode, let it time out so
        // we get partial output quickly.
        let _ = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: true,
                timeout: Some(0.1),
                reset: false,
            })
            .await;
        // Send Ctrl-C to kill the still-running sleep.
        let result = tool
            .call(TypeArgs {
                keys: "\x03".into(),
                interactive: false,
                timeout: Some(2.0),
                reset: false,
            })
            .await
            .unwrap();
        assert!(
            matches!(result.outcome, Outcome::Completed { .. }),
            "Ctrl-C must terminate the foreground process and produce a sentinel; got: {:?}", result.outcome
        );
    }

    // ── Escape sequence handling: JSON → PTY byte path ────────────────────
    //
    // The LLM sends JSON like `{"keys": "ls\n"}`. The tool description
    // advertises `\n`, `\x03`, `\x04` etc.  JSON only supports `\n`, `\t`,
    // `\uXXXX`, and a few others — NOT `\xNN`.  These tests verify what
    // actually arrives at the PTY after serde_json deserialization.

    #[test]
    fn json_backslash_n_deserializes_to_newline_byte() {
        // JSON supports \n natively. This is the happy path for Enter.
        let args: TypeArgs = serde_json::from_str(r#"{"keys": "ls\n"}"#).unwrap();
        assert!(
            args.keys.contains('\n'),
            r#"JSON "ls\n" must deserialize to a string containing 0x0A; got bytes: {:?}"#,
            args.keys.as_bytes()
        );
    }

    #[test]
    fn json_backslash_x03_does_not_deserialize_to_ctrl_c_byte() {
        // JSON does NOT support \xNN escape sequences.  serde_json rejects
        // them outright.  If the LLM sends {"keys": "\x03"}, deserialization
        // fails — the Ctrl-C never reaches the PTY.
        //
        // This documents the current behavior.  The tool description advertises
        // \x03 for Ctrl-C, but it only works if the framework pre-processes
        // escape sequences before calling serde, or if the LLM uses \u0003.
        let result = serde_json::from_str::<TypeArgs>(r#"{"keys": "\x03"}"#);
        assert!(
            result.is_err(),
            r#"JSON "\x03" is not a valid JSON escape; serde_json must reject it. \
               If this passes, the LLM's \x03 silently becomes something other than Ctrl-C."#
        );
    }

    #[test]
    fn json_backslash_u0003_deserializes_to_ctrl_c_byte() {
        // \u0003 is JSON's way to encode Ctrl-C (ETX, byte 0x03).
        // This is the correct escape for sending control characters via JSON.
        let args: TypeArgs = serde_json::from_str(r#"{"keys": "\u0003"}"#).unwrap();
        assert_eq!(
            args.keys.as_bytes(),
            &[0x03],
            r#"JSON "\u0003" must deserialize to byte 0x03 (Ctrl-C)"#
        );
    }

    #[test]
    fn json_backslash_u0004_deserializes_to_ctrl_d_byte() {
        // \u0004 = Ctrl-D (EOT). Same pattern as Ctrl-C.
        let args: TypeArgs = serde_json::from_str(r#"{"keys": "\u0004"}"#).unwrap();
        assert_eq!(
            args.keys.as_bytes(),
            &[0x04],
            r#"JSON "\u0004" must deserialize to byte 0x04 (Ctrl-D)"#
        );
    }

    #[tokio::test]
    async fn ctrl_c_via_json_u0003_terminates_foreground_process() {
        // End-to-end: the LLM sends \u0003 (the only correct JSON encoding
        // of Ctrl-C). Verify it actually delivers SIGINT to the foreground.
        let tool = TypeTool::new().spawn().await.unwrap();

        // Start a long-running foreground process.
        let _ = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: true,
                timeout: Some(0.1),
                reset: false,
            })
            .await;

        // Deserialize from JSON the way the framework would.
        let args: TypeArgs = serde_json::from_str(r#"{"keys": "\u0003", "interactive": false, "timeout": 2.0}"#).unwrap();
        let result = tool.call(args).await.unwrap();

        assert!(
            matches!(result.outcome, Outcome::Completed { .. }),
            r#"Ctrl-C via JSON \u0003 must terminate the foreground process; got: {:?}"#,
            result.outcome
        );
    }

    // ── Invalid timeout returns Err(InvalidTimeout) ───────────────────────

    #[tokio::test]
    async fn call_with_nan_timeout_returns_error() {
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs { keys: "echo hi\n".into(), interactive: false, timeout: Some(f64::NAN) , reset: false})
            .await;
        assert!(matches!(result, Err(TypeError::InvalidTimeout(t)) if t.is_nan()));
    }

    #[tokio::test]
    async fn call_with_negative_timeout_returns_error() {
        let tool = TypeTool::new();
        let result = tool
            .call(TypeArgs { keys: "echo hi\n".into(), interactive: false, timeout: Some(-1.0) , reset: false})
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
                reset: false,
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
                reset: false,
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
                reset: false,
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
                reset: false,
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
        let tool = TypeTool::new().spawn().await.unwrap();

        // `printf` flushes immediately (unlike `echo` which may buffer).
        // The marker appears in the PTY stream before `cat` blocks on stdin,
        // so it must be present in the partial output returned on timeout.
        let result = tool
            .call(TypeArgs {
                keys: "printf 'partial_output_marker'; cat\n".into(),
                interactive: true,
                timeout: Some(0.3),
                reset: false,
            })
            .await
            .unwrap();

        // Clean up: Ctrl-C kills the still-running cat.
        let _ = tool
            .call(TypeArgs { keys: "\x03".into(), interactive: false, timeout: Some(2.0) , reset: false})
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
        let tool = TypeTool::new().spawn().await.unwrap();
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
                reset: false,
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

    #[tokio::test]
    async fn watcher_backfills_command_field_for_tracked_job() {
        // After call() discovers a background PID with an empty command,
        // the watcher's next sysinfo poll must backfill the command field.
        let tool = TypeTool::new().spawn().await.unwrap();
        let watcher = tool.spawn_watcher();

        // Launch a background job with a recognizable command.
        tool.call(TypeArgs {
            keys: "sleep 30 &\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();

        // sync() adds the PID with an empty command.
        let pid = {
            let jm = tool.job_manager.lock().unwrap();
            assert!(!jm.jobs.is_empty(), "background job must be tracked");
            let (&pid, job) = jm.jobs.iter().next().unwrap();
            assert!(job.command.is_empty(), "command must start empty after sync()");
            pid
        };

        // Wait for the watcher to poll sysinfo and backfill the command.
        let backfilled = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                tokio::time::sleep(Duration::from_millis(200)).await;
                let jm = tool.job_manager.lock().unwrap();
                if let Some(job) = jm.jobs.get(&pid) {
                    if !job.command.is_empty() {
                        return job.command.clone();
                    }
                } else {
                    break String::new(); // job was reaped, shouldn't happen for sleep 30
                }
            }
        })
        .await
        .expect("watcher must backfill the command field within 5s");

        assert!(
            backfilled.contains("sleep"),
            "backfilled command must contain 'sleep'; got: {:?}",
            backfilled
        );

        watcher.abort();
    }

    #[tokio::test]
    async fn watcher_completion_notification_includes_backfilled_command() {
        // End-to-end: the completion notification must carry the command string
        // that was backfilled by sysinfo, not the empty string from sync().
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let tool = TypeTool::new().spawn().await.unwrap();
        let tool = tool.with_job_callback(move |notif| {
            let _ = tx.send(notif.command);
        });
        let watcher = tool.spawn_watcher();

        // Launch a short-lived background job.
        tool.call(TypeArgs {
            keys: "sleep 2 &\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();

        // Wait for the watcher to detect completion and fire the callback.
        let command = tokio::time::timeout(Duration::from_secs(8), rx.recv())
            .await
            .expect("watcher must fire completion callback within 8s")
            .expect("channel must not be closed");

        assert!(
            command.contains("sleep"),
            "completion notification command must contain 'sleep'; got: {:?}",
            command
        );

        watcher.abort();
    }

    #[tokio::test]
    async fn watcher_does_not_overwrite_already_populated_command() {
        // If the command was already populated (e.g. by the harness), the
        // watcher must not overwrite it with sysinfo data.
        let tool = TypeTool::new().spawn().await.unwrap();
        let watcher = tool.spawn_watcher();

        tool.call(TypeArgs {
            keys: "sleep 30 &\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();

        // Manually set the command before the watcher polls.
        {
            let mut jm = tool.job_manager.lock().unwrap();
            let job = jm.jobs.values_mut().next().unwrap();
            job.command = "harness-provided-command".to_string();
        }

        // Let a few watcher poll cycles pass.
        tokio::time::sleep(Duration::from_millis(2500)).await;

        let jm = tool.job_manager.lock().unwrap();
        let job = jm.jobs.values().next().unwrap();
        assert_eq!(
            job.command, "harness-provided-command",
            "watcher must not overwrite a command that was already populated"
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
        let tool = TypeTool::new().spawn().await.unwrap();

        let result = tool
            .call(TypeArgs {
                keys: "exec /bin/echo 'Hello Mom!'\n".into(),
                interactive: false,
                timeout: Some(2.0),
                reset: false,
            })
            .await
            .expect("call must return Ok with buffered output on shell exit");

        match &result.outcome {
            Outcome::ShellExited { working_directory } => {
                let expected = format!(
                    "Hello Mom!\r\nShell exited, respawned\n{}",
                    expected_prompt("", 0, working_directory),
                );
                assert_eq!(result.output, expected);
            }
            other => panic!("expected Outcome::ShellExited; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn exec_replaces_shell_subsequent_call_recovers_with_new_session() {
        // After `exec` kills the shell, the PTY slave side is closed.
        //
        // Correct behavior: When the shell dies, call() detects EOF and
        // respawns a new bash session transparently. The first call returns
        // ShellExited, and the second call should work normally.
        let tool = TypeTool::new().spawn().await.unwrap();

        // Kill the shell.
        let first_result = tool
            .call(TypeArgs {
                keys: "exec /bin/echo 'Hello Mom!'\n".into(),
                interactive: false,
                timeout: Some(2.0),
                reset: false,
            })
            .await
            .expect("call must return Ok with ShellExited on exec");

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
                reset: false,
            })
            .await
            .expect("call on a respawned session must work normally");

        match &result.outcome {
            Outcome::Completed { working_directory, .. } => {
                assert_eq!(
                    result.output,
                    format!("still alive!\r\n{}", expected_prompt("", 0, working_directory)),
                );
            }
            other => panic!("expected Outcome::Completed; got: {other:?}"),
        }
    }

    // ── Bug exposure: silently swallowed errors ─────────────────────────

    #[test]
    fn drain_failure_exit_code_is_not_zero() {
        // When drain_after_kill returns None (drain timed out after kill),
        // the fallback exit code must indicate failure, not success.
        // DRAIN_FAILURE_EXIT_CODE = -1, distinguishable from a genuine exit 0.
        assert_ne!(
            DRAIN_FAILURE_EXIT_CODE, 0,
            "drain failure exit code must not be 0 (success); \
             drain_after_kill failure must be distinguishable from success"
        );
    }

    #[test]
    fn run_jobs_p_sync_error_does_not_cause_spurious_completions() {
        // When run_jobs_p_sync returns None (error), call() skips the sync.
        // This test verifies the gating pattern: only Some(pids) triggers sync.
        let completed = Arc::new(Mutex::new(Vec::new()));
        let cc = Arc::clone(&completed);
        let mut manager = JobManager::new().with_callback(move |n| {
            cc.lock().unwrap().push(n.pid);
        });

        // Three background jobs are running.
        manager.sync(&HashSet::from([100, 200, 300]));

        // Simulate run_jobs_p_sync returning None (error) — sync is skipped.
        let active_pids: Option<HashSet<u32>> = None;
        if let Some(pids) = active_pids {
            manager.sync(&pids);
        }

        let fired = completed.lock().unwrap().clone();
        assert!(
            fired.is_empty(),
            "when run_jobs_p_sync returns None, sync must be skipped; \
             got spurious completions for: {:?}",
            fired
        );
        assert_eq!(manager.jobs.len(), 3, "all jobs must remain tracked");
    }

    #[tokio::test]
    async fn killed_outcome_working_directory_must_not_be_empty_after_drain_failure() {
        // When drain_after_kill times out, unwrap_or_default() produces
        // working_directory: "". The model loses all navigation context.
        // This is a real scenario: if the killed process left the PTY in a
        // dirty state, the drain sentinel might never arrive.
        //
        // This test checks the contract: Killed outcome must always have
        // a non-empty working_directory (fall back to last-known cwd).
        let tool = TypeTool::new().spawn().await.unwrap();

        // Navigate somewhere specific so we have a known cwd.
        tool.call(TypeArgs {
            keys: "cd /tmp\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();

        // Trigger a non-interactive timeout + kill.
        let result = tool
            .call(TypeArgs {
                keys: "sleep 9999\n".into(),
                interactive: false,
                timeout: Some(0.1),
                reset: false,
            })
            .await
            .unwrap();

        match &result.outcome {
            Outcome::Killed { working_directory, .. } => {
                assert!(
                    !working_directory.is_empty(),
                    "Killed outcome must report a non-empty working_directory; \
                     if drain_after_kill fails, fall back to the last-known cwd."
                );
            }
            other => panic!("expected Outcome::Killed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn watcher_no_op_when_no_jobs() {
        // The watcher must not panic or fire callbacks when no jobs are tracked.
        let fired = Arc::new(Mutex::new(false));
        let fc = Arc::clone(&fired);
        let tool = TypeTool::new().spawn().await.unwrap();
        let tool = tool.with_job_callback(move |_| {
            *fc.lock().unwrap() = true;
        });

        let watcher = tool.spawn_watcher();

        // Let several poll cycles pass.
        tokio::time::sleep(Duration::from_millis(2500)).await;

        assert!(!*fired.lock().unwrap(), "no callback should fire with no tracked jobs");
        watcher.abort();
    }

    // ── Reset tests ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn reset_spawns_fresh_session() {
        // After reset, the tool must have a working session at a valid cwd.
        let tool = TypeTool::new().spawn().await.unwrap();

        let result = tool
            .call(TypeArgs {
                keys: String::new(),
                interactive: false,
                timeout: None,
                reset: true,
            })
            .await
            .unwrap();

        match &result.outcome {
            Outcome::Reset { working_directory } => {
                assert!(!working_directory.is_empty(), "reset must report a non-empty cwd");
                let expected = format!("Session reset\n{}", expected_prompt("", 0, working_directory));
                assert_eq!(result.output, expected);
            }
            other => panic!("expected Outcome::Reset; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn reset_ignores_keys_argument() {
        // When reset=true, the keys field is ignored — no keystrokes are sent.
        // Verify by sending a command that would set an env var, then checking
        // it doesn't exist after reset.
        let tool = TypeTool::new().spawn().await.unwrap();

        // Reset with keys that would normally set a variable.
        tool.call(TypeArgs {
            keys: "export TEKTON_RESET_TEST=should_not_exist\n".into(),
            interactive: false,
            timeout: None,
            reset: true,
        })
        .await
        .unwrap();

        // The variable must not exist in the fresh session.
        let result = tool
            .call(TypeArgs {
                keys: "echo ${TEKTON_RESET_TEST:-UNSET}\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();

        match &result.outcome {
            Outcome::Completed { working_directory, .. } => {
                assert_eq!(
                    result.output,
                    format!("UNSET\r\n{}", expected_prompt("", 0, working_directory)),
                );
            }
            other => panic!("expected Outcome::Completed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn reset_clears_tracked_background_jobs() {
        // Background jobs from the old shell must not carry over after reset.
        let tool = TypeTool::new().spawn().await.unwrap();

        // Start a background job.
        tool.call(TypeArgs {
            keys: "sleep 60 &\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();

        assert!(
            !tool.job_manager.lock().unwrap().jobs.is_empty(),
            "background job must be tracked before reset"
        );

        // Reset the session.
        tool.call(TypeArgs {
            keys: String::new(),
            interactive: false,
            timeout: None,
            reset: true,
        })
        .await
        .unwrap();

        assert!(
            tool.job_manager.lock().unwrap().jobs.is_empty(),
            "tracked jobs must be cleared after reset"
        );
    }

    #[tokio::test]
    async fn reset_recovers_from_stuck_session() {
        // Simulate a stuck session: start an interactive process, then reset
        // instead of trying to clean it up manually.
        let tool = TypeTool::new().spawn().await.unwrap();

        // Start a blocking process (cat waits for input indefinitely).
        tool.call(TypeArgs {
            keys: "cat\n".into(),
            interactive: true,
            timeout: Some(0.1),
            reset: false,
        })
        .await
        .unwrap();

        // Reset instead of trying Ctrl-C/Ctrl-D.
        let reset_result = tool
            .call(TypeArgs {
                keys: String::new(),
                interactive: false,
                timeout: None,
                reset: true,
            })
            .await
            .unwrap();

        assert!(matches!(reset_result.outcome, Outcome::Reset { .. }));

        // The new session must work normally.
        let result = tool
            .call(TypeArgs {
                keys: "echo recovered\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();

        match &result.outcome {
            Outcome::Completed { working_directory, .. } => {
                assert_eq!(
                    result.output,
                    format!("recovered\r\n{}", expected_prompt("", 0, working_directory)),
                );
            }
            other => panic!("expected Outcome::Completed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn reset_discards_old_session_state() {
        // Environment variables, functions, and cwd from the old session
        // must not survive a reset.
        let tool = TypeTool::new().spawn().await.unwrap();

        // Set up state in the old session.
        tool.call(TypeArgs {
            keys: "export TEKTON_OLD=42; cd /tmp\n".into(),
            interactive: false,
            timeout: None,
            reset: false,
        })
        .await
        .unwrap();

        // Reset.
        tool.call(TypeArgs {
            keys: String::new(),
            interactive: false,
            timeout: None,
            reset: true,
        })
        .await
        .unwrap();

        // Verify the env var is gone.
        let result = tool
            .call(TypeArgs {
                keys: "echo ${TEKTON_OLD:-GONE}\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();

        match &result.outcome {
            Outcome::Completed { working_directory, .. } => {
                assert_eq!(
                    result.output,
                    format!("GONE\r\n{}", expected_prompt("", 0, working_directory)),
                );
                // The cwd should be the fresh session's default, not /tmp.
                assert_ne!(working_directory, "/tmp", "cwd must not carry over from old session");
            }
            other => panic!("expected Outcome::Completed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn reset_without_prior_session_spawns_new_one() {
        // If the session was never initialized, reset should spawn one.
        let tool = TypeTool::new();

        let result = tool
            .call(TypeArgs {
                keys: String::new(),
                interactive: false,
                timeout: None,
                reset: true,
            })
            .await
            .unwrap();

        assert!(matches!(result.outcome, Outcome::Reset { .. }));

        // Verify the new session works.
        let result = tool
            .call(TypeArgs {
                keys: "echo alive\n".into(),
                interactive: false,
                timeout: None,
                reset: false,
            })
            .await
            .unwrap();

        match &result.outcome {
            Outcome::Completed { working_directory, .. } => {
                assert_eq!(
                    result.output,
                    format!("alive\r\n{}", expected_prompt("", 0, working_directory)),
                );
            }
            other => panic!("expected Outcome::Completed; got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn reset_with_agent_name_includes_name_in_prompt() {
        let tool = TypeTool::new()
            .with_name("claude")
            .spawn()
            .await
            .unwrap();

        let result = tool
            .call(TypeArgs {
                keys: String::new(),
                interactive: false,
                timeout: None,
                reset: true,
            })
            .await
            .unwrap();

        match &result.outcome {
            Outcome::Reset { working_directory } => {
                let expected = format!(
                    "Session reset\n{}",
                    expected_prompt("claude", 0, working_directory),
                );
                assert_eq!(result.output, expected);
            }
            other => panic!("expected Outcome::Reset; got: {other:?}"),
        }
    }

    #[test]
    fn reset_true_deserializes_from_json() {
        // The LLM sends {"keys": "", "reset": true}. Verify serde handles it.
        let args: TypeArgs = serde_json::from_str(r#"{"keys": "", "reset": true}"#).unwrap();
        assert!(args.reset);
    }

    #[test]
    fn reset_defaults_to_false_in_json() {
        // When reset is omitted from JSON, it defaults to false.
        let args: TypeArgs = serde_json::from_str(r#"{"keys": "ls\n"}"#).unwrap();
        assert!(!args.reset);
    }
}
