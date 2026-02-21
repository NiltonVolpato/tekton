//! The Tekton `type` tool — sends keystrokes to a persistent PTY session.
//!
//! This is the single LLM-facing tool in the Tekton framework. The model sends
//! keystrokes to a terminal the same way a human would: by typing. Everything
//! else — file I/O, web search, code execution, sub-agents — is a shell command.
//!
//! # Harness wiring
//!
//! After constructing the tool and before passing it to the agent builder,
//! the harness retrieves two shared handles:
//!
//! - [`TypeTool::job_manager`] — to call [`JobManager::sync`] from the idle path.
//! - [`TypeTool::is_running`] — to decide whether to act or ignore a pipe signal.
//!
//! The harness spawns a background task that reads the named pipe and, on each
//! signal, checks `is_running`:
//!
//! - **Running**: ignore; [`TypeTool::call`] will sync the manager at the end of
//!   the current command anyway.
//! - **Idle**: run `jobs -p` on the PTY, parse the output, call
//!   [`JobManager::sync`]. If any completions are detected, the callback fires
//!   and the harness can queue a proactive message to the agent.
//!
//! ```no_run
//! use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
//! use std::collections::HashSet;
//! use tekton_type_tool::{JobManager, TypeTool, parse_jobs_p};
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
//!     let tool = TypeTool::new().with_job_callback(|notif| {
//!         // Queue a proactive message to the agent, wake it up if idle, etc.
//!         eprintln!("job {} done (exit {:?})", notif.pid, notif.exit_code);
//!     });
//!
//!     // Retrieve shared handles before moving `tool` into the agent builder.
//!     let job_manager = tool.job_manager();
//!     let is_running = tool.is_running();
//!
//!     // Harness task: read the named pipe and sync the manager when idle.
//!     tokio::spawn(async move {
//!         loop {
//!             // TODO: read one signal byte from the named pipe.
//!             if !is_running.load(Ordering::SeqCst) {
//!                 // TODO: run `jobs -p` on the PTY and parse the output.
//!                 let active_pids: HashSet<u32> = HashSet::new(); // stub
//!                 job_manager.lock().unwrap().sync(&active_pids);
//!             }
//!             // If running: call() will sync at the end of the command.
//!         }
//!     });
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
//! }
//! ```

use std::collections::HashSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

/// Non-interactive timeout in seconds: how long to wait before killing the foreground process.
const DEFAULT_TIMEOUT_SECS: f64 = 300.0;

/// Interactive timeout in seconds: how long to wait before returning partial output to the model.
const DEFAULT_INTERACTIVE_TIMEOUT_SECS: f64 = 5.0;

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

    /// Exit code of the last foreground command, as read from the OSC sentinel.
    ///
    /// `None` if the command timed out before emitting a sentinel.
    pub exit_code: Option<i32>,

    /// If the call timed out, a human-readable explanation.
    pub timeout_message: Option<String>,
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
/// The shell's `SIGCHLD` trap writes a signal byte to the named pipe; that byte
/// carries no structured data. Completions are detected by diffing the set of
/// PIDs reported by `jobs -p` against the set of known tracked jobs.
///
/// [`JobManager::sync`] is the single update point: call it after every
/// foreground command completes (via [`TypeTool::call`]) and from the harness
/// task when the signal arrives while the tool is idle.
pub struct JobManager {
    jobs: HashMap<u32, Job>,
    on_complete: Option<Box<dyn Fn(JobNotification) + Send + Sync>>,
}

use std::collections::HashMap;

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
    /// Call this after every foreground command and from the idle signal path.
    pub fn sync(&mut self, active_pids: &HashSet<u32>) {
        // Detect completions: tracked PIDs no longer in active set.
        let completed: Vec<u32> = self
            .jobs
            .keys()
            .filter(|pid| !active_pids.contains(pid))
            .copied()
            .collect();

        for pid in completed {
            if let Some(job) = self.jobs.remove(&pid) {
                if let Some(cb) = &self.on_complete {
                    cb(JobNotification {
                        pid,
                        // TODO: use waitpid(WNOHANG) to get the actual exit code.
                        exit_code: None,
                        command: job.command,
                    });
                }
            }
        }

        // Track newly appeared PIDs.
        for &pid in active_pids {
            self.jobs.entry(pid).or_insert_with(|| Job {
                pid,
                // TODO: populate from `jobs -l` or `ps` output for richer info.
                command: String::new(),
            });
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
}

/// The Tekton `type` tool.
///
/// Wraps a persistent PTY session. The model calls this tool to send keystrokes
/// to the terminal; the harness captures output until the shell emits a sentinel
/// (command complete) or the timeout fires.
///
/// After each command, `call` runs `jobs -p` internally and syncs the
/// [`JobManager`]. The harness task does the same when the named pipe signals
/// while the tool is idle (see [`TypeTool::is_running`]).
pub struct TypeTool {
    // TODO: hold the PTY session handle (e.g., Arc<Mutex<PtySession>>).
    // Shared with the harness task so it can run `jobs -p` on the idle path.
    job_manager: Arc<Mutex<JobManager>>,

    /// `true` while a foreground command is executing inside `call`.
    ///
    /// The harness task checks this flag when the named pipe fires:
    /// if `true`, it ignores the signal; if `false`, it runs `jobs -p` and
    /// calls [`JobManager::sync`].
    is_running: Arc<AtomicBool>,
}

impl TypeTool {
    /// Create a new `TypeTool` with an empty [`JobManager`] and no callback.
    pub fn new() -> Self {
        Self {
            job_manager: Arc::new(Mutex::new(JobManager::new())),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Register a callback invoked whenever a background job completes.
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
            job_manager: Arc::new(Mutex::new(manager)),
            is_running: self.is_running,
        }
    }

    /// Return a shared handle to the [`JobManager`].
    ///
    /// The harness task uses this to call [`JobManager::sync`] on the idle path.
    pub fn job_manager(&self) -> Arc<Mutex<JobManager>> {
        Arc::clone(&self.job_manager)
    }

    /// Return a shared handle to the "is running" flag.
    ///
    /// The harness task reads this when the named pipe fires:
    /// - `true` → a `call` is in progress; ignore the signal.
    /// - `false` → the tool is idle; run `jobs -p` and sync the manager.
    pub fn is_running(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_running)
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
        self.is_running.store(true, Ordering::SeqCst);

        // TODO: implement actual PTY interaction:
        //   1. Write `args.keys` to the PTY stdin (interpreting escape sequences).
        //   2. Read PTY output, buffering bytes.
        //   3. Wait for either:
        //      a. OSC sentinel `\e]999;EXIT_CODE\a` → command finished.
        //      b. Timeout with interactive=false → SIGKILL foreground process group.
        //      c. Timeout with interactive=true → return partial output.

        self.is_running.store(false, Ordering::SeqCst);

        // Sync the job manager with whatever background jobs are now running.
        // TODO: run `jobs -p` on the PTY and parse the output.
        let active_pids = HashSet::new(); // stub
        self.job_manager.lock().unwrap().sync(&active_pids);

        let _timeout = args.resolved_timeout();
        Err(TypeError::SessionNotInitialized)
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

    #[test]
    fn is_running_starts_false() {
        let tool = TypeTool::new();
        assert!(!tool.is_running().load(Ordering::SeqCst));
    }
}
