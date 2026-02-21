//! The Tekton `type` tool — sends keystrokes to a persistent PTY session.
//!
//! This is the single LLM-facing tool in the Tekton framework. The model sends
//! keystrokes to a terminal the same way a human would: by typing. Everything
//! else — file I/O, web search, code execution, sub-agents — is a shell command.
//!
//! # Example
//!
//! ```no_run
//! use tekton_type_tool::{JobNotification, TypeTool};
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
//!     let agent = AgentBuilder::new(model)
//!         .preamble(
//!             "You're looking at a terminal. The only tool you have is `type`, \
//!              which sends keystrokes to the terminal.\n\n\
//!              The terminal says:\n\n\
//!              Welcome back Claude! For help run the command `help`.\n\
//!              [0] claude@alpha:/Users/nilton/src/tekton $",
//!         )
//!         .tool(
//!             TypeTool::new().with_job_callback(|notif: JobNotification| {
//!                 // Forward the notification to the model on its next turn,
//!                 // inject it into the UI, log it, etc.
//!                 eprintln!(
//!                     "Background job {} ({}) exited with code {}",
//!                     notif.pid, notif.command, notif.exit_code,
//!                 );
//!             }),
//!         )
//!         .build();
//! }
//! ```

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

/// Non-interactive timeout in seconds: how long to wait before killing the foreground process.
const DEFAULT_TIMEOUT_SECS: f64 = 300.0;

/// Interactive timeout in seconds: how long to wait before returning partial output to the model.
const DEFAULT_INTERACTIVE_TIMEOUT_SECS: f64 = 5.0;

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

/// A background job completion event, delivered via the callback on [`TypeTool`].
///
/// The harness reads these from the named pipe that the shell's `SIGCHLD` trap
/// writes to; they arrive independently of any `type` call and so are delivered
/// out-of-band through a caller-supplied callback rather than in [`TypeOutput`].
#[derive(Debug, Clone)]
pub struct JobNotification {
    /// PID of the completed background process.
    pub pid: u32,
    /// Exit code returned by the process.
    pub exit_code: i32,
    /// The command string, as reported by the shell trap.
    pub command: String,
}

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
/// Background job completions arrive asynchronously from the named pipe and are
/// delivered via an optional callback set with [`TypeTool::with_job_callback`].
///
/// Construct via [`TypeTool::new`] and pass to the agent builder's `.tool()` method.
pub struct TypeTool {
    // TODO: hold the PTY session handle (e.g., Arc<Mutex<PtySession>>)
    // The session will be initialized before the first tool call and kept alive
    // for the lifetime of the agent.
    on_job_complete: Option<Box<dyn Fn(JobNotification) + Send + Sync>>,
}

impl TypeTool {
    /// Create a new `TypeTool` with no job-notification callback.
    ///
    /// The underlying PTY session is not started here; it will be lazily
    /// initialized on the first call, or you can add an explicit `start()`
    /// method when the harness is implemented.
    pub fn new() -> Self {
        Self {
            on_job_complete: None,
        }
    }

    /// Register a callback invoked whenever a background job completes.
    ///
    /// The harness reads from the shell's named pipe on a background task and
    /// calls this whenever a `SIGCHLD` notification arrives — independently of
    /// any `type` call. Use it to forward notifications to the model on its next
    /// turn, log them, update UI state, etc.
    pub fn with_job_callback(
        mut self,
        callback: impl Fn(JobNotification) + Send + Sync + 'static,
    ) -> Self {
        self.on_job_complete = Some(Box::new(callback));
        self
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
        let _timeout = args.resolved_timeout();

        // TODO: implement actual PTY interaction:
        //   1. Write `args.keys` to the PTY stdin (interpreting escape sequences).
        //   2. Read PTY output, buffering bytes.
        //   3. Wait for either:
        //      a. OSC sentinel `\e]999;EXIT_CODE\a` → command finished; return output + exit code.
        //      b. Timeout with interactive=false → SIGKILL foreground process group; return output + timeout message.
        //      c. Timeout with interactive=true → return partial output; model decides next keystroke.
        //   4. Return TypeOutput.
        //
        // Job notifications from the named pipe are delivered separately via
        // self.on_job_complete, driven by a background task in the harness.

        Err(TypeError::SessionNotInitialized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

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

    #[tokio::test]
    async fn tool_definition_has_correct_name() {
        let tool = TypeTool::new();
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "type");
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
        let args = TypeArgs {
            keys: "ls\n".to_string(),
            interactive: false,
            timeout: None,
        };
        let result = tool.call(args).await;
        assert!(matches!(result, Err(TypeError::SessionNotInitialized)));
    }

    #[test]
    fn job_callback_is_invoked_with_correct_data() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);

        let tool = TypeTool::new().with_job_callback(move |notif| {
            received_clone.lock().unwrap().push(notif);
        });

        // Simulate the harness firing the callback.
        let notif = JobNotification {
            pid: 12345,
            exit_code: 0,
            command: "long_task".to_string(),
        };
        tool.on_job_complete.as_ref().unwrap()(notif);

        let notifications = received.lock().unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].pid, 12345);
        assert_eq!(notifications[0].exit_code, 0);
        assert_eq!(notifications[0].command, "long_task");
    }

    #[test]
    fn no_callback_by_default() {
        let tool = TypeTool::new();
        assert!(tool.on_job_complete.is_none());
    }
}
