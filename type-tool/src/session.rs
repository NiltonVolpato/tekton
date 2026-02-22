//! PTY session management for the Tekton `type` tool.
//!
//! [`PtySession`] wraps an `expectrl::OsSession` (a persistent bash PTY)
//! and provides the sentinel-aware helpers used by [`crate::TypeTool::call`].

use std::{
    collections::HashSet,
    path::PathBuf,
    process::Command,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use expectrl::{Expect, process::Termios, session::OsSession};

/// Serializes PTY fork calls to avoid a thundering-herd of simultaneous
/// `fork()`+`exec()` calls when many sessions start concurrently (e.g. tests).
///
/// Held only for the duration of `Session::spawn()` (the fork itself),
/// released before waiting for the sentinel.  This still allows all sessions
/// to wait for their initial prompt in parallel.
static FORK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

use crate::{TypeError, parse_jobs_p};

/// Regex matching the OSC 999 sentinel emitted by `PROMPT_COMMAND`.
///
/// Format: `ESC ] 999 ; EXIT_CODE ; /current/dir BEL`
/// - Group 1: exit-code digits
/// - Group 2: `$PWD` at prompt time
pub(crate) const SENTINEL_REGEX: &str = r"\x1b\]999;(\d+);([^\x07]*)\x07";

/// Drain timeout after killing the foreground process.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Poll interval for [`poll_for_sentinel`].
///
/// `expectrl::expect()` uses a CPU-spinning busy-wait loop, which saturates
/// cores when many sessions are active concurrently (e.g. during tests).
/// [`poll_for_sentinel`] uses non-blocking `check()` calls separated by this
/// sleep instead, keeping CPU usage low while still detecting the sentinel
/// within ~10 ms of its arrival.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// A persistent bash PTY session.
pub(crate) struct PtySession {
    /// The underlying expectrl session.
    pub(crate) session: OsSession,
    /// PID of the bash shell (used for `pgrep -P`).
    pub(crate) shell_pid: u32,
    /// Path to the SIGCHLD named pipe (`/tmp/tekton-jobs-<shell_pid>`).
    #[allow(dead_code)]
    pub(crate) pipe_path: PathBuf,
}

impl PtySession {
    /// Spawn a new bash PTY session and wait for the initial prompt sentinel.
    ///
    /// All blocking PTY I/O runs inside `spawn_blocking` so it does not starve
    /// the async executor when multiple sessions are created concurrently.
    pub(crate) async fn spawn() -> Result<Self, TypeError> {
        tokio::task::spawn_blocking(Self::spawn_blocking_inner)
            .await
            .map_err(|e| TypeError::PtySpawn(format!("spawn_blocking panicked: {e}")))?
    }

    fn spawn_blocking_inner() -> Result<Self, TypeError> {
        // 1. Write the init script to a temp file.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let init_path = format!("/tmp/tekton-init-{nanos}.sh");

        // The PROMPT_COMMAND sentinel format: ESC]999;EXIT_CODE;$PWD BEL
        //
        // NOTE: The SIGCHLD trap + named pipe for background job notifications
        // is intentionally omitted here.  The pipe's open() call blocks until a
        // reader opens the other end; without the harness running (e.g. in tests)
        // the trap handler would deadlock the shell the first time any external
        // command exits.  The SIGCHLD mechanism will be wired back once the
        // harness implementation exists.
        let init_content = concat!(
            "PROMPT_COMMAND='printf \"\\033]999;%d;%s\\007\" $? \"$PWD\"'\n",
            "set +m\n",
            "PS1='[$?] \\u@\\h:\\w \\$ '\n",
        );

        std::fs::write(&init_path, init_content).map_err(|e| TypeError::PtySpawn(e.to_string()))?;

        // 2. Spawn bash with the init file.
        //    Acquire FORK_LOCK only for the fork/exec itself so concurrent PTY
        //    allocations don't create a thundering-herd of simultaneous forks.
        let mut cmd = Command::new("bash");
        cmd.arg("--init-file").arg(&init_path);

        let mut session = {
            let _fork_guard = FORK_LOCK
                .get_or_init(|| Mutex::new(()))
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            expectrl::Session::spawn(cmd).map_err(|e| TypeError::PtySpawn(e.to_string()))?
        }; // _fork_guard released here; next spawn may fork concurrently

        // 3. Disable echo so typed input doesn't appear in the output stream.
        session
            .set_echo(false)
            .map_err(|e| TypeError::PtySpawn(format!("set_echo failed: {e}")))?;

        // 4. Derive pipe path from the shell PID.
        let shell_pid = session.get_process().pid().as_raw() as u32;
        let pipe_path = PathBuf::from(format!("/tmp/tekton-jobs-{shell_pid}"));

        // 5. Wait up to 15 s for the first sentinel (shell ready).
        //    Use the polled helper to avoid busy-waiting under concurrent load.
        poll_for_sentinel(&mut session, Duration::from_secs(15))
            .map_err(|e| TypeError::PtySpawn(format!("waiting for initial prompt: {e}")))?;

        // 6. Clean up the temp init file.
        let _ = std::fs::remove_file(&init_path);

        Ok(Self {
            session,
            shell_pid,
            pipe_path,
        })
    }
}

/// Wait for the next OSC 999 prompt sentinel, sleeping between polls.
///
/// Unlike `session.expect()` (which busy-waits), this function sleeps
/// [`POLL_INTERVAL`] between calls to `session.check()`.  This keeps CPU
/// usage low when many sessions are active simultaneously.
///
/// Returns the [`expectrl::Captures`] on success, or
/// [`expectrl::Error::ExpectTimeout`] if `timeout` elapses first.
pub(crate) fn poll_for_sentinel(
    session: &mut OsSession,
    timeout: Duration,
) -> Result<expectrl::Captures, expectrl::Error> {
    let start = Instant::now();
    loop {
        let captures = session.check(expectrl::Regex(SENTINEL_REGEX))?;
        if !captures.is_empty() {
            return Ok(captures);
        }
        if start.elapsed() >= timeout {
            return Err(expectrl::Error::ExpectTimeout);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Extract `(exit_code, cwd)` from OSC sentinel regex captures.
///
/// The regex has two capture groups:
/// - Group 1: exit code digits
/// - Group 2: `$PWD` path
pub(crate) fn parse_sentinel(captures: &expectrl::Captures) -> (i32, String) {
    let exit_code = captures
        .get(1)
        .and_then(|b| std::str::from_utf8(b).ok())
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    let cwd = captures
        .get(2)
        .and_then(|b| std::str::from_utf8(b).ok())
        .unwrap_or("")
        .to_string();
    (exit_code, cwd)
}

/// Run `jobs -p` on the session and return the set of active background PIDs.
///
/// Sends the built-in and waits for the next prompt sentinel.
/// Returns an empty set on any error or timeout.
pub(crate) fn run_jobs_p_sync(session: &mut OsSession) -> HashSet<u32> {
    if session.send("jobs -p\n").is_err() {
        return HashSet::new();
    }
    match poll_for_sentinel(session, Duration::from_secs(5)) {
        Ok(captures) => parse_jobs_p(&String::from_utf8_lossy(captures.before())),
        Err(_) => HashSet::new(),
    }
}

/// Send SIGTERM then SIGKILL to foreground children of `shell_pid`,
/// skipping known background PIDs.
// TODO: Use sysinfo crate instead for pgrep and kill.
pub(crate) fn kill_foreground(shell_pid: u32, background_pids: &HashSet<u32>) {
    let output = Command::new("pgrep")
        .args(["-P", &shell_pid.to_string()])
        .output();

    let foreground_pids: Vec<u32> = match output {
        Ok(out) => parse_jobs_p(&String::from_utf8_lossy(&out.stdout))
            .into_iter()
            .filter(|pid| !background_pids.contains(pid))
            .collect(),
        Err(_) => return,
    };

    // SIGTERM first.
    for &pid in &foreground_pids {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    }

    // Give processes a moment to exit gracefully.
    std::thread::sleep(Duration::from_millis(500));

    // SIGKILL any survivors.
    for &pid in &foreground_pids {
        if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
            unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        }
    }
}

/// Drain the PTY after `kill_foreground` to return bash to a clean prompt.
///
/// Returns the `$PWD` reported in the post-kill sentinel, or `None` on timeout.
pub(crate) fn drain_after_kill(session: &mut OsSession) -> Option<String> {
    poll_for_sentinel(session, DRAIN_TIMEOUT)
        .ok()
        .map(|c| parse_sentinel(&c).1)
}
