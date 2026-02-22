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
        // Combine PID with subsecond nanos for a collision-resistant filename.
        // Using subsec_nanos alone wraps every second; two spawns exactly 1 s
        // apart would collide.  The PID component makes the name unique per
        // process even across second boundaries.
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let init_path = format!("/tmp/tekton-init-{pid}-{nanos}.sh");

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
        // -1 (not 0) so a corrupted/truncated sentinel is distinguishable from
        // a genuinely successful command (exit 0).
        .unwrap_or(-1);
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
            // PID 0 is valid as a u32 but kill(0, sig) sends the signal to the
            // entire process group, which would kill the harness and its parent.
            .filter(|&pid| pid != 0)
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashSet, time::Duration};

    // ── SENTINEL_REGEX: pure pattern validation ────────────────────────────
    //
    // These tests use the `regex` crate (dev-dependency) to exercise the regex
    // pattern itself, completely independent of the expectrl session machinery.
    // Any change to SENTINEL_REGEX that breaks the expected capture groups will
    // fail here before a PTY is ever opened.

    fn re() -> regex::Regex {
        regex::Regex::new(SENTINEL_REGEX).unwrap()
    }

    /// Build a sentinel string the same way bash's PROMPT_COMMAND does:
    /// `printf "\033]999;%d;%s\007" $? "$PWD"`
    fn make_sentinel(exit_code: u32, cwd: &str) -> String {
        format!("\x1b]999;{exit_code};{cwd}\x07")
    }

    #[test]
    fn sentinel_regex_matches_well_formed_sentinel() {
        assert!(re().is_match(&make_sentinel(0, "/home/user")));
    }

    #[test]
    fn sentinel_regex_captures_exit_code_zero() {
        let sentinel = make_sentinel(0, "/");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[1], "0");
    }

    #[test]
    fn sentinel_regex_captures_exit_code_one() {
        let sentinel = make_sentinel(1, "/");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[1], "1");
    }

    #[test]
    fn sentinel_regex_captures_exit_code_127() {
        // 127 = "command not found" — the most important non-zero bash code.
        let sentinel = make_sentinel(127, "/tmp");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[1], "127");
    }

    #[test]
    fn sentinel_regex_captures_exit_code_255() {
        // 255 is the maximum valid bash exit code (exit codes are 8-bit).
        let sentinel = make_sentinel(255, "/");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[1], "255");
    }

    #[test]
    fn sentinel_regex_captures_cwd() {
        let sentinel = make_sentinel(0, "/home/user/project");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[2], "/home/user/project");
    }

    #[test]
    fn sentinel_regex_captures_empty_cwd() {
        // An empty CWD is syntactically valid (though unusual in practice).
        let sentinel = make_sentinel(0, "");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[2], "");
    }

    #[test]
    fn sentinel_regex_captures_cwd_with_spaces() {
        // Paths with spaces are legal on all platforms and must be captured whole.
        let sentinel = make_sentinel(0, "/home/user/my project");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[2], "/home/user/my project");
    }

    #[test]
    fn sentinel_regex_captures_cwd_with_unicode() {
        // Non-ASCII paths (common on macOS and Linux) must be captured verbatim.
        let sentinel = make_sentinel(0, "/home/用户/项目");
        let caps = re().captures(&sentinel).unwrap();
        assert_eq!(&caps[2], "/home/用户/项目");
    }

    #[test]
    fn sentinel_regex_cwd_stops_at_first_bel_character() {
        // [^\x07]* must stop at the very first BEL in the stream.
        // If a path somehow contained BEL, only the part before it is captured.
        // This guards against a greedy variant swallowing the terminator.
        let s = "\x1b]999;0;/good/path\x07spurious_data\x07";
        let caps = re().captures(s).unwrap();
        assert_eq!(&caps[2], "/good/path");
    }

    #[test]
    fn sentinel_regex_does_not_match_wrong_osc_number() {
        // OSC 998 is not our sentinel. A bug in the regex might match any OSC code.
        assert!(!re().is_match("\x1b]998;0;/tmp\x07"));
    }

    #[test]
    fn sentinel_regex_does_not_match_missing_bel_terminator() {
        // An unterminated sentinel (partial read / truncated stream) must not match.
        assert!(!re().is_match("\x1b]999;0;/tmp"));
    }

    #[test]
    fn sentinel_regex_does_not_match_missing_esc_prefix() {
        // Without ESC this is not a valid OSC sequence.
        assert!(!re().is_match("]999;0;/tmp\x07"));
    }

    #[test]
    fn sentinel_regex_does_not_match_non_digit_exit_code() {
        // \d+ requires digits; an alphabetic exit code must not match.
        assert!(!re().is_match("\x1b]999;abc;/tmp\x07"));
    }

    #[test]
    fn sentinel_regex_does_not_match_empty_exit_code() {
        // \d+ (one-or-more) must reject an empty exit-code field.
        assert!(!re().is_match("\x1b]999;;/tmp\x07"));
    }

    #[test]
    fn sentinel_regex_matches_large_decimal_exit_code_overflowing_i32() {
        // \d+ accepts any run of digits, including values that overflow i32.
        // The regex accepts them; parse_sentinel is then responsible for
        // handling the integer parse error gracefully (see integration test below).
        assert!(re().is_match("\x1b]999;9999999999;/tmp\x07"));
    }

    // ── parse_sentinel: integration tests ─────────────────────────────────
    //
    // These spawn a real PTY and call parse_sentinel on the Captures returned
    // by poll_for_sentinel, testing the full parsing path end-to-end.

    #[tokio::test]
    async fn parse_sentinel_exit_code_zero_for_true() {
        let pty = PtySession::spawn().await.unwrap();
        let (exit_code, _) = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            pty.session.send(b"true\n").unwrap();
            let caps = poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            parse_sentinel(&caps)
        })
        .await
        .unwrap();
        assert_eq!(exit_code, 0);
    }

    #[tokio::test]
    async fn parse_sentinel_exit_code_one_for_false() {
        let pty = PtySession::spawn().await.unwrap();
        let (exit_code, _) = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            pty.session.send(b"false\n").unwrap();
            let caps = poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            parse_sentinel(&caps)
        })
        .await
        .unwrap();
        assert_eq!(exit_code, 1);
    }

    #[tokio::test]
    async fn parse_sentinel_arbitrary_exit_code() {
        // Verifies that exit codes other than 0 and 1 are faithfully relayed.
        let pty = PtySession::spawn().await.unwrap();
        let (exit_code, _) = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            pty.session.send(b"sh -c 'exit 99'\n").unwrap();
            let caps = poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            parse_sentinel(&caps)
        })
        .await
        .unwrap();
        assert_eq!(exit_code, 99);
    }

    #[tokio::test]
    async fn parse_sentinel_cwd_is_root_after_cd_slash() {
        // The sentinel's CWD field must reflect the real working directory.
        // Using "/" avoids the macOS /tmp → /private/tmp symlink ambiguity.
        let pty = PtySession::spawn().await.unwrap();
        let (_, cwd) = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            pty.session.send(b"cd /\n").unwrap();
            let caps = poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            parse_sentinel(&caps)
        })
        .await
        .unwrap();
        assert_eq!(cwd, "/", "sentinel CWD must reflect the cd target");
    }

    #[tokio::test]
    async fn parse_sentinel_overflow_exit_code_must_not_silently_become_zero() {
        // BUG (known): parse_sentinel uses .unwrap_or(0) for the exit code.
        // When the sentinel's exit-code field contains a value that overflows i32
        // (e.g. "9999999999"), parse::<i32>() returns Err and the result is
        // silently 0 — identical to a genuinely successful command exit.
        // This masks parse errors and can mislead the harness.
        //
        // Fix: return Option<i32> where None means "parse failed", or use
        // a reserved value like -1 that is impossible for a real bash exit code
        // (bash exit codes are always 0–255).
        //
        // THIS TEST WILL FAIL with the current implementation. That is intentional:
        // it documents the bug and will pass once the fix is applied.
        let pty = PtySession::spawn().await.unwrap();
        let exit_code = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            // Override PROMPT_COMMAND to emit an exit-code field that overflows i32.
            pty.session
                .send(b"PROMPT_COMMAND='printf \"\\033]999;9999999999;$PWD\\007\"'\n")
                .unwrap();
            // Consume the sentinel produced by this PROMPT_COMMAND change itself.
            poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            // Run any command; the sentinel will carry "9999999999" as the exit code.
            pty.session.send(b"false\n").unwrap();
            let caps = poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            let (exit_code, _) = parse_sentinel(&caps);
            exit_code
        })
        .await
        .unwrap();
        assert_ne!(
            exit_code, 0,
            "an overflowing exit-code field must not silently become 0; \
             fix: change parse_sentinel to return Option<i32> with None on parse error \
             (bash exit codes are always 0-255 so -1 works as an explicit error sentinel)"
        );
    }

    // ── poll_for_sentinel: timeout behaviour ──────────────────────────────

    #[tokio::test]
    async fn poll_for_sentinel_times_out_when_no_new_sentinel_emitted() {
        // After spawn the initial sentinel is consumed. With no command sent,
        // no new sentinel arrives and poll_for_sentinel must return ExpectTimeout.
        let pty = PtySession::spawn().await.unwrap();
        let result = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            // No command sent — no sentinel will arrive.
            poll_for_sentinel(&mut pty.session, Duration::from_millis(200))
        })
        .await
        .unwrap();
        assert!(
            matches!(result, Err(expectrl::Error::ExpectTimeout)),
            "poll_for_sentinel must return ExpectTimeout when no sentinel arrives; \
             got: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn poll_for_sentinel_does_not_dramatically_overshoot_timeout() {
        // Guards against an implementation that loops with a fixed sleep much
        // larger than POLL_INTERVAL, making the effective timeout 10× the request.
        let pty = PtySession::spawn().await.unwrap();
        let start = std::time::Instant::now();
        let _result = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            poll_for_sentinel(&mut pty.session, Duration::from_millis(150))
        })
        .await
        .unwrap();
        let elapsed = start.elapsed();
        // Allow 3× margin for slow CI, but a 10× overshoot (1.5 s) must fail.
        assert!(
            elapsed < Duration::from_millis(500),
            "150 ms timeout should not overshoot dramatically; elapsed: {elapsed:?}"
        );
    }

    // ── run_jobs_p_sync ───────────────────────────────────────────────────

    #[tokio::test]
    async fn run_jobs_p_sync_empty_when_no_background_jobs() {
        let pty = PtySession::spawn().await.unwrap();
        let pids = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            run_jobs_p_sync(&mut pty.session)
        })
        .await
        .unwrap();
        assert!(pids.is_empty(), "freshly spawned shell must report no background jobs");
    }

    #[tokio::test]
    async fn run_jobs_p_sync_returns_pid_of_running_background_job() {
        let pty = PtySession::spawn().await.unwrap();
        let pids = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            // Start a long-running background job.
            pty.session.send(b"sleep 9999 &\n").unwrap();
            // Shell returns to prompt after backgrounding; wait for sentinel.
            poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            run_jobs_p_sync(&mut pty.session)
        })
        .await
        .unwrap();
        assert_eq!(pids.len(), 1, "exactly one background job must be tracked");
    }

    #[tokio::test]
    async fn run_jobs_p_sync_job_absent_after_it_completes() {
        // A short-lived background job must disappear from `jobs -p` once done.
        let pty = PtySession::spawn().await.unwrap();
        let pids = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            pty.session.send(b"sleep 0.1 &\n").unwrap();
            poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            // Wait long enough for the background job to finish.
            std::thread::sleep(Duration::from_millis(500));
            run_jobs_p_sync(&mut pty.session)
        })
        .await
        .unwrap();
        assert!(pids.is_empty(), "completed background job must not appear in jobs -p");
    }

    // ── kill_foreground ───────────────────────────────────────────────────

    #[tokio::test]
    async fn kill_foreground_terminates_running_foreground_process() {
        let pty = PtySession::spawn().await.unwrap();
        let shell_pid = pty.shell_pid;
        let result = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            // Start a foreground process that would run indefinitely.
            pty.session.send(b"sleep 9999\n").unwrap();
            std::thread::sleep(Duration::from_millis(100)); // let it start

            // Kill foreground processes (no background PIDs to spare).
            kill_foreground(shell_pid, &HashSet::new());

            // Bash regains control, runs PROMPT_COMMAND, emits a sentinel.
            poll_for_sentinel(&mut pty.session, Duration::from_secs(5))
        })
        .await
        .unwrap();
        assert!(
            result.is_ok(),
            "shell must return to prompt after kill_foreground; got: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn kill_foreground_spares_processes_in_background_pid_set() {
        // A PID listed in `background_pids` must survive kill_foreground.
        // This is the mechanism that prevents the harness from accidentally
        // killing long-running background tasks when a foreground command times out.
        let pty = PtySession::spawn().await.unwrap();
        let shell_pid = pty.shell_pid;
        let bg_survived = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;

            // Start a background job and record its PID.
            pty.session.send(b"sleep 9999 &\n").unwrap();
            poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            let bg_pids = run_jobs_p_sync(&mut pty.session);
            assert_eq!(bg_pids.len(), 1, "expected exactly one background job");
            let bg_pid = *bg_pids.iter().next().unwrap();

            // Start a foreground process.
            pty.session.send(b"sleep 9999\n").unwrap();
            std::thread::sleep(Duration::from_millis(100));

            // Kill foreground only — background PID is in the protection set.
            kill_foreground(shell_pid, &bg_pids);
            poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();

            // The background job must still be running.
            let remaining = run_jobs_p_sync(&mut pty.session);
            remaining.contains(&bg_pid)
        })
        .await
        .unwrap();
        assert!(bg_survived, "background PID must survive kill_foreground");
    }

    // ── drain_after_kill ──────────────────────────────────────────────────

    #[tokio::test]
    async fn drain_after_kill_returns_cwd_from_post_kill_sentinel() {
        // After kill_foreground, bash runs PROMPT_COMMAND and emits a new sentinel.
        // drain_after_kill must extract the CWD from that sentinel and return it.
        let pty = PtySession::spawn().await.unwrap();
        let shell_pid = pty.shell_pid;
        let cwd = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;

            // Navigate to a known directory first.
            pty.session.send(b"cd /\n").unwrap();
            poll_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();

            // Start and immediately kill a foreground process.
            pty.session.send(b"sleep 9999\n").unwrap();
            std::thread::sleep(Duration::from_millis(100));
            kill_foreground(shell_pid, &HashSet::new());

            drain_after_kill(&mut pty.session)
        })
        .await
        .unwrap();
        assert!(
            cwd.is_some(),
            "drain_after_kill must return Some(cwd) after a successful kill"
        );
        assert_eq!(
            cwd.unwrap(),
            "/",
            "drained CWD must match the directory set before the kill"
        );
    }
}
