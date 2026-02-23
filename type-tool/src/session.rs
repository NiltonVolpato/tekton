//! PTY session management for the Tekton `type` tool.
//!
//! [`PtySession`] wraps an `expectrl::OsSession` (a persistent bash PTY)
//! and provides the sentinel-aware helpers used by [`crate::TypeTool::call`].

use std::{collections::HashSet, process::Command, time::Duration};

use sysinfo::{Pid as SysPid, ProcessesToUpdate, System};

use expectrl::{Expect, process::Termios, session::OsSession};

use crate::{TypeError, parse_jobs_p};

/// Regex matching the OSC 999 sentinel emitted by `PROMPT_COMMAND`.
///
/// Format: `ESC ] 999 ; EXIT_CODE ; /current/dir BEL`
/// - Group 1: exit-code digits
/// - Group 2: `$PWD` at prompt time
pub(crate) const SENTINEL_REGEX: &str = r"\x1b\]999;(\d+);([^\x07]*)\x07";

/// Drain timeout after killing the foreground process.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// A persistent bash PTY session.
pub(crate) struct PtySession {
    /// The underlying expectrl session.
    pub(crate) session: OsSession,
    /// PID of the bash shell (used for `pgrep -P`).
    pub(crate) shell_pid: u32,
    /// Working directory from the initial sentinel (`$PWD` at shell startup).
    pub(crate) working_directory: String,
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

    pub(crate) fn spawn_blocking_inner() -> Result<Self, TypeError> {
        // 1. Spawn bash with no startup files.
        let mut cmd = Command::new("bash");
        cmd.args(["--norc", "--noprofile", "--noediting"]);

        let mut session =
            expectrl::Session::spawn(cmd).map_err(|e| TypeError::PtySpawn(e.to_string()))?;

        // 2. Disable echo so typed input doesn't appear in the output stream.
        session
            .set_echo(false)
            .map_err(|e| TypeError::PtySpawn(format!("set_echo failed: {e}")))?;

        // 3. Get the shell PID.
        let shell_pid = session.get_process().pid().as_raw() as u32;

        // 4. Wait for bash's default prompt so the shell is ready for input.
        //    Without this, send_line can race with bash's startup and the
        //    init commands may arrive before bash is fully initialized —
        //    breaking foreground process group management.
        use expectrl::Expect;
        session.set_expect_timeout(Some(Duration::from_secs(1)));
        let _ = session.expect(expectrl::Regex(r"\$ "));

        // 5. Send the init commands inline. PS1='' ensures no shell prompt
        //    leaks into the output stream — the harness builds and appends its
        //    own prompt from sentinel data.
        session
            .send_line(
                "PROMPT_COMMAND='printf \"\\033]999;%d;%s\\007\" $? \"$PWD\"'; PS1=''; set +m; bind 'set enable-bracketed-paste off' 2>/dev/null",
            )
            .map_err(|e| TypeError::PtySpawn(format!("send init line failed: {e}")))?;

        // 6. Wait up to 15 s for the first sentinel (shell ready).
        let captures = wait_for_sentinel(&mut session, Duration::from_secs(15))
            .map_err(|e| TypeError::PtySpawn(format!("waiting for initial prompt: {e}")))?;
        let (_, working_directory) = parse_sentinel(&captures);

        Ok(Self {
            session,
            shell_pid,
            working_directory,
        })
    }
}

/// Wait for the next OSC 999 prompt sentinel.
///
/// Uses expectrl's built-in `expect()` with its timeout mechanism.
///
/// TODO: Verify that `expect()` does not cause high CPU usage under concurrent
/// load. The old polling implementation assumed it did, but that was never
/// confirmed.
///
/// Returns the [`expectrl::Captures`] on success, or
/// [`expectrl::Error::ExpectTimeout`] if `timeout` elapses first.
pub(crate) fn wait_for_sentinel(
    session: &mut OsSession,
    timeout: Duration,
) -> Result<expectrl::Captures, expectrl::Error> {
    session.set_expect_timeout(Some(timeout));
    session.expect(expectrl::Regex(SENTINEL_REGEX))
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
/// First runs `true` to force a full prompt cycle so bash reaps any
/// recently-completed background jobs before we query the job table.
/// Returns `None` on any error or timeout — the caller must skip the
/// `sync()` call to avoid spurious completion callbacks.
pub(crate) fn run_jobs_p_sync(session: &mut OsSession) -> Option<HashSet<u32>> {
    // Force bash to reap completed background jobs. With `set +m` (monitor
    // mode off), bash only reaps during a full prompt cycle — PROMPT_COMMAND
    // + PS1. Running `true` triggers that cycle.
    session.send_line("true").ok()?;
    wait_for_sentinel(session, Duration::from_secs(5)).ok()?;

    session.send_line("jobs -p").ok()?;
    let captures = wait_for_sentinel(session, Duration::from_secs(5)).ok()?;
    Some(parse_jobs_p(&String::from_utf8_lossy(captures.before())))
}

/// Send SIGTERM then SIGKILL to foreground children of `shell_pid`,
/// skipping known background PIDs.
///
/// Uses `sysinfo` to enumerate and signal child processes, avoiding both
/// the fragility of spawning an external `pgrep` and raw `libc::kill` calls.
///
/// Returns `true` if all foreground children are confirmed dead after
/// signalling, `false` if any survived SIGKILL (the shell is unrecoverable).
pub(crate) fn kill_foreground(shell_pid: u32, background_pids: &HashSet<u32>) -> bool {
    use sysinfo::Signal;

    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, false);

    let parent = SysPid::from_u32(shell_pid);
    let foreground_pids: Vec<SysPid> = sys
        .processes()
        .values()
        .filter(|p| p.parent() == Some(parent))
        .map(|p| p.pid())
        // Skip known background jobs.
        .filter(|pid| !background_pids.contains(&pid.as_u32()))
        // PID 0 is a valid u32 but signalling it would affect the entire
        // process group, which would kill the harness and its parent.
        .filter(|&pid| pid.as_u32() != 0)
        .collect();

    if foreground_pids.is_empty() {
        return true;
    }

    // SIGTERM first.
    for &pid in &foreground_pids {
        if let Some(process) = sys.process(pid) {
            let _ = process.kill_with(Signal::Term);
        }
    }

    // Give processes a moment to exit gracefully, then refresh to check survivors.
    std::thread::sleep(Duration::from_millis(500));
    sys.refresh_processes(ProcessesToUpdate::Some(&foreground_pids), true);

    // SIGKILL any survivors.
    for &pid in &foreground_pids {
        if let Some(process) = sys.process(pid) {
            process.kill();
        }
    }

    // Final check: verify all foreground processes are dead.
    std::thread::sleep(Duration::from_millis(100));
    sys.refresh_processes(ProcessesToUpdate::Some(&foreground_pids), true);
    foreground_pids
        .iter()
        .all(|pid| sys.process(*pid).is_none())
}

/// Kill the shell process itself via SIGKILL.
///
/// Used when `kill_foreground` fails — the shell is in an unrecoverable
/// state and must be replaced.
pub(crate) fn kill_shell(shell_pid: u32) {
    let mut sys = System::new();
    let pid = SysPid::from_u32(shell_pid);
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
    if let Some(process) = sys.process(pid) {
        process.kill();
    }
}

/// Drain the PTY after `kill_foreground` to return bash to a clean prompt.
///
/// Returns `(exit_code, cwd)` from the post-kill sentinel, or `None` on timeout.
pub(crate) fn drain_after_kill(session: &mut OsSession) -> Option<(i32, String)> {
    wait_for_sentinel(session, DRAIN_TIMEOUT)
        .ok()
        .map(|c| parse_sentinel(&c))
}

/// Drain the internal buffer of an expectrl session (non-blocking).
///
/// After `expect()` times out or hits EOF, data it read is sitting in
/// expectrl's internal buffer (via `keep_in_buffer`). This helper extracts
/// it without blocking.
///
/// We use `check()` (single non-blocking read + pattern match) instead of
/// `fill_buf()`. The latter delegates to `BufReader::fill_buf()` which
/// does a blocking `read()` when the BufReader layer is empty — even if
/// expectrl's own buffer has data. With PS1='', a command that produces
/// no output (e.g. `sleep`) leaves both the OS fd and BufReader empty,
/// so `fill_buf()` blocks forever. `check()` correctly consults both
/// the OS fd (via `read_available`) and expectrl's internal buffer (via
/// `get_available`), returning immediately in either case.
pub(crate) fn drain_buf(session: &mut OsSession) -> Result<Vec<u8>, crate::TypeError> {
    use expectrl::Expect;
    // check() does: read_available() (non-blocking OS read into internal
    // buffer) + get_available() (inspect internal buffer) + needle.check().
    // The (?s).+ pattern matches any non-empty content across newlines.
    match session.check(expectrl::Regex(r"(?s).+")) {
        Ok(captures) => {
            let mut data = captures.before().to_vec();
            if let Some(matched) = captures.get(0) {
                data.extend_from_slice(matched);
            }
            Ok(data)
        }
        // No data available — return empty without blocking.
        Err(_) => Ok(vec![]),
    }
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
    // by wait_for_sentinel, testing the full parsing path end-to-end.

    #[tokio::test]
    async fn parse_sentinel_exit_code_zero_for_true() {
        let pty = PtySession::spawn().await.unwrap();
        let (exit_code, _) = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            pty.session.send_line("true").unwrap();
            let caps = wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
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
            pty.session.send_line("false").unwrap();
            let caps = wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
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
            pty.session.send_line("sh -c 'exit 99'").unwrap();
            let caps = wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
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
            pty.session.send_line("cd /").unwrap();
            let caps = wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            parse_sentinel(&caps)
        })
        .await
        .unwrap();
        assert_eq!(cwd, "/", "sentinel CWD must reflect the cd target");
    }

    #[tokio::test]
    async fn parse_sentinel_overflow_exit_code_returns_minus_one() {
        // When the sentinel's exit-code field overflows i32 (e.g. "9999999999"),
        // parse::<i32>() returns Err.  parse_sentinel must NOT fall back to 0
        // (indistinguishable from a successful command); it must return -1,
        // a value that bash cannot produce (exit codes are always 0–255).
        //
        // This test was previously a known-failing bug-documentation test.
        // The fix — changing .unwrap_or(0) to .unwrap_or(-1) — has been applied.
        let pty = PtySession::spawn().await.unwrap();
        let exit_code = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            // Override PROMPT_COMMAND to emit an exit-code field that overflows i32.
            pty.session
                .send_line("PROMPT_COMMAND='printf \"\\033]999;9999999999;$PWD\\007\"'")
                .unwrap();
            // Consume the sentinel produced by this PROMPT_COMMAND change itself.
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            // Run any command; the sentinel will carry "9999999999" as the exit code.
            pty.session.send_line("false").unwrap();
            let caps = wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            let (exit_code, _) = parse_sentinel(&caps);
            exit_code
        })
        .await
        .unwrap();
        assert_eq!(
            exit_code, -1,
            "a sentinel with an overflowing exit-code field must produce -1, \
             not 0 (which is indistinguishable from genuine success)"
        );
    }

    // ── wait_for_sentinel: timeout behaviour ──────────────────────────────

    #[tokio::test]
    async fn wait_for_sentinel_times_out_when_no_new_sentinel_emitted() {
        // After spawn the initial sentinel is consumed. With no command sent,
        // no new sentinel arrives and wait_for_sentinel must return ExpectTimeout.
        let pty = PtySession::spawn().await.unwrap();
        let result = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            // No command sent — no sentinel will arrive.
            wait_for_sentinel(&mut pty.session, Duration::from_millis(200))
        })
        .await
        .unwrap();
        assert!(
            matches!(result, Err(expectrl::Error::ExpectTimeout)),
            "wait_for_sentinel must return ExpectTimeout when no sentinel arrives; \
             got: {:?}",
            result
        );
    }

    #[ignore = "expectrl's expect() overshoots short timeouts by ~670ms"]
    #[tokio::test]
    async fn wait_for_sentinel_does_not_dramatically_overshoot_timeout() {
        // Guards against an implementation that loops with a fixed sleep much
        // larger than POLL_INTERVAL, making the effective timeout 10× the request.
        let pty = PtySession::spawn().await.unwrap();
        let start = std::time::Instant::now();
        let _result = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            wait_for_sentinel(&mut pty.session, Duration::from_millis(150))
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
        let pids = pids.expect("run_jobs_p_sync must succeed on a healthy session");
        assert!(
            pids.is_empty(),
            "freshly spawned shell must report no background jobs"
        );
    }

    #[tokio::test]
    async fn run_jobs_p_sync_returns_pid_of_running_background_job() {
        let pty = PtySession::spawn().await.unwrap();
        let pids = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            // Start a long-running background job.
            pty.session.send_line("sleep 9999 &").unwrap();
            // Shell returns to prompt after backgrounding; wait for sentinel.
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            run_jobs_p_sync(&mut pty.session)
        })
        .await
        .unwrap();
        let pids = pids.expect("run_jobs_p_sync must succeed");
        assert_eq!(pids.len(), 1, "exactly one background job must be tracked");
    }

    #[tokio::test]
    async fn run_jobs_p_sync_job_absent_after_it_completes() {
        // A short-lived background job must disappear from `jobs -p` once done.
        //
        // With `set +m` (job monitoring off), bash only reaps completed
        // background jobs during a full prompt cycle — when it displays PS1 and
        // evaluates PROMPT_COMMAND.  Sleeping and then immediately running
        // `jobs -p` is NOT sufficient: bash has received SIGCHLD but hasn't
        // cleaned up the job table yet.  We must force a prompt cycle first by
        // sending a no-op command (`true`) and waiting for its sentinel.
        //
        // Note: run_jobs_p_sync itself now runs `true` first, plus the explicit
        // one below — two reap cycles, ensuring robustness.
        let pty = PtySession::spawn().await.unwrap();
        let pids = tokio::task::spawn_blocking(move || {
            let mut pty = pty;
            use expectrl::Expect;
            pty.session.send_line("sleep 0.1 &").unwrap();
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            // Wait for the background job to finish, then force a full prompt
            // cycle so bash reaps it and purges it from the job table.
            std::thread::sleep(Duration::from_millis(300));
            pty.session.send_line("true").unwrap();
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            run_jobs_p_sync(&mut pty.session)
        })
        .await
        .unwrap();
        let pids = pids.expect("run_jobs_p_sync must succeed");
        assert!(
            pids.is_empty(),
            "completed background job must not appear in jobs -p"
        );
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
            pty.session.send_line("sleep 9999").unwrap();
            // Read from the PTY so bash can make progress forking the child.
            // sleep produces no sentinel, so this will time out — that's expected.
            let _ = wait_for_sentinel(&mut pty.session, Duration::from_secs(1));

            // Kill foreground processes (no background PIDs to spare).
            kill_foreground(shell_pid, &HashSet::new());

            // Bash regains control, runs PROMPT_COMMAND, emits a sentinel.
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5))
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
            pty.session.send_line("sleep 9999 &").unwrap();
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();
            let bg_pids = run_jobs_p_sync(&mut pty.session).expect("run_jobs_p_sync must succeed");
            assert_eq!(bg_pids.len(), 1, "expected exactly one background job");
            let bg_pid = *bg_pids.iter().next().unwrap();

            // Start a foreground process.
            pty.session.send_line("sleep 9999").unwrap();
            // Read from the PTY so bash can make progress forking the child.
            let _ = wait_for_sentinel(&mut pty.session, Duration::from_secs(1));

            // Kill foreground only — background PID is in the protection set.
            kill_foreground(shell_pid, &bg_pids);
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();

            // The background job must still be running.
            let remaining =
                run_jobs_p_sync(&mut pty.session).expect("run_jobs_p_sync must succeed");
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
            pty.session.send_line("cd /").unwrap();
            wait_for_sentinel(&mut pty.session, Duration::from_secs(5)).unwrap();

            // Start a foreground process and read from the PTY so bash can
            // make progress forking the child.
            pty.session.send_line("sleep 9999").unwrap();
            let _ = wait_for_sentinel(&mut pty.session, Duration::from_secs(1));
            kill_foreground(shell_pid, &HashSet::new());

            drain_after_kill(&mut pty.session)
        })
        .await
        .unwrap();
        let (exit_code, working_directory) =
            cwd.expect("drain_after_kill must return Some after a successful kill");
        assert_eq!(
            working_directory, "/",
            "drained CWD must match the directory set before the kill"
        );
        // SIGTERM → 128+15=143, SIGKILL → 128+9=137
        assert!(
            exit_code == 143 || exit_code == 137,
            "killed process exit code must be 143 (SIGTERM) or 137 (SIGKILL); got: {exit_code}"
        );
    }
}
