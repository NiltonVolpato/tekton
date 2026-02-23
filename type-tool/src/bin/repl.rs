use std::io::{self, BufRead, Write};

use rig::tool::Tool;
use tekton_type_tool::{Outcome, TypeArgs, TypeTool};

/// Default timeout in seconds for non-interactive commands.
const DEFAULT_TIMEOUT: f64 = 30.0;

#[tokio::main]
async fn main() {
    let tool = TypeTool::new()
        .spawn()
        .await
        .expect("failed to spawn PTY session");
    let cwd = tool.working_directory().await;
    eprintln!("[cwd: {cwd}]");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut interactive = false;
    let mut timeout = DEFAULT_TIMEOUT;

    loop {
        let mode = if interactive { "i" } else { "n" };
        print!("[{mode} {timeout:.0}s] > ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }

        // Handle REPL meta-commands (prefixed with /).
        let trimmed = line.trim();
        if trimmed.starts_with('/') {
            match trimmed {
                "/i" | "/interactive" => {
                    interactive = true;
                    eprintln!("[mode: interactive]");
                }
                "/n" | "/non-interactive" => {
                    interactive = false;
                    eprintln!("[mode: non-interactive]");
                }
                s if s.starts_with("/timeout ") || s.starts_with("/t ") => {
                    let val = s.split_whitespace().nth(1).unwrap_or("");
                    match val.parse::<f64>() {
                        Ok(t) if t > 0.0 => {
                            timeout = t;
                            eprintln!("[timeout: {timeout:.1}s]");
                        }
                        _ => eprintln!("[error: invalid timeout '{val}']"),
                    }
                }
                "/help" | "/h" => {
                    eprintln!("REPL commands:");
                    eprintln!("  /i, /interactive       Switch to interactive mode");
                    eprintln!("  /n, /non-interactive   Switch to non-interactive mode");
                    eprintln!("  /t <secs>, /timeout    Set timeout in seconds");
                    eprintln!("  /h, /help              Show this help");
                }
                _ => eprintln!("[error: unknown command '{trimmed}'; try /help]"),
            }
            continue;
        }

        // Strip only the trailing newline from read_line, preserving all other
        // whitespace (leading spaces, trailing spaces, tabs, etc.).
        let keys = if line.ends_with('\n') {
            line.clone()
        } else {
            // EOF without trailing newline — append one so it acts like Enter.
            format!("{line}\n")
        };

        let args = TypeArgs {
            keys,
            interactive,
            timeout: Some(timeout),
        };

        match tool.call(args).await {
            Ok(out) => {
                if !out.output.is_empty() {
                    print!("{}", out.output);
                }
                match &out.outcome {
                    Outcome::Completed {
                        exit_code,
                        working_directory,
                    } => {
                        eprintln!("[cwd: {working_directory}]");
                        if *exit_code != 0 {
                            eprintln!("[exit: {exit_code}]");
                        }
                    }
                    Outcome::Killed {
                        exit_code,
                        working_directory,
                        timeout_secs,
                    } => {
                        eprintln!(
                            "[timeout: Command timed out after {timeout_secs:.1}s and was terminated.]"
                        );
                        eprintln!("[cwd: {working_directory}]");
                        eprintln!("[exit: {exit_code}]");
                    }
                    Outcome::Waiting => {
                        eprintln!("[waiting]");
                    }
                    Outcome::ShellExited { working_directory } => {
                        eprintln!("[shell exited, respawned]");
                        eprintln!("[cwd: {working_directory}]");
                    }
                }
            }
            Err(e) => {
                eprintln!("[error: {e}]");
            }
        }
    }
}
