use std::io::{self, BufRead, Write};

use rig::tool::Tool;
use tekton_type_tool::{Outcome, TypeArgs, TypeTool};

#[tokio::main]
async fn main() {
    let (tool, cwd) = TypeTool::spawn().await.expect("failed to spawn PTY session");
    eprintln!("[cwd: {cwd}]");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Append newline so it acts like pressing Enter.
        let keys = format!("{trimmed}\n");

        let args = TypeArgs {
            keys,
            interactive: false,
            timeout: Some(10.0),
        };

        match tool.call(args).await {
            Ok(out) => {
                if !out.output.is_empty() {
                    print!("{}", out.output);
                }
                match &out.outcome {
                    Outcome::Completed { exit_code, working_directory } => {
                        eprintln!("[cwd: {working_directory}]");
                        if *exit_code != 0 {
                            eprintln!("[exit: {exit_code}]");
                        }
                    }
                    Outcome::Killed { working_directory, timeout_secs } => {
                        eprintln!("[timeout: Command timed out after {timeout_secs:.1}s and was terminated.]");
                        eprintln!("[cwd: {working_directory}]");
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
