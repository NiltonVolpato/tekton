use std::io::{self, BufRead, Write};

use rig::tool::Tool;
use tekton_type_tool::{TypeArgs, TypeTool};

#[tokio::main]
async fn main() {
    let tool = TypeTool::spawn().await.expect("failed to spawn PTY session");

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
                if let Some(msg) = &out.timeout_message {
                    eprintln!("[timeout: {msg}]");
                }
                if let Some(cwd) = &out.working_directory {
                    eprintln!("[cwd: {cwd}]");
                }
                if let Some(code) = out.exit_code {
                    if code != 0 {
                        eprintln!("[exit: {code}]");
                    }
                }
            }
            Err(e) => {
                eprintln!("[error: {e}]");
            }
        }
    }
}
