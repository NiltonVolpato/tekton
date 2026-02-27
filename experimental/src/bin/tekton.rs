use std::io::{self, BufRead, Write};

use futures::StreamExt;
use tekton_experimental::{build_agent, load_config, StreamEvent};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: tekton <config.pkl>");
        std::process::exit(1);
    }

    let config = load_config(&args[1]).unwrap_or_else(|e| {
        eprintln!("Failed to load config: {e}");
        std::process::exit(1);
    });

    eprintln!("Building agent '{}'...", config.name);
    let agent = build_agent(&config).await.unwrap_or_else(|e| {
        eprintln!("Failed to build agent: {e}");
        std::process::exit(1);
    });
    eprintln!("Ready. Type a message (Ctrl-D to quit).\n");

    let stdin = io::stdin();
    let mut history = Vec::new();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF (Ctrl-D)
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut stream = agent.stream_chat(line, history.clone()).await;
        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::Text(text)) => {
                    print!("{text}");
                    io::stdout().flush().unwrap();
                }
                Ok(StreamEvent::ToolCall { name, args }) => {
                    eprintln!("\n[tool: {name}({args})]");
                }
                Err(e) => {
                    eprintln!("\nStream error: {e}");
                    break;
                }
            }
        }
        println!();

        // Add user message to history for next turn
        history.push(rig::message::Message::user(line));
    }

    eprintln!("\nGoodbye.");
}
