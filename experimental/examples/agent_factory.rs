use std::io::{self, BufRead, Write};

use futures::StreamExt;
use tekton_experimental::{build_agent, load_config, StreamEvent};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: cargo run --example agent_factory -- <config.pkl> <global-dir>");
        std::process::exit(1);
    }

    let config = load_config(&args[1], &args[2]).unwrap_or_else(|e| {
        eprintln!("Failed to load config: {e}");
        std::process::exit(1);
    });

    let agent_name = &config.default_agent;
    eprintln!("Building agent '{agent_name}'...");
    let agent = build_agent(&config, agent_name).await.unwrap_or_else(|e| {
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
        let mut response = String::new();
        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::Text(text)) => {
                    print!("{text}");
                    io::stdout().flush().unwrap();
                    response.push_str(&text);
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

        history.push(rig::message::Message::user(line));
        if !response.is_empty() {
            history.push(rig::message::Message::assistant(&response));
        }
    }

    eprintln!("\nGoodbye.");
}
