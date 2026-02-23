use std::io::{self, BufRead, Write};

use anyhow::Result;
use rig::{
    agent::{PromptHook, ToolCallHookAction},
    client::ProviderClient,
    completion::{CompletionModel, Prompt},
    prelude::*,
    providers::openai,
};
use tekton_type_tool::TypeTool;
use tracing::Level;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Clone)]
struct LogToolCalls;

impl<M: CompletionModel> PromptHook<M> for LogToolCalls {
    fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> impl Future<Output = ToolCallHookAction> + Send {
        eprintln!("[tool call] {tool_name}({args})");
        async { ToolCallHookAction::cont() }
    }
}

const DEFAULT_MODEL: &str = "gpt-4o";
const DEFAULT_MAX_TURNS: usize = 20;

#[cfg(feature = "otel")]
fn init_tracing() -> Option<opentelemetry_sdk::trace::SdkTracerProvider> {
    use opentelemetry::trace::TracerProvider;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
        .build()
        .expect("failed to create OTLP exporter");

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder()
                .with_service_name("tekton-agent")
                .build(),
        )
        .build();

    let tracer = provider.tracer("tekton-agent");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let filter = tracing_subscriber::filter::EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().pretty())
        .with(otel_layer)
        .init();

    Some(provider)
}

#[cfg(not(feature = "otel"))]
fn init_tracing() -> Option<()> {
    let filter = tracing_subscriber::filter::EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().pretty())
        .init();

    None
}

#[tokio::main]
async fn main() -> Result<()> {
    let _otel_provider = init_tracing();

    let model_name = std::env::var("TEKTON_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    // Use the Completions API (Chat Completions) for maximum compatibility
    // with local model servers (ollama, vLLM, LM Studio, etc.).
    // Set OPENAI_BASE_URL to point at a local server.
    let client = openai::CompletionsClient::from_env();

    let tool = TypeTool::new()
        .with_name("agent")
        .with_job_callback(|notification| {
            eprintln!(
                "[job done] pid={} exit={:?} cmd={}",
                notification.pid, notification.exit_code, notification.command
            );
        })
        .spawn()
        .await?;

    let watcher = tool.spawn_watcher();

    let agent = client
        .agent(&model_name)
        .preamble("\
You're a helpful assistant. You can chat with the user and you have access to \
a terminal from which you can execute linux commands on bash.
The tool `type` gives you access to this terminal, by sending keystrokes.

The terminal is interactive, so remember to press Enter (represented by `\\n` at the end of keys) to execute commands.

Examples:
- Run a command (including the Enter key \\n)
  `{\"keys\": \"ls -la\\n\"}`
- Send Ctrl-C (no \\n):
  `{\"keys\": \"\\u0003\"}`

Incorrect: `{\"keys\": \"ls -la\"}` as it's missing the Enter key `\\n`

",
        )
        .max_tokens(4096)
        .default_max_turns(DEFAULT_MAX_TURNS)
        .hook(LogToolCalls)
        .tool(tool)
        .build();

    eprintln!("Tekton agent ready (model: {model_name})");
    eprintln!("Type a prompt and press Enter. Ctrl-D to quit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut chat_history: Vec<rig::message::Message> = Vec::new();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }

        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        match agent.prompt(prompt).with_history(&mut chat_history).await {
            Ok(response) => {
                println!("\n{response}\n");
            }
            Err(err) => {
                eprintln!("\n[error] {err}\n");
            }
        }
    }

    watcher.abort();

    #[cfg(feature = "otel")]
    if let Some(provider) = _otel_provider {
        let _ = provider.shutdown();
    }

    Ok(())
}
