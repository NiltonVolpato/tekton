mod common;

use std::path::Path;

use tekton_experimental::{build_agent, load_config};

use common::{write_base_schema, write_pkl};

fn mock_agent_config(dir: &Path) -> std::path::PathBuf {
    write_base_schema(dir);
    write_pkl(
        dir,
        "test.pkl",
        r#"
amends "AgentConfig.pkl"
name = "mock-agent"
model {
  provider = "OpenAICompatible"
  name = "openai"
}
system_prompt = "You are a test agent."
"#,
    )
}

#[tokio::test]
async fn integration_prompt() {
    // Guard: ensure mock server env is configured (the agent reads OPENAI_BASE_URL directly).
    let _ = common::test_server_url();
    let dir = tempfile::tempdir().unwrap();
    let pkl = mock_agent_config(dir.path());
    let config = load_config(&pkl).unwrap();
    let agent = build_agent(&config).await.unwrap();
    let response = agent.prompt("Hello").await.unwrap();
    assert!(!response.is_empty());
}

#[tokio::test]
async fn integration_chat() {
    // Guard: ensure mock server env is configured (the agent reads OPENAI_BASE_URL directly).
    let _ = common::test_server_url();
    let dir = tempfile::tempdir().unwrap();
    let pkl = mock_agent_config(dir.path());
    let config = load_config(&pkl).unwrap();
    let agent = build_agent(&config).await.unwrap();
    let response = agent.chat("Hello", vec![]).await.unwrap();
    assert!(!response.is_empty());
}

#[tokio::test]
async fn integration_stream_chat() {
    // Guard: ensure mock server env is configured (the agent reads OPENAI_BASE_URL directly).
    let _ = common::test_server_url();
    let dir = tempfile::tempdir().unwrap();
    let pkl = mock_agent_config(dir.path());
    let config = load_config(&pkl).unwrap();
    let agent = build_agent(&config).await.unwrap();

    use futures::StreamExt;
    use tekton_experimental::StreamEvent;

    let mut stream = agent.stream_chat("Hello", vec![]).await;
    let mut texts = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::Text(t)) => texts.push(t),
            Ok(StreamEvent::ToolCall { .. }) => {}
            Err(e) => panic!("stream returned an error: {e}"),
        }
    }
    assert!(!texts.is_empty());
}
