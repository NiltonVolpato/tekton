mod common;

use tekton_experimental::{build_agent, load_config};

use common::{global_dir, workspace_pkl};

// These tests require the mock server on localhost:8100 (started by `just test`).
// They intentionally fail if the server is not running — do not skip or ignore them.

#[tokio::test]
async fn integration_prompt() {
    let config = load_config(workspace_pkl("mock-server"), global_dir()).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();
    let response = agent.prompt("Hello").await.unwrap();
    assert!(!response.is_empty());
}

#[tokio::test]
async fn integration_chat() {
    let config = load_config(workspace_pkl("mock-server"), global_dir()).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();
    let response = agent.chat("Hello", vec![]).await.unwrap();
    assert!(!response.is_empty());
}

#[tokio::test]
async fn integration_stream_chat() {
    let config = load_config(workspace_pkl("mock-server"), global_dir()).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();

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
