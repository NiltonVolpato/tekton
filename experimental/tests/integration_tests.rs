mod common;

use tekton_experimental::{build_agent, load_config};

use common::{global_dir, workspace_pkl};

// These tests require the mock server on localhost:8100 (started by `just test`).
// They intentionally fail if the server is not running — do not skip or ignore them.

const MOCK_TEXT_RESPONSE: &str =
    "This is a mock response from the generic engine! Model: openai";

const MOCK_TOOL_CALL_TEXT_RESPONSE: &str = "The result of the computation is 2.";

#[tokio::test]
async fn integration_prompt() {
    let config = load_config(workspace_pkl("mock-server"), global_dir()).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();
    let response = agent.prompt("Hello").await.unwrap();
    assert_eq!(response, MOCK_TEXT_RESPONSE);
}

#[tokio::test]
async fn integration_chat() {
    let config = load_config(workspace_pkl("mock-server"), global_dir()).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();
    let response = agent.chat("Hello", vec![]).await.unwrap();
    assert_eq!(response, MOCK_TEXT_RESPONSE);
}

#[tokio::test]
async fn integration_stream_chat() {
    let config = load_config(workspace_pkl("mock-server"), global_dir()).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();

    use futures::StreamExt;
    use tekton_experimental::StreamEvent;

    let mut stream = agent.stream_chat("Hello", vec![]).await;
    let mut full_response = String::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::Text(t)) => full_response.push_str(&t),
            Ok(StreamEvent::ToolCall { .. } | StreamEvent::ToolResult { .. }) => {
                panic!("unexpected tool event in text-only stream")
            }
            Err(e) => panic!("stream returned an error: {e}"),
        }
    }
    // Streaming splits into words with trailing spaces; trim the final one.
    assert_eq!(full_response.trim_end(), MOCK_TEXT_RESPONSE);
}

#[tokio::test]
async fn integration_tool_call_stream() {
    let config = load_config(workspace_pkl("mock-tool-call"), global_dir()).unwrap();
    let agent = build_agent(&config, "tool-call-agent").await.unwrap();

    use futures::StreamExt;
    use tekton_experimental::StreamEvent;

    let mut stream = agent.stream_chat("Hello", vec![]).await;

    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();
    let mut full_response = String::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::Text(t)) => full_response.push_str(&t),
            Ok(StreamEvent::ToolCall { name, args }) => {
                tool_calls.push((name, args));
            }
            Ok(StreamEvent::ToolResult { id, content }) => {
                tool_results.push((id, content));
            }
            Err(e) => panic!("stream returned an error: {e}"),
        }
    }

    // The mock returns one tool call to the terminal tool.
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].0, "terminal");
    assert_eq!(
        tool_calls[0].1,
        serde_json::json!({"command": "echo $(( 1 + 1 ))"})
    );

    // The terminal tool should produce one result.
    assert_eq!(tool_results.len(), 1);

    // After the tool result, the mock returns a text response.
    assert_eq!(full_response.trim_end(), MOCK_TOOL_CALL_TEXT_RESPONSE);
}
