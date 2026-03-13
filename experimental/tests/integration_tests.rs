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

    #[derive(Debug)]
    enum Event {
        ToolCall { id: String, name: String, args: serde_json::Value },
        ToolResult { call_id: String },
        Text(String),
    }

    let mut events = Vec::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::Text(t)) => events.push(Event::Text(t)),
            Ok(StreamEvent::ToolCall { id, name, args }) => {
                events.push(Event::ToolCall { id, name, args });
            }
            Ok(StreamEvent::ToolResult { call_id, content: _ }) => {
                events.push(Event::ToolResult { call_id });
            }
            Err(e) => panic!("stream returned an error: {e}"),
        }
    }

    // Extract events by type in order.
    let tool_call = events.iter().find_map(|e| match e {
        Event::ToolCall { id, name, args } => Some((id, name, args)),
        _ => None,
    }).expect("expected a ToolCall event");
    let tool_result = events.iter().find_map(|e| match e {
        Event::ToolResult { call_id } => Some(call_id),
        _ => None,
    }).expect("expected a ToolResult event");
    let full_response: String = events.iter().filter_map(|e| match e {
        Event::Text(t) => Some(t.as_str()),
        _ => None,
    }).collect();

    // Verify the tool call.
    assert_eq!(tool_call.1, "terminal");
    assert_eq!(*tool_call.2, serde_json::json!({"command": "echo $(( 1 + 1 ))"}));

    // The tool result's call_id must match the tool call's id.
    assert_eq!(tool_result, tool_call.0);

    // Ordering: ToolCall appears before ToolResult, which appears before Text.
    let call_pos = events.iter().position(|e| matches!(e, Event::ToolCall { .. })).unwrap();
    let result_pos = events.iter().position(|e| matches!(e, Event::ToolResult { .. })).unwrap();
    let text_pos = events.iter().position(|e| matches!(e, Event::Text(_))).unwrap();
    assert!(call_pos < result_pos, "ToolCall must come before ToolResult");
    assert!(result_pos < text_pos, "ToolResult must come before Text");

    // After the tool result, the mock returns a text response.
    assert_eq!(full_response.trim_end(), MOCK_TOOL_CALL_TEXT_RESPONSE);
}
