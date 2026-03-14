mod common;

use tekton_unstable::{build_agent, load_config};

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
    use tekton_unstable::StreamEvent;

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
    use tekton_unstable::StreamEvent;

    let events: Vec<StreamEvent> = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        async {
            agent
                .stream_chat("Hello", vec![])
                .await
                .map(|e| e.expect("stream returned an error"))
                .collect::<Vec<_>>()
                .await
        },
    )
    .await
    .expect("stream timed out after 10s");

    assert!(
        events.len() >= 3,
        "expected at least 3 events (ToolCall, ToolResult, Text), got {}",
        events.len()
    );

    // First event: tool call to terminal with known id.
    assert_eq!(
        events[0],
        StreamEvent::ToolCall {
            id: "call_1".into(),
            name: "terminal".into(),
            args: serde_json::json!({"command": "echo $(( 1 + 1 ))"}),
        }
    );

    // Second event: tool result correlated to the call.
    match &events[1] {
        StreamEvent::ToolResult { call_id, content } => {
            assert_eq!(call_id, "call_1");
            let parsed: serde_json::Value =
                serde_json::from_str(content).expect("tool result content should be valid JSON");
            assert_eq!(parsed["outcome"]["exit_code"], 0, "expected successful exit: {content}");
        }
        other => panic!("expected ToolResult as second event, got {other:?}"),
    }

    // Remaining events: text chunks forming the final response.
    let full_response: String = events[2..].iter().map(|e| match e {
        StreamEvent::Text(t) => t.as_str(),
        other => panic!("expected only Text events after ToolResult, got {other:?}"),
    }).collect();
    assert_eq!(full_response.trim_end(), MOCK_TOOL_CALL_TEXT_RESPONSE);
}
