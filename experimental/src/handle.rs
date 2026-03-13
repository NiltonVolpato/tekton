use std::pin::Pin;

use futures::Stream;
use rig::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig::completion::PromptError;
use rig::message::Message;
use rig::providers::{anthropic, gemini, openai};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent};

pub enum AgentHandle {
    Anthropic(Agent<anthropic::completion::CompletionModel>),
    OpenAI(Agent<openai::responses_api::ResponsesCompletionModel>),
    OpenAICompatible(Agent<openai::completion::CompletionModel>),
    Gemini(Agent<gemini::completion::CompletionModel>),
}

#[derive(Debug)]
pub enum StreamEvent {
    Text(String),
    ToolCall {
        name: String,
        args: serde_json::Value,
    },
    ToolResult {
        id: String,
        content: String,
    },
}

pub type TextStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, StreamingError>> + Send>>;

pub(crate) fn map_chunk<R>(
    item: Result<MultiTurnStreamItem<R>, StreamingError>,
) -> Option<Result<StreamEvent, StreamingError>> {
    match item {
        Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
            Some(Ok(StreamEvent::Text(text.text)))
        }
        Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
            tool_call,
            internal_call_id: _,
        })) => Some(Ok(StreamEvent::ToolCall {
            name: tool_call.function.name,
            args: tool_call.function.arguments,
        })),
        Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
            tool_result,
            internal_call_id: _,
        })) => {
            use rig::completion::message::ToolResultContent;
            let content = tool_result
                .content
                .iter()
                .filter_map(|c| match c {
                    ToolResultContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            Some(Ok(StreamEvent::ToolResult {
                id: tool_result.id,
                content,
            }))
        }
        Err(e) => Some(Err(e)),
        // TODO: do not silently skip other items (final response, deltas, reasoning)
        _ => None,
    }
}

impl AgentHandle {
    /// Send a single prompt and return the full response.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] if the provider request fails.
    pub async fn prompt(&self, prompt: &str) -> Result<String, PromptError> {
        use rig::completion::Prompt;
        match self {
            Self::Anthropic(a) => a.prompt(prompt).await,
            Self::OpenAI(a) => a.prompt(prompt).await,
            Self::OpenAICompatible(a) => a.prompt(prompt).await,
            Self::Gemini(a) => a.prompt(prompt).await,
        }
    }

    /// Send a prompt with conversation history and return the full response.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] if the provider request fails.
    pub async fn chat(&self, prompt: &str, history: Vec<Message>) -> Result<String, PromptError> {
        use rig::completion::Chat;
        match self {
            Self::Anthropic(a) => a.chat(prompt, history).await,
            Self::OpenAI(a) => a.chat(prompt, history).await,
            Self::OpenAICompatible(a) => a.chat(prompt, history).await,
            Self::Gemini(a) => a.chat(prompt, history).await,
        }
    }

    pub async fn stream_chat(&self, prompt: &str, history: Vec<Message>) -> TextStream {
        use futures::StreamExt;
        use rig::streaming::StreamingChat;

        match self {
            Self::Anthropic(a) => {
                let s = a.stream_chat(prompt, history).await;
                Box::pin(s.filter_map(|item| async { map_chunk(item) }))
            }
            Self::OpenAI(a) => {
                let s = a.stream_chat(prompt, history).await;
                Box::pin(s.filter_map(|item| async { map_chunk(item) }))
            }
            Self::OpenAICompatible(a) => {
                let s = a.stream_chat(prompt, history).await;
                Box::pin(s.filter_map(|item| async { map_chunk(item) }))
            }
            Self::Gemini(a) => {
                let s = a.stream_chat(prompt, history).await;
                Box::pin(s.filter_map(|item| async { map_chunk(item) }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::completion::message::{Text, ToolCall, ToolFunction};
    use rig::streaming::StreamedUserContent;

    #[test]
    fn map_chunk_text_event() {
        let item: Result<MultiTurnStreamItem<()>, StreamingError> = Ok(
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(Text {
                text: "hello".to_string(),
            })),
        );
        let result = map_chunk(item);
        let event = result.expect("should be Some").expect("should be Ok");
        match event {
            StreamEvent::Text(t) => assert_eq!(t, "hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn map_chunk_tool_call_event() {
        let args = serde_json::json!({"key": "value"});
        let item: Result<MultiTurnStreamItem<()>, StreamingError> = Ok(
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call: ToolCall::new(
                    "call-1".to_string(),
                    ToolFunction::new("my_tool".to_string(), args.clone()),
                ),
                internal_call_id: "internal-1".to_string(),
            }),
        );
        let result = map_chunk(item);
        let event = result.expect("should be Some").expect("should be Ok");
        match event {
            StreamEvent::ToolCall { name, args: a } => {
                assert_eq!(name, "my_tool");
                assert_eq!(a, args);
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn map_chunk_tool_result_event() {
        use rig::completion::message::{ToolResult, ToolResultContent};
        use rig::one_or_many::OneOrMany;

        let item: Result<MultiTurnStreamItem<()>, StreamingError> = Ok(
            MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result: ToolResult {
                    id: "id".to_string(),
                    call_id: None,
                    content: OneOrMany::one(ToolResultContent::Text(Text {
                        text: "result".to_string(),
                    })),
                },
                internal_call_id: "internal".to_string(),
            }),
        );
        let result = map_chunk(item);
        let event = result.expect("should be Some").expect("should be Ok");
        match event {
            StreamEvent::ToolResult { id, content } => {
                assert_eq!(id, "id");
                assert_eq!(content, "result");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn map_chunk_filters_final_response() {
        use rig::agent::FinalResponse;

        let final_response: FinalResponse = serde_json::from_value(serde_json::json!({
            "response": "done",
            "aggregatedUsage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0
            }
        }))
        .expect("should deserialize FinalResponse");

        let item: Result<MultiTurnStreamItem<()>, StreamingError> =
            Ok(MultiTurnStreamItem::FinalResponse(final_response));
        assert!(map_chunk(item).is_none());
    }

    #[test]
    fn map_chunk_filters_tool_call_delta() {
        use rig::streaming::ToolCallDeltaContent;

        let item: Result<MultiTurnStreamItem<()>, StreamingError> = Ok(
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCallDelta {
                id: "id".to_string(),
                internal_call_id: "internal".to_string(),
                content: ToolCallDeltaContent::Delta("chunk".to_string()),
            }),
        );
        assert!(map_chunk(item).is_none());
    }

    #[test]
    fn map_chunk_filters_reasoning() {
        use rig::completion::message::Reasoning;

        let reasoning: Reasoning = serde_json::from_value(serde_json::json!({"content": []}))
            .expect("should deserialize Reasoning");
        let item: Result<MultiTurnStreamItem<()>, StreamingError> =
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::Reasoning(reasoning),
            ));
        assert!(map_chunk(item).is_none());
    }

    #[test]
    fn map_chunk_filters_reasoning_delta() {
        let item: Result<MultiTurnStreamItem<()>, StreamingError> = Ok(
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ReasoningDelta {
                id: None,
                reasoning: "thinking...".to_string(),
            }),
        );
        assert!(map_chunk(item).is_none());
    }

    #[test]
    fn map_chunk_passes_through_errors() {
        use rig::completion::CompletionError;

        // Create a CompletionError via JsonError variant
        let json_err: serde_json::Error = serde_json::from_str::<()>("invalid").unwrap_err();
        let completion_err = CompletionError::from(json_err);
        let streaming_err = StreamingError::from(completion_err);

        let item: Result<MultiTurnStreamItem<()>, StreamingError> = Err(streaming_err);
        let result = map_chunk(item);
        let err = result.expect("should be Some").unwrap_err();
        assert!(
            matches!(
                err,
                StreamingError::Completion(CompletionError::JsonError(_))
            ),
            "expected StreamingError::Completion(CompletionError::JsonError(_)), got: {err:?}"
        );
    }
}
