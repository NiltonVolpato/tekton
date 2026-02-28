use std::pin::Pin;

use futures::Stream;
use rig::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig::completion::PromptError;
use rig::message::Message;
use rig::providers::{anthropic, gemini, openai};
use rig::streaming::StreamedAssistantContent;

pub enum AgentHandle {
    Anthropic(Agent<anthropic::completion::CompletionModel>),
    OpenAI(Agent<openai::responses_api::ResponsesCompletionModel>),
    OpenAICompatible(Agent<openai::completion::CompletionModel>),
    Gemini(Agent<gemini::completion::CompletionModel>),
}

pub enum StreamEvent {
    Text(String),
    ToolCall { name: String, args: serde_json::Value },
}

pub type TextStream =
    Pin<Box<dyn Stream<Item = Result<StreamEvent, StreamingError>> + Send>>;

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
    pub async fn chat(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> Result<String, PromptError> {
        use rig::completion::Chat;
        match self {
            Self::Anthropic(a) => a.chat(prompt, history).await,
            Self::OpenAI(a) => a.chat(prompt, history).await,
            Self::OpenAICompatible(a) => a.chat(prompt, history).await,
            Self::Gemini(a) => a.chat(prompt, history).await,
        }
    }

    pub async fn stream_chat(
        &self,
        prompt: &str,
        history: Vec<Message>,
    ) -> TextStream {
        use futures::StreamExt;
        use rig::streaming::StreamingChat;

        fn map_chunk<R>(
            item: Result<MultiTurnStreamItem<R>, StreamingError>,
        ) -> Option<Result<StreamEvent, StreamingError>> {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::Text(text),
                )) => Some(Ok(StreamEvent::Text(text.text))),
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall {
                        tool_call,
                        internal_call_id: _,
                    },
                )) => Some(Ok(StreamEvent::ToolCall {
                    name: tool_call.function.name,
                    args: tool_call.function.arguments,
                })),
                Err(e) => Some(Err(e)),
                // Skip other items (user items, final response, deltas, reasoning)
                _ => None,
            }
        }

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
