use rig::client::{CompletionClient, ProviderClient};
use rig::providers::{anthropic, gemini, openai};
use tekton_terminal_tool::TerminalTool;

use crate::config::{AgentConfig, Provider};
use crate::error::FactoryError;
use crate::handle::AgentHandle;

pub async fn build_agent(config: &AgentConfig) -> Result<AgentHandle, FactoryError> {
    let tool = TerminalTool::new()
        .with_name(&config.name)
        .spawn()
        .await?;

    let handle = match config.model.provider {
        Provider::Anthropic => {
            let client = anthropic::Client::from_env();
            let mut builder = client
                .agent(&config.model.name)
                .preamble(&config.system_prompt)
                .default_max_turns(config.max_turns)
                .tool(tool);
            if let Some(t) = config.temperature {
                builder = builder.temperature(t);
            }
            if let Some(m) = config.max_tokens {
                builder = builder.max_tokens(m);
            }
            AgentHandle::Anthropic(builder.build())
        }
        Provider::OpenAI => {
            let client = openai::Client::from_env();
            let mut builder = client
                .agent(&config.model.name)
                .preamble(&config.system_prompt)
                .default_max_turns(config.max_turns)
                .tool(tool);
            if let Some(t) = config.temperature {
                builder = builder.temperature(t);
            }
            if let Some(m) = config.max_tokens {
                builder = builder.max_tokens(m);
            }
            AgentHandle::OpenAI(builder.build())
        }
        Provider::OpenAICompatible => {
            let client = openai::CompletionsClient::from_env();
            let mut builder = client
                .agent(&config.model.name)
                .preamble(&config.system_prompt)
                .default_max_turns(config.max_turns)
                .tool(tool);
            if let Some(t) = config.temperature {
                builder = builder.temperature(t);
            }
            if let Some(m) = config.max_tokens {
                builder = builder.max_tokens(m);
            }
            AgentHandle::OpenAICompatible(builder.build())
        }
        Provider::Gemini => {
            let client = gemini::Client::from_env();
            let mut builder = client
                .agent(&config.model.name)
                .preamble(&config.system_prompt)
                .default_max_turns(config.max_turns)
                .tool(tool);
            if let Some(t) = config.temperature {
                builder = builder.temperature(t);
            }
            if let Some(m) = config.max_tokens {
                builder = builder.max_tokens(m);
            }
            AgentHandle::Gemini(builder.build())
        }
    };

    Ok(handle)
}
