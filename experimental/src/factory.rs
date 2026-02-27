use rig::agent::{Agent, AgentBuilder, WithBuilderTools};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::{anthropic, gemini, openai};
use tekton_terminal_tool::TerminalTool;

use crate::config::{AgentConfig, Provider};
use crate::error::FactoryError;
use crate::handle::AgentHandle;

fn configure_agent<M: CompletionModel>(
    builder: AgentBuilder<M, (), WithBuilderTools>,
    config: &AgentConfig,
) -> Agent<M> {
    let mut builder = builder
        .preamble(&config.system_prompt)
        .default_max_turns(config.max_turns);
    if let Some(t) = config.temperature {
        builder = builder.temperature(t);
    }
    if let Some(m) = config.max_tokens {
        builder = builder.max_tokens(m);
    }
    builder.build()
}

/// Build an [`AgentHandle`] from the given configuration.
///
/// Spawns a [`TerminalTool`] PTY session and constructs a rig `Agent` for the
/// configured provider and model.
///
/// # Errors
///
/// Returns [`FactoryError::TerminalSpawn`] if the PTY session fails to start.
pub async fn build_agent(config: &AgentConfig) -> Result<AgentHandle, FactoryError> {
    let tool = TerminalTool::new()
        .with_name(&config.name)
        .spawn()
        .await?;

    let model_name = &config.model.name;

    let handle = match config.model.provider {
        Provider::Anthropic => {
            let client = anthropic::Client::from_env();
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::Anthropic(configure_agent(builder, config))
        }
        Provider::OpenAI => {
            let client = openai::Client::from_env();
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::OpenAI(configure_agent(builder, config))
        }
        Provider::OpenAICompatible => {
            let client = openai::CompletionsClient::from_env();
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::OpenAICompatible(configure_agent(builder, config))
        }
        Provider::Gemini => {
            let client = gemini::Client::from_env();
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::Gemini(configure_agent(builder, config))
        }
    };

    Ok(handle)
}
