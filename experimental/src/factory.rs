use rig::agent::{Agent, AgentBuilder, WithBuilderTools};
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::providers::{anthropic, gemini, openai};
use tekton_terminal_tool::TerminalTool;

use crate::config::{AgentConfig, ApiType, Config, Credentials, ResolvedProvider};
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

/// Resolve an API key from credentials or environment variables.
///
/// - If a `Credentials` entry exists with an `api_key`, use it (no env fallback).
/// - If no `Credentials` entry exists, try env vars from the provider's `env` list.
/// - If neither, return an error.
fn resolve_api_key(
    provider: &ResolvedProvider,
    creds: Option<&Credentials>,
) -> Result<String, FactoryError> {
    if let Some(c) = creds
        && let Some(key) = &c.api_key
    {
        return Ok(key.clone());
    }
    // No explicit credentials — try env vars from catalog
    for var in &provider.env {
        if let Ok(key) = std::env::var(var) {
            return Ok(key);
        }
    }
    Err(FactoryError::MissingApiKey {
        provider: provider.name.clone(),
        env: provider.env.join(" or "),
    })
}

/// Build an [`AgentHandle`] from the given configuration and agent name.
///
/// Looks up the agent, resolves its provider and credentials, spawns a
/// [`TerminalTool`] PTY session, and constructs a rig `Agent`.
///
/// # Errors
///
/// Returns [`FactoryError`] if the agent or provider is not found, if
/// credentials cannot be resolved, or if the PTY session fails to start.
pub async fn build_agent(
    config: &Config,
    agent_name: &str,
) -> Result<AgentHandle, FactoryError> {
    let agent_config = config
        .agents
        .get(agent_name)
        .ok_or_else(|| FactoryError::UnknownAgent(agent_name.to_string()))?;
    let provider = config
        .resolved_providers
        .get(&agent_config.model.provider)
        .ok_or_else(|| FactoryError::UnknownProvider(agent_config.model.provider.clone()))?;

    let creds = config.credentials.get(&agent_config.model.provider);
    let api_key = resolve_api_key(provider, creds)?;

    // Credential base_url overrides catalog base_url
    let base_url = creds
        .and_then(|c| c.base_url.as_deref())
        .or(provider.base_url.as_deref());

    let tool = TerminalTool::new()
        .with_name(agent_name)
        .spawn()
        .await?;

    let model_name = &agent_config.model.name;

    let handle = match provider.api_type {
        ApiType::Anthropic => {
            let client = match base_url {
                Some(url) => anthropic::Client::builder().api_key(&api_key).base_url(url).build()?,
                None => anthropic::Client::new(&api_key)?,
            };
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::Anthropic(configure_agent(builder, agent_config))
        }
        ApiType::OpenAI => {
            let client = match base_url {
                Some(url) => openai::Client::builder().api_key(&api_key).base_url(url).build()?,
                None => openai::Client::new(&api_key)?,
            };
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::OpenAI(configure_agent(builder, agent_config))
        }
        ApiType::OpenAICompatible => {
            let client = match base_url {
                Some(url) => openai::CompletionsClient::builder().api_key(&api_key).base_url(url).build()?,
                None => openai::CompletionsClient::new(&api_key)?,
            };
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::OpenAICompatible(configure_agent(builder, agent_config))
        }
        ApiType::Gemini => {
            let client = match base_url {
                Some(url) => gemini::Client::builder().api_key(&api_key).base_url(url).build()?,
                None => gemini::Client::new(&api_key)?,
            };
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::Gemini(configure_agent(builder, agent_config))
        }
    };

    Ok(handle)
}
