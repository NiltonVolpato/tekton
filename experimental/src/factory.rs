use rig::agent::{Agent, AgentBuilder, WithBuilderTools};
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::providers::{anthropic, gemini, openai};
use tekton_terminal_tool::TerminalTool;

use crate::config::{AgentConfig, ApiType, Config, Credentials, ResolvedProvider};
use crate::environment::{EnvironmentView, RealEnvironment};
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
    env: &dyn EnvironmentView,
) -> Result<String, FactoryError> {
    if let Some(c) = creds {
        // Credentials entry exists — api_key must be present.
        // Having a credentials entry signals explicit configuration;
        // if the key is missing, that's an error, not a fallback.
        return match &c.api_key {
            Some(key) => Ok(key.clone()),
            None => Err(FactoryError::MissingApiKey {
                provider: provider.name.clone(),
                env: "api_key in credentials".to_string(),
            }),
        };
    }
    // No credentials entry — try env vars from catalog
    for var in &provider.env {
        if let Some(key) = env.var(var) {
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
    let api_key = resolve_api_key(provider, creds, &RealEnvironment)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::MockEnvironment;

    fn test_provider() -> ResolvedProvider {
        ResolvedProvider {
            name: "test-provider".to_string(),
            api_type: ApiType::Anthropic,
            base_url: None,
            env: vec![
                "PRIMARY_KEY".to_string(),
                "SECONDARY_KEY".to_string(),
            ],
        }
    }

    #[test]
    fn resolve_api_key_from_credentials() {
        let provider = test_provider();
        let creds = Credentials {
            api_key: Some("explicit-key".to_string()),
            base_url: None,
        };
        let env = MockEnvironment::new();
        let result = resolve_api_key(&provider, Some(&creds), &env).unwrap();
        assert_eq!(result, "explicit-key");
    }

    #[test]
    fn resolve_api_key_credentials_without_key_is_error() {
        let provider = test_provider();
        let creds = Credentials {
            api_key: None,
            base_url: Some("https://custom.api".to_string()),
        };
        let env = MockEnvironment::new();
        let err = resolve_api_key(&provider, Some(&creds), &env).unwrap_err();
        let FactoryError::MissingApiKey { provider: p, env: e } = &err else {
            panic!("expected MissingApiKey, got: {err:?}");
        };
        assert_eq!(p, "test-provider");
        assert_eq!(e, "api_key in credentials");
    }

    #[test]
    fn resolve_api_key_credentials_without_key_does_not_fall_back_to_env() {
        let provider = test_provider();
        let creds = Credentials {
            api_key: None,
            base_url: None,
        };
        // Even if the env var is set, having a credentials entry without
        // api_key should be an error — not a silent fallback.
        let env = MockEnvironment::new().set("PRIMARY_KEY", "env-key");
        let result = resolve_api_key(&provider, Some(&creds), &env);
        assert!(result.is_err(), "should not fall back to env var when credentials entry exists");
    }

    #[test]
    fn resolve_api_key_no_credentials_uses_env_var() {
        let provider = test_provider();
        let env = MockEnvironment::new().set("PRIMARY_KEY", "from-env");
        let result = resolve_api_key(&provider, None, &env).unwrap();
        assert_eq!(result, "from-env");
    }

    #[test]
    fn resolve_api_key_no_credentials_tries_secondary_env_var() {
        let provider = test_provider();
        // Primary not set, secondary is set
        let env = MockEnvironment::new().set("SECONDARY_KEY", "from-secondary");
        let result = resolve_api_key(&provider, None, &env).unwrap();
        assert_eq!(result, "from-secondary");
    }

    #[test]
    fn resolve_api_key_no_credentials_no_env_is_error() {
        let provider = test_provider();
        let env = MockEnvironment::new();
        let err = resolve_api_key(&provider, None, &env).unwrap_err();
        let FactoryError::MissingApiKey { provider: p, env: e } = &err else {
            panic!("expected MissingApiKey, got: {err:?}");
        };
        assert_eq!(p, "test-provider");
        assert_eq!(e, "PRIMARY_KEY or SECONDARY_KEY");
    }
}
