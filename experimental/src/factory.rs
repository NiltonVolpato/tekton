use rig::agent::{Agent, AgentBuilder, WithBuilderTools};
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::providers::{anthropic, gemini, openai};
use tekton_terminal_tool::TerminalTool;

use crate::config::{AgentConfig, ClientType, Config, Credentials, Provider};
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
/// - If a `Credentials` entry exists, use its `api_key` (always present, no env fallback).
/// - If no `Credentials` entry exists, try env vars from the provider's `env` list.
/// - If neither, return an error.
fn resolve_api_key(
    provider: &Provider,
    creds: Option<&Credentials>,
    env: &dyn EnvironmentView,
) -> Result<String, FactoryError> {
    if let Some(c) = creds {
        return Ok(c.api_key.clone());
    }
    for var in &provider.env {
        if let Some(key) = env.var(var) {
            return Ok(key);
        }
    }
    Err(FactoryError::MissingApiKey {
        provider: provider.metadata.name.clone(),
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
        .providers
        .get(&agent_config.model.provider)
        .ok_or_else(|| FactoryError::UnknownProvider(agent_config.model.provider.clone()))?;

    let creds = config.credentials.get(&agent_config.model.provider);
    let api_key = resolve_api_key(provider, creds, &RealEnvironment)?;

    // Apply model-specific overrides if present
    let model_override = provider
        .models
        .get(&agent_config.model.name)
        .and_then(|m| m.provider_override.as_ref());

    let effective_client_type = model_override
        .and_then(|o| o.client_type)
        .unwrap_or(provider.client_type);

    let effective_base_url = model_override
        .and_then(|o| o.base_url.as_deref())
        .or(provider.base_url.as_deref());

    let tool = TerminalTool::new()
        .with_name(agent_name)
        .spawn()
        .await?;

    let model_name = &agent_config.model.name;

    macro_rules! build_client {
        ($client:ty, $api_key:expr, $base_url:expr) => {{
            let mut builder = <$client>::builder().api_key($api_key);
            if let Some(url) = $base_url {
                builder = builder.base_url(url);
            }
            builder.build()?
        }};
    }

    let handle = match effective_client_type {
        ClientType::Anthropic => {
            let client = build_client!(anthropic::Client, &api_key, effective_base_url);
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::Anthropic(configure_agent(builder, agent_config))
        }
        ClientType::OpenAI => {
            let client = build_client!(openai::Client, &api_key, effective_base_url);
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::OpenAI(configure_agent(builder, agent_config))
        }
        ClientType::OpenAICompatible => {
            let client = build_client!(openai::CompletionsClient, &api_key, effective_base_url);
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::OpenAICompatible(configure_agent(builder, agent_config))
        }
        ClientType::Gemini => {
            let client = build_client!(gemini::Client, &api_key, effective_base_url);
            let builder = client.agent(model_name).tool(tool);
            AgentHandle::Gemini(configure_agent(builder, agent_config))
        }
    };

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::environment::MockEnvironment;

    fn test_provider() -> Provider {
        Provider {
            id: "test-provider".to_string(),
            metadata: crate::config::ProviderMetadata {
                name: "test-provider".to_string(),
                doc: "https://example.com".to_string(),
            },
            client_type: ClientType::Anthropic,
            env: vec![
                "PRIMARY_KEY".to_string(),
                "SECONDARY_KEY".to_string(),
            ],
            base_url: None,
            models: HashMap::new(),
        }
    }

    #[test]
    fn resolve_api_key_from_credentials() {
        let provider = test_provider();
        let creds = Credentials {
            api_key: "explicit-key".to_string(),
        };
        let env = MockEnvironment::new();
        let result = resolve_api_key(&provider, Some(&creds), &env).unwrap();
        assert_eq!(result, "explicit-key");
    }

    #[test]
    fn resolve_api_key_credentials_take_precedence_over_env() {
        let provider = test_provider();
        let creds = Credentials {
            api_key: "explicit-key".to_string(),
        };
        let env = MockEnvironment::new().set("PRIMARY_KEY", "env-key");
        let result = resolve_api_key(&provider, Some(&creds), &env).unwrap();
        assert_eq!(result, "explicit-key");
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
