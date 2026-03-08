mod common;

use tekton_experimental::{ClientType, load_config};

use common::{global_dir, workspace_pkl};

#[test]
fn load_config_with_anthropic_agent() {
    let config = load_config(workspace_pkl("anthropic"), global_dir()).unwrap();

    assert_eq!(config.default_agent, "test");

    let agent = &config.agents["test"];
    assert_eq!(agent.model.provider, "anthropic");
    assert_eq!(agent.model.name, "claude-sonnet-4-6");
    assert_eq!(agent.system_prompt, "You are a test agent.");
    assert_eq!(agent.temperature, Some(0.5));
    assert_eq!(agent.max_tokens, Some(4096));
    assert_eq!(agent.max_turns, 20); // default

    let provider = &config.providers["anthropic"];
    assert_eq!(provider.metadata.name, "Anthropic");
    assert_eq!(provider.client_type, ClientType::Anthropic);
    assert_eq!(provider.env, vec!["ANTHROPIC_API_KEY"]);
}

#[test]
fn load_config_with_openai_agent() {
    let config = load_config(workspace_pkl("openai"), global_dir()).unwrap();

    let agent = &config.agents["oai"];
    assert_eq!(agent.model.provider, "openai");
    assert_eq!(agent.model.name, "gpt-5.3-codex");
    assert_eq!(agent.temperature, None);
    assert_eq!(agent.max_tokens, None);

    let provider = &config.providers["openai"];
    assert_eq!(provider.client_type, ClientType::OpenAI);
}

#[test]
fn load_config_with_openai_compatible_agent() {
    let config = load_config(workspace_pkl("openai-compatible"), global_dir()).unwrap();

    let agent = &config.agents["router"];
    assert_eq!(agent.model.provider, "openrouter");
    assert_eq!(agent.model.name, "llama-3.3-70b");
    assert_eq!(agent.max_turns, 5);

    let provider = &config.providers["openrouter"];
    assert_eq!(provider.client_type, ClientType::OpenAICompatible);
    assert_eq!(
        provider.base_url.as_deref(),
        Some("https://openrouter.ai/api/v1")
    );
}

#[test]
fn load_config_with_gemini_agent() {
    let config = load_config(workspace_pkl("gemini"), global_dir()).unwrap();

    let agent = &config.agents["gem"];
    assert_eq!(agent.model.provider, "google");
    assert_eq!(agent.model.name, "gemini-3.1-pro");
    assert_eq!(agent.temperature, Some(1.0));

    let provider = &config.providers["google"];
    assert_eq!(provider.client_type, ClientType::Gemini);
}

#[test]
fn load_config_multiple_agents_deduplicates_providers() {
    let config = load_config(workspace_pkl("multiple-agents"), global_dir()).unwrap();

    assert_eq!(config.agents.len(), 3);
    // Only 2 unique providers despite 3 agents (two use anthropic)
    assert_eq!(config.providers.len(), 2);
    assert!(config.providers.contains_key("anthropic"));
    assert!(config.providers.contains_key("openai"));
}

#[test]
fn load_config_with_credentials() {
    let config = load_config(workspace_pkl("credentials"), global_dir()).unwrap();

    let anthropic_creds = &config.credentials["anthropic"];
    assert_eq!(anthropic_creds.api_key, "sk-ant-test-key");

    let custom_creds = &config.credentials["custom"];
    assert_eq!(custom_creds.api_key, "custom-key");
}

#[test]
fn load_config_with_custom_provider() {
    let config = load_config(workspace_pkl("custom-provider"), global_dir()).unwrap();

    let provider = &config.providers["my-local"];
    assert_eq!(provider.metadata.name, "My Local LLM");
    assert_eq!(provider.client_type, ClientType::OpenAICompatible);
    assert_eq!(
        provider.base_url.as_deref(),
        Some("http://localhost:8080/v1")
    );
    assert!(provider.env.is_empty());
}

#[test]
fn load_config_unknown_model_returns_error() {
    let result = load_config(workspace_pkl("unknown-model"), global_dir());
    assert!(result.is_err());
}

#[test]
fn load_config_missing_provider_returns_error() {
    // Pkl will fail because "nonexistent" is not in the providers catalog
    let result = load_config(workspace_pkl("missing-provider"), global_dir());
    assert!(result.is_err());
}

#[test]
fn load_config_invalid_pkl_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.pkl");
    std::fs::write(&path, "this is not valid pkl {{{").unwrap();
    let result = load_config(&path, global_dir());
    assert!(result.is_err());
}

#[test]
fn load_example_config() {
    let config = load_config(workspace_pkl("example"), global_dir()).unwrap();

    assert_eq!(config.default_agent, "coder");

    let agent = &config.agents["coder"];
    assert_eq!(agent.model.provider, "anthropic");
    assert_eq!(agent.model.name, "claude-sonnet-4-6");
    assert_eq!(agent.temperature, Some(0.3));
    assert_eq!(agent.max_tokens, Some(8192));
    assert_eq!(agent.max_turns, 20);

    let provider = &config.providers["anthropic"];
    assert_eq!(provider.client_type, ClientType::Anthropic);
}
