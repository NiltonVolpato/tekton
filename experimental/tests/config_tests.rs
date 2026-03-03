mod common;

use tekton_experimental::{ApiType, load_config};

use common::{write_config_schema, write_pkl};

#[test]
fn load_config_with_anthropic_agent() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "test"

agents {
  ["test"] = new {
    model = new { provider = "anthropic"; name = "claude-sonnet-4-20250514" }
    system_prompt = "You are a test agent."
    temperature = 0.5
    max_tokens = 4096
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.default_agent, "test");

    let agent = &config.agents["test"];
    assert_eq!(agent.model.provider, "anthropic");
    assert_eq!(agent.model.name, "claude-sonnet-4-20250514");
    assert_eq!(agent.system_prompt, "You are a test agent.");
    assert_eq!(agent.temperature, Some(0.5));
    assert_eq!(agent.max_tokens, Some(4096));
    assert_eq!(agent.max_turns, 20); // default

    // resolved_providers should contain anthropic
    let provider = &config.resolved_providers["anthropic"];
    assert_eq!(provider.name, "Anthropic");
    assert_eq!(provider.api_type, ApiType::Anthropic);
    assert_eq!(provider.env, vec!["ANTHROPIC_API_KEY"]);
}

#[test]
fn load_config_with_openai_agent() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "oai"

agents {
  ["oai"] = new {
    model = new { provider = "openai"; name = "gpt-4o" }
    system_prompt = "Hello."
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    let agent = &config.agents["oai"];
    assert_eq!(agent.model.provider, "openai");
    assert_eq!(agent.model.name, "gpt-4o");
    assert_eq!(agent.temperature, None);
    assert_eq!(agent.max_tokens, None);

    let provider = &config.resolved_providers["openai"];
    assert_eq!(provider.api_type, ApiType::OpenAI);
}

#[test]
fn load_config_with_openai_compatible_agent() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "router"

agents {
  ["router"] = new {
    model = new { provider = "openrouter"; name = "llama-3.3-70b" }
    system_prompt = "You are a local model."
    max_turns = 5
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    let agent = &config.agents["router"];
    assert_eq!(agent.model.provider, "openrouter");
    assert_eq!(agent.model.name, "llama-3.3-70b");
    assert_eq!(agent.max_turns, 5);

    let provider = &config.resolved_providers["openrouter"];
    assert_eq!(provider.api_type, ApiType::OpenAICompatible);
    assert_eq!(
        provider.base_url.as_deref(),
        Some("https://openrouter.ai/api/v1")
    );
}

#[test]
fn load_config_with_gemini_agent() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "gem"

agents {
  ["gem"] = new {
    model = new { provider = "google"; name = "gemini-2.0-flash" }
    system_prompt = "You are a Gemini agent."
    temperature = 1.0
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    let agent = &config.agents["gem"];
    assert_eq!(agent.model.provider, "google");
    assert_eq!(agent.model.name, "gemini-2.0-flash");
    assert_eq!(agent.temperature, Some(1.0));

    let provider = &config.resolved_providers["google"];
    assert_eq!(provider.api_type, ApiType::Gemini);
}

#[test]
fn load_config_multiple_agents_deduplicates_providers() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "a"

agents {
  ["a"] = new {
    model = new { provider = "anthropic"; name = "claude-sonnet-4-20250514" }
    system_prompt = "Agent A."
  }
  ["b"] = new {
    model = new { provider = "anthropic"; name = "claude-haiku-4-5-20251001" }
    system_prompt = "Agent B."
  }
  ["c"] = new {
    model = new { provider = "openai"; name = "gpt-4o" }
    system_prompt = "Agent C."
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.agents.len(), 3);
    // Only 2 unique providers despite 3 agents (two use anthropic)
    assert_eq!(config.resolved_providers.len(), 2);
    assert!(config.resolved_providers.contains_key("anthropic"));
    assert!(config.resolved_providers.contains_key("openai"));
}

#[test]
fn load_config_with_credentials() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "test"

credentials {
  ["anthropic"] = new {
    api_key = "sk-ant-test-key"
  }
  ["custom"] = new {
    api_key = "custom-key"
    base_url = "http://localhost:8080/v1"
  }
}

agents {
  ["test"] = new {
    model = new { provider = "anthropic"; name = "claude-sonnet-4-20250514" }
    system_prompt = "Test."
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    let anthropic_creds = &config.credentials["anthropic"];
    assert_eq!(anthropic_creds.api_key.as_deref(), Some("sk-ant-test-key"));
    assert_eq!(anthropic_creds.base_url, None);

    let custom_creds = &config.credentials["custom"];
    assert_eq!(custom_creds.api_key.as_deref(), Some("custom-key"));
    assert_eq!(
        custom_creds.base_url.as_deref(),
        Some("http://localhost:8080/v1")
    );
}

#[test]
fn load_config_with_custom_provider() {
    let dir = tempfile::tempdir().unwrap();

    // Write a catalog that includes a custom provider
    write_pkl(
        dir.path(),
        "models_dev/models_dev_providers.pkl",
        r#"module models_dev_providers

class ProviderInfo {
  name: String
  api_type: "Anthropic"|"OpenAI"|"OpenAICompatible"|"Gemini"
  base_url: String?
  env: Listing<String>
}

providers: Mapping<String, ProviderInfo> = new {
  ["anthropic"] {
    name = "Anthropic"
    api_type = "Anthropic"
    env { "ANTHROPIC_API_KEY" }
  }
}
"#,
    );

    // providers.pkl adds a custom provider
    write_pkl(
        dir.path(),
        "models_dev/providers.pkl",
        r#"amends "models_dev_providers.pkl"

providers {
  ["my-local"] {
    name = "My Local LLM"
    api_type = "OpenAICompatible"
    base_url = "http://localhost:8080/v1"
    env {}
  }
}
"#,
    );

    let config_schema = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("pkl/Config.pkl"),
    )
    .unwrap();
    write_pkl(dir.path(), "Config.pkl", &config_schema);

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "local"

agents {
  ["local"] = new {
    model = new { provider = "my-local"; name = "llama-3" }
    system_prompt = "Local model."
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    let provider = &config.resolved_providers["my-local"];
    assert_eq!(provider.name, "My Local LLM");
    assert_eq!(provider.api_type, ApiType::OpenAICompatible);
    assert_eq!(
        provider.base_url.as_deref(),
        Some("http://localhost:8080/v1")
    );
    assert!(provider.env.is_empty());
}

#[test]
fn load_config_missing_provider_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "test"

agents {
  ["test"] = new {
    model = new { provider = "nonexistent"; name = "some-model" }
    system_prompt = "hello"
  }
}
"#,
    );

    // Pkl will fail because "nonexistent" is not in the providers catalog
    let result = load_config(&pkl);
    assert!(result.is_err());
}

#[test]
fn load_config_invalid_pkl_returns_error() {
    let dir = tempfile::tempdir().unwrap();

    let pkl = write_pkl(dir.path(), "bad.pkl", "this is not valid pkl {{{");
    let result = load_config(&pkl);
    assert!(result.is_err());
}

#[test]
fn load_example_coder_config() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("pkl/examples/coder.pkl");
    let config = load_config(&path).unwrap();
    assert_eq!(config.default_agent, "coder");

    let agent = &config.agents["coder"];
    assert_eq!(agent.model.provider, "anthropic");
    assert_eq!(agent.model.name, "claude-sonnet-4-20250514");
    assert_eq!(agent.temperature, Some(0.3));
    assert_eq!(agent.max_tokens, Some(8192));
    assert_eq!(agent.max_turns, 20);

    let provider = &config.resolved_providers["anthropic"];
    assert_eq!(provider.api_type, ApiType::Anthropic);
}
