mod common;

use tekton_experimental::{ClientType, load_config};

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
    model = new { provider = "anthropic"; name = "claude-sonnet-4-6" }
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
    assert_eq!(agent.model.name, "claude-sonnet-4-6");
    assert_eq!(agent.system_prompt, "You are a test agent.");
    assert_eq!(agent.temperature, Some(0.5));
    assert_eq!(agent.max_tokens, Some(4096));
    assert_eq!(agent.max_turns, 20); // default

    // providers should contain anthropic
    let provider = &config.providers["anthropic"];
    assert_eq!(provider.metadata.name, "Anthropic");
    assert_eq!(provider.client_type, ClientType::Anthropic);
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
    model = new { provider = "openai"; name = "gpt-5.3-codex" }
    system_prompt = "Hello."
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
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

    let provider = &config.providers["openrouter"];
    assert_eq!(provider.client_type, ClientType::OpenAICompatible);
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
    model = new { provider = "google"; name = "gemini-3.1-pro" }
    system_prompt = "You are a Gemini agent."
    temperature = 1.0
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    let agent = &config.agents["gem"];
    assert_eq!(agent.model.provider, "google");
    assert_eq!(agent.model.name, "gemini-3.1-pro");
    assert_eq!(agent.temperature, Some(1.0));

    let provider = &config.providers["google"];
    assert_eq!(provider.client_type, ClientType::Gemini);
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
    model = new { provider = "anthropic"; name = "claude-sonnet-4-6" }
    system_prompt = "Agent A."
  }
  ["b"] = new {
    model = new { provider = "anthropic"; name = "claude-haiku-4-5" }
    system_prompt = "Agent B."
  }
  ["c"] = new {
    model = new { provider = "openai"; name = "gpt-5.3-codex" }
    system_prompt = "Agent C."
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.agents.len(), 3);
    // Only 2 unique providers despite 3 agents (two use anthropic)
    assert_eq!(config.providers.len(), 2);
    assert!(config.providers.contains_key("anthropic"));
    assert!(config.providers.contains_key("openai"));
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
    model = new { provider = "anthropic"; name = "claude-sonnet-4-6" }
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
    write_config_schema(dir.path());

    // providers.pkl adds a custom provider with a local model
    write_pkl(
        dir.path(),
        "models_dev/providers.pkl",
        r#"amends "providers_models_dev.pkl"

providers {
  ["my-local"] {
    id = "my-local"
    metadata {
      name = "My Local LLM"
      doc = "https://example.com"
    }
    client_type = "OpenAICompatible"
    base_url = "http://localhost:8080/v1"
    env {}
    model {
      ["qwen3.5-397b-a17b"] {
        id = "qwen3.5-397b-a17b"
        metadata {
          name = "Qwen 3.5 397B A17B"
          release_date = "2025-06"
          last_updated = "2025-06"
        }
        capabilities { "tool_call"; "open_weights" }
        modalities {
          input { "text" }
          output { "text" }
        }
        limit {
          context = 131072
          output = 8192
        }
      }
    }
  }
}
"#,
    );

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "local"

agents {
  ["local"] = new {
    model = new { provider = "my-local"; name = "qwen3.5-397b-a17b" }
    system_prompt = "Local model."
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
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
    model = new { provider = "anthropic"; name = "claude-nonexistent-9000" }
    system_prompt = "hello"
  }
}
"#,
    );

    let result = load_config(&pkl);
    assert!(result.is_err());
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
fn load_example_config() {
    let dir = tempfile::tempdir().unwrap();
    write_config_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "coder"

credentials {
  ["anthropic"] = new {
    api_key = "sk-test"
  }
}

agents {
  ["coder"] = new {
    model = new { provider = "anthropic"; name = "claude-sonnet-4-6" }
    system_prompt = """
      You are a software engineer. You have access to a terminal tool
      for running shell commands.
      """
    temperature = 0.3
    max_tokens = 8192
  }
}
"#,
    );

    let config = load_config(&pkl).unwrap();
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

/// Test hierarchical config: a "global" config directory (like ~/.tekton)
/// provides the schema, and a project directory amends it via --module-path.
#[test]
fn load_config_with_module_path_hierarchy() {
    let root = tempfile::tempdir().unwrap();

    // Global config location (e.g. ~/.tekton)
    let global_dir = root.path().join("global");
    std::fs::create_dir_all(&global_dir).unwrap();
    write_config_schema(&global_dir);

    // Project config in a separate directory
    let project_dir = root.path().join("project");
    std::fs::create_dir_all(&project_dir).unwrap();

    // The project config amends Config.pkl via modulepath: URI scheme.
    let project_config = project_dir.join("tekton.pkl");
    std::fs::write(
        &project_config,
        r#"amends "modulepath:/Config.pkl"

default_agent = "dev"

agents {
  ["dev"] = new {
    model = new { provider = "anthropic"; name = "claude-sonnet-4-6" }
    system_prompt = "Project agent."
  }
}
"#,
    )
    .unwrap();

    // Use pkl eval with --module-path to verify it works
    let output = std::process::Command::new("pkl")
        .args([
            "eval",
            "--module-path",
            global_dir.to_str().unwrap(),
            "-f",
            "json",
            project_config.to_str().unwrap(),
        ])
        .output()
        .expect("pkl must be installed");

    assert!(
        output.status.success(),
        "pkl eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse the JSON output to verify it's valid
    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("pkl output should be valid JSON");

    assert_eq!(json["default_agent"], "dev");
    assert_eq!(json["providers"]["anthropic"]["metadata"]["name"], "Anthropic");
    assert_eq!(
        json["providers"]["anthropic"]["client_type"],
        "Anthropic"
    );
}
