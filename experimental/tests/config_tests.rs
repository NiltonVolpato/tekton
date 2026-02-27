mod common;

use tekton_experimental::{Provider, load_config};

use common::{write_base_schema, write_pkl};

#[test]
fn load_config_from_pkl_file() {
    let dir = tempfile::tempdir().unwrap();
    write_base_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "AgentConfig.pkl"

name = "test-agent"
model {
  provider = "Anthropic"
  name = "claude-sonnet-4-20250514"
}
system_prompt = "You are a test agent."
temperature = 0.5
max_tokens = 4096
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.name, "test-agent");
    assert_eq!(config.model.provider, Provider::Anthropic);
    assert_eq!(config.model.name, "claude-sonnet-4-20250514");
    assert_eq!(config.system_prompt, "You are a test agent.");
    assert_eq!(config.temperature, Some(0.5));
    assert_eq!(config.max_tokens, Some(4096));
    assert_eq!(config.max_turns, 20); // default
}

#[test]
fn load_config_with_openai_provider() {
    let dir = tempfile::tempdir().unwrap();
    write_base_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "AgentConfig.pkl"

name = "openai-agent"
model {
  provider = "OpenAI"
  name = "gpt-4o"
}
system_prompt = "Hello."
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.model.provider, Provider::OpenAI);
    assert_eq!(config.model.name, "gpt-4o");
    assert_eq!(config.temperature, None);
    assert_eq!(config.max_tokens, None);
}

#[test]
fn load_config_with_openai_compatible_provider() {
    let dir = tempfile::tempdir().unwrap();
    write_base_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "AgentConfig.pkl"

name = "local-agent"
model {
  provider = "OpenAICompatible"
  name = "llama-3.3-70b"
}
system_prompt = "You are a local model."
max_turns = 5
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.model.provider, Provider::OpenAICompatible);
    assert_eq!(config.model.name, "llama-3.3-70b");
    assert_eq!(config.max_turns, 5);
}

#[test]
fn load_config_with_gemini_provider() {
    let dir = tempfile::tempdir().unwrap();
    write_base_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "AgentConfig.pkl"

name = "gemini-agent"
model {
  provider = "Gemini"
  name = "gemini-2.0-flash"
}
system_prompt = "You are a Gemini agent."
temperature = 1.0
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.model.provider, Provider::Gemini);
    assert_eq!(config.model.name, "gemini-2.0-flash");
    assert_eq!(config.temperature, Some(1.0));
}

#[test]
fn load_config_with_amends_chain() {
    let dir = tempfile::tempdir().unwrap();
    write_base_schema(dir.path());

    // Base concrete config
    write_pkl(
        dir.path(),
        "base.pkl",
        r#"
amends "AgentConfig.pkl"

name = "base-agent"
model {
  provider = "Anthropic"
  name = "claude-sonnet-4-20250514"
}
system_prompt = "Base prompt."
temperature = 0.7
max_tokens = 2048
max_turns = 10
"#,
    );

    // Override that amends the base
    let pkl = write_pkl(
        dir.path(),
        "override.pkl",
        r#"
amends "base.pkl"

name = "override-agent"
system_prompt = "Override prompt."
max_turns = 30
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(config.name, "override-agent");
    // Inherited from base
    assert_eq!(config.model.provider, Provider::Anthropic);
    assert_eq!(config.model.name, "claude-sonnet-4-20250514");
    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_tokens, Some(2048));
    // Overridden
    assert_eq!(config.system_prompt, "Override prompt.");
    assert_eq!(config.max_turns, 30);
}

#[test]
fn load_config_with_string_interpolation() {
    let dir = tempfile::tempdir().unwrap();
    write_base_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "AgentConfig.pkl"

name = "interp-agent"
model {
  provider = "Anthropic"
  name = "claude-sonnet-4-20250514"
}
system_prompt = "You are \(name). Your model is \(model.name)."
"#,
    );

    let config = load_config(&pkl).unwrap();
    assert_eq!(
        config.system_prompt,
        "You are interp-agent. Your model is claude-sonnet-4-20250514."
    );
}

#[test]
fn load_config_invalid_pkl_returns_error() {
    let dir = tempfile::tempdir().unwrap();

    let pkl = write_pkl(dir.path(), "bad.pkl", "this is not valid pkl {{{");
    let result = load_config(&pkl);
    assert!(result.is_err());
}

#[test]
fn load_config_invalid_provider_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    write_base_schema(dir.path());

    let pkl = write_pkl(
        dir.path(),
        "test.pkl",
        r#"
amends "AgentConfig.pkl"

name = "bad-provider"
model {
  provider = "NotAProvider"
  name = "some-model"
}
system_prompt = "hello"
"#,
    );

    // Pkl itself validates the union type, so this should fail at eval time
    let result = load_config(&pkl);
    assert!(result.is_err());
}

#[test]
fn load_example_coder_config() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("pkl/examples/coder.pkl");
    let config = load_config(&path).unwrap();
    assert_eq!(config.name, "coder");
    assert_eq!(config.model.provider, Provider::Anthropic);
    assert_eq!(config.model.name, "claude-sonnet-4-20250514");
    assert_eq!(config.temperature, Some(0.3));
    assert_eq!(config.max_tokens, Some(8192));
    assert_eq!(config.max_turns, 20);
}
