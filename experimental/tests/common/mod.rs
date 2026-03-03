use std::io::Write;
use std::path::Path;

pub fn write_pkl(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

/// Write the full Config.pkl schema chain into a temp dir.
///
/// Creates:
/// - `models_dev/models_dev_providers.pkl` — a minimal provider catalog for tests
/// - `models_dev/providers.pkl` — amends the catalog (no custom providers)
/// - `Config.pkl` — the system config schema
///
/// Returns the path to `Config.pkl` (for amending in test configs).
pub fn write_config_schema(dir: &Path) -> std::path::PathBuf {
    // Minimal test catalog with a few providers
    write_pkl(
        dir,
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
  ["openai"] {
    name = "OpenAI"
    api_type = "OpenAI"
    env { "OPENAI_API_KEY" }
  }
  ["openrouter"] {
    name = "OpenRouter"
    api_type = "OpenAICompatible"
    base_url = "https://openrouter.ai/api/v1"
    env { "OPENROUTER_API_KEY" }
  }
  ["google"] {
    name = "Google"
    api_type = "Gemini"
    env { "GEMINI_API_KEY" }
  }
}
"#,
    );

    write_pkl(
        dir,
        "models_dev/providers.pkl",
        r#"amends "models_dev_providers.pkl"

providers {}
"#,
    );

    // Copy Config.pkl from the real source
    let config_schema = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("pkl/Config.pkl"),
    )
    .unwrap();
    write_pkl(dir, "Config.pkl", &config_schema)
}
