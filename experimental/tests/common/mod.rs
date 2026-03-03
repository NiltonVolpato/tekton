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

/// Minimal Pkl model entry for tests.
/// Provides all required fields with sensible defaults.
fn test_model_entry(id: &str, name: &str) -> String {
    format!(
        r#"    ["{id}"] {{
      id = "{id}"
      name = "{name}"
      attachment = false
      reasoning = false
      tool_call = false
      release_date = "2025-01"
      last_updated = "2025-01"
      modalities {{
        input {{ "text" }}
        output {{ "text" }}
      }}
      open_weights = false
      limit {{
        context = 200000
        output = 8192
      }}
    }}"#
    )
}

/// Write the full Config.pkl schema chain into a temp dir.
///
/// Creates:
/// - `models_dev/providerSchema.pkl` — minimal provider schema for tests
/// - `models_dev/providers_base.pkl` — declares typed providers property
/// - `models_dev/providers_models_dev.pkl` — a minimal provider catalog for tests
/// - `models_dev/providers.pkl` — amends the catalog (no custom providers)
/// - `Config.pkl` — the system config schema
///
/// Returns the path to `Config.pkl` (for amending in test configs).
pub fn write_config_schema(dir: &Path) -> std::path::PathBuf {
    // Minimal provider schema for tests
    write_pkl(
        dir,
        "models_dev/providerSchema.pkl",
        r#"module providerSchema

class Provider {
  id: String
  env: Listing<String>
  npm: String
  client_type: "Anthropic"|"OpenAI"|"OpenAICompatible"|"Gemini"
  api: String?
  name: String
  doc: String
  models: Mapping<String, Models>
}

class Models {
  id: String
  name: String
  family: String?
  attachment: Boolean
  reasoning: Boolean
  tool_call: Boolean
  interleaved: (Boolean(this == true)|InterleavedAlternate1)?
  structured_output: Boolean?
  temperature: Boolean?
  knowledge: String?
  release_date: String
  last_updated: String
  modalities: Modalities
  open_weights: Boolean
  cost: Cost?
  limit: Limit
  status: ("alpha"|"beta"|"deprecated")?
  provider: ModelsProvider?
}

class InterleavedAlternate1 {
  field: "reasoning_content"|"reasoning_details"
}

class Modalities {
  input: Listing<String>
  output: Listing<String>
}

class Cost {
  input: Number
  output: Number
  reasoning: Number?
  cache_read: Number?
  cache_write: Number?
  input_audio: Number?
  output_audio: Number?
  context_over_200k: ContextOver200k?
}

class ContextOver200k {
  input: Number
  output: Number
  reasoning: Number?
  cache_read: Number?
  cache_write: Number?
  input_audio: Number?
  output_audio: Number?
}

class Limit {
  context: Number
  input: Number?
  output: Number
}

class ModelsProvider {
  npm: String?
  api: String?
}
"#,
    );

    // Base module declaring typed property
    write_pkl(
        dir,
        "models_dev/providers_base.pkl",
        r#"import "providerSchema.pkl"

providers: Mapping<String, providerSchema.Provider>
"#,
    );

    let claude_sonnet = test_model_entry("claude-sonnet-4-6", "Claude Sonnet 4.6");
    let claude_haiku = test_model_entry("claude-haiku-4-5", "Claude Haiku 4.5");
    let gpt5_codex = test_model_entry("gpt-5.3-codex", "GPT-5.3 Codex");
    let gemini_pro = test_model_entry("gemini-3.1-pro", "Gemini 3.1 Pro");
    let llama = test_model_entry("llama-3.3-70b", "Llama 3.3 70B");

    // Test catalog with a few providers
    write_pkl(
        dir,
        "models_dev/providers_models_dev.pkl",
        &format!(
            r#"amends "providers_base.pkl"

providers {{
  ["anthropic"] {{
    id = "anthropic"
    name = "Anthropic"
    npm = "@ai-sdk/anthropic"
    client_type = "Anthropic"
    doc = "https://docs.anthropic.com"
    env {{ "ANTHROPIC_API_KEY" }}
    models {{
{claude_sonnet}
{claude_haiku}
    }}
  }}
  ["openai"] {{
    id = "openai"
    name = "OpenAI"
    npm = "@ai-sdk/openai"
    client_type = "OpenAI"
    doc = "https://platform.openai.com/docs"
    env {{ "OPENAI_API_KEY" }}
    models {{
{gpt5_codex}
    }}
  }}
  ["openrouter"] {{
    id = "openrouter"
    name = "OpenRouter"
    npm = "@openrouter/ai-sdk-provider"
    client_type = "OpenAICompatible"
    api = "https://openrouter.ai/api/v1"
    doc = "https://openrouter.ai/docs"
    env {{ "OPENROUTER_API_KEY" }}
    models {{
{llama}
    }}
  }}
  ["google"] {{
    id = "google"
    name = "Google"
    npm = "@ai-sdk/google"
    client_type = "Gemini"
    doc = "https://ai.google.dev/docs"
    env {{ "GEMINI_API_KEY" }}
    models {{
{gemini_pro}
    }}
  }}
}}
"#
        ),
    );

    // User amendment layer
    write_pkl(
        dir,
        "models_dev/providers.pkl",
        r#"amends "providers_models_dev.pkl"

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
