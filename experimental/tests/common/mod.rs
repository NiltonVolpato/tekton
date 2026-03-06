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
      metadata {{
        name = "{name}"
        release_date = "2025-01"
        last_updated = "2025-01"
      }}
      capabilities {{}}
      modalities {{
        input {{ "text" }}
        output {{ "text" }}
      }}
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
/// - `providerSchema.pkl` — minimal provider schema for tests
/// - `models_dev/providersModelsDev.pkl` — a minimal provider catalog for tests
/// - `models_dev/providers.pkl` — amends the catalog (no custom providers)
/// - `Config.pkl` — the system config schema
///
/// Returns the path to `Config.pkl` (for amending in test configs).
pub fn write_config_schema(dir: &Path) -> std::path::PathBuf {
    // Minimal provider schema for tests
    write_pkl(
        dir,
        "providerSchema.pkl",
        r#"module providerSchema

class ProviderMetadata {
  name: String
  doc: String
}

typealias ClientType = "Anthropic"|"OpenAI"|"OpenAICompatible"|"Gemini"

class Provider {
  id: String
  metadata: ProviderMetadata
  client_type: ClientType
  env: Listing<String>
  base_url: String?
  model: Mapping<String, Model>
}

class ModelMetadata {
  name: String
  release_date: String
  last_updated: String
  knowledge: String?
  family: String?
}

typealias Modality = "text"|"audio"|"image"|"video"|"pdf"

class Modalities {
  input: Listing<Modality>
  output: Listing<Modality>
}

typealias ModelCapability =
  "attachment"
  | "reasoning"
  | "tool_call"
  | "temperature"
  | "structured_output"
  | "open_weights"

class ProviderOverride {
  client_type: ClientType?
  base_url: String?
}

class Limit {
  context: Int
  input: Int?
  output: Int
}

typealias ModelStatus = "alpha"|"beta"|"deprecated"

class Model {
  id: String
  metadata: ModelMetadata
  capabilities: Listing<ModelCapability>
  modalities: Modalities
  limit: Limit
  provider_override: ProviderOverride?
  status: ModelStatus?
}
"#,
    );

    let claude_sonnet = test_model_entry("claude-sonnet-4-6", "Claude Sonnet 4.6");
    let claude_haiku = test_model_entry("claude-haiku-4-5", "Claude Haiku 4.5");
    let gpt5_codex = test_model_entry("gpt-5.3-codex", "GPT-5.3 Codex");
    let gemini_pro = test_model_entry("gemini-3.1-pro", "Gemini 3.1 Pro");
    let llama = test_model_entry("llama-3.3-70b", "Llama 3.3 70B");

    // Test catalog with a few providers (self-contained, mirrors generated output)
    write_pkl(
        dir,
        "models_dev/providersModelsDev.pkl",
        &format!(
            r#"import "../providerSchema.pkl" as providerSchema

providers: Mapping<String, providerSchema.Provider> = new {{
  ["anthropic"] {{
    id = "anthropic"
    metadata {{
      name = "Anthropic"
      doc = "https://docs.anthropic.com"
    }}
    client_type = "Anthropic"
    env {{ "ANTHROPIC_API_KEY" }}
    model {{
{claude_sonnet}
{claude_haiku}
    }}
  }}
  ["openai"] {{
    id = "openai"
    metadata {{
      name = "OpenAI"
      doc = "https://platform.openai.com/docs"
    }}
    client_type = "OpenAI"
    env {{ "OPENAI_API_KEY" }}
    model {{
{gpt5_codex}
    }}
  }}
  ["openrouter"] {{
    id = "openrouter"
    metadata {{
      name = "OpenRouter"
      doc = "https://openrouter.ai/docs"
    }}
    client_type = "OpenAICompatible"
    base_url = "https://openrouter.ai/api/v1"
    env {{ "OPENROUTER_API_KEY" }}
    model {{
{llama}
    }}
  }}
  ["google"] {{
    id = "google"
    metadata {{
      name = "Google"
      doc = "https://ai.google.dev/docs"
    }}
    client_type = "Gemini"
    env {{ "GEMINI_API_KEY" }}
    model {{
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
        r#"amends "providersModelsDev.pkl"

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
