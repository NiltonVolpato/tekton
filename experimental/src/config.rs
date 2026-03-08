use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rpkl::api::reader::{PathElements, PklModuleReader};
use serde::Deserialize;

use crate::error::ConfigError;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelIdentifier {
    pub provider: String,
    pub name: String,
}

/// Determines which rig client to use for API calls.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum ClientType {
    Anthropic,
    OpenAI,
    OpenAICompatible,
    Gemini,
}

/// Provider credentials, separate from the catalog.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Credentials {
    pub api_key: String,
}

/// Display name and documentation link for a provider.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ProviderMetadata {
    pub name: String,
    pub doc: String,
}

/// A provider from the models.dev catalog.
/// Mirrors the Pkl `Provider` class in `providerSchema.pkl`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Provider {
    pub id: String,
    pub metadata: ProviderMetadata,
    pub client_type: ClientType,
    pub env: Vec<String>,
    pub base_url: Option<String>,
    pub models: HashMap<String, Model>,
}

/// Human-readable metadata about a model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub release_date: String,
    pub last_updated: String,
    pub knowledge: Option<String>,
    pub family: Option<String>,
}

/// A capability supported by a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    Attachment,
    Reasoning,
    ToolCall,
    Temperature,
    StructuredOutput,
    OpenWeights,
}

/// A modality supported by a model (input or output).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Audio,
    Image,
    Video,
    Pdf,
}

/// Input/output modalities supported by a model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Modalities {
    pub input: Vec<Modality>,
    pub output: Vec<Modality>,
}

/// Per-model overrides for provider-level settings.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ProviderOverride {
    pub client_type: Option<ClientType>,
    pub base_url: Option<String>,
}

/// Context window and token limits for a model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Limit {
    pub context: i64,
    pub input: Option<i64>,
    pub output: i64,
}

/// Lifecycle status of a model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Alpha,
    Beta,
    Deprecated,
}

/// A model within a provider's catalog.
/// Mirrors the Pkl `Model` class in `providerSchema.pkl`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Model {
    pub id: String,
    pub metadata: ModelMetadata,
    pub capabilities: Vec<ModelCapability>,
    pub modalities: Modalities,
    pub limit: Limit,
    pub provider_override: Option<ProviderOverride>,
    pub status: Option<ModelStatus>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AgentConfig {
    pub model: ModelIdentifier,
    pub system_prompt: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub max_turns: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub default_agent: String,
    pub agents: HashMap<String, AgentConfig>,
    pub credentials: HashMap<String, Credentials>,
    pub providers: HashMap<String, Provider>,
}

/// Resolves `global:/` URIs to files under a root directory.
///
/// Pkl files can use `import "global:/schema/Config.pkl"` to reference
/// files from the global install directory (e.g. `~/.tekton/`).
struct GlobalModuleReader {
    root: PathBuf,
}

impl GlobalModuleReader {
    fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Strip the `global:/` prefix and resolve against the root directory.
    fn resolve(&self, uri: &str) -> PathBuf {
        let relative = uri.strip_prefix("global:/").unwrap_or(uri);
        self.root.join(relative)
    }
}

impl PklModuleReader for GlobalModuleReader {
    fn scheme(&self) -> &str {
        "global"
    }

    fn has_hierarchical_uris(&self) -> bool {
        true
    }

    fn is_local(&self) -> bool {
        true
    }

    fn read(&self, uri: &str) -> Result<String, Box<dyn std::error::Error>> {
        let path = self.resolve(uri);
        std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path.display(), e).into())
    }

    fn list(&self, uri: &str) -> Result<Vec<PathElements>, Box<dyn std::error::Error>> {
        let path = self.resolve(uri);
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&path).map_err(|e| format!("{}: {}", path.display(), e))? {
            let entry = entry?;
            entries.push(PathElements::new(
                entry.file_name().to_string_lossy().to_string(),
                entry.file_type()?.is_dir(),
            ));
        }
        Ok(entries)
    }
}

/// Load a [`Config`] from a Pkl file.
///
/// The `global_dir` is the root for `global:/` URI resolution (e.g. `~/.tekton/`).
/// Pkl files can reference global modules with `import "global:/schema/Config.pkl"`.
///
/// # Errors
///
/// Returns [`ConfigError::PklError`] if the file cannot be read, parsed, or
/// deserialized into a `Config`.
pub fn load_config(
    path: impl AsRef<Path>,
    global_dir: impl Into<PathBuf>,
) -> Result<Config, ConfigError> {
    let reader = GlobalModuleReader::new(global_dir);
    let options = rpkl::EvaluatorOptions::new().add_client_module_readers(reader);
    rpkl::from_config_with_options(path.as_ref(), options).map_err(ConfigError::from)
}
