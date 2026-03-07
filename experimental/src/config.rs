use std::collections::HashMap;
use std::path::Path;

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
/// Supports explicit API keys and base URL overrides.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Credentials {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
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
    pub model: HashMap<String, Model>,
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

/// Load a [`Config`] from a Pkl file.
///
/// # Errors
///
/// Returns [`ConfigError::PklError`] if the file cannot be read, parsed, or
/// deserialized into a `Config`.
pub fn load_config(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
    rpkl::from_config(path.as_ref()).map_err(ConfigError::from)
}
