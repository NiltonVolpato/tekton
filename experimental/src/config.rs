use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::ConfigError;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelIdentifier {
    pub provider: String,
    pub name: String,
}

/// Maps npm package to rig client type. Determines which rig client to use.
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

/// A provider from the models.dev catalog.
/// Mirrors the Pkl `providerSchema.Provider` class.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Provider {
    pub id: String,
    pub env: Vec<String>,
    pub npm: String,
    pub client_type: ClientType,
    pub api: Option<String>,
    pub name: String,
    pub doc: String,
    pub models: HashMap<String, Model>,
}

/// A model within a provider's catalog.
/// Mirrors the Pkl `providerSchema.Models` class.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub family: Option<String>,
    pub attachment: bool,
    pub reasoning: bool,
    pub tool_call: bool,
    pub interleaved: Option<Interleaved>,
    pub structured_output: Option<bool>,
    pub temperature: Option<bool>,
    pub knowledge: Option<String>,
    pub release_date: String,
    pub last_updated: String,
    pub modalities: Modalities,
    pub open_weights: bool,
    pub cost: Option<Cost>,
    pub limit: Limit,
    pub status: Option<ModelStatus>,
    pub provider: Option<ModelProvider>,
}

/// Interleaved reasoning configuration.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Interleaved {
    Enabled(bool),
    Config { field: String },
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

/// Release status of a model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Alpha,
    Beta,
    Deprecated,
}

/// Pricing information for a model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub reasoning: Option<f64>,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
    pub input_audio: Option<f64>,
    pub output_audio: Option<f64>,
    pub context_over_200k: Option<ContextOver200k>,
}

/// Pricing for prompts exceeding 200k tokens.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ContextOver200k {
    pub input: f64,
    pub output: f64,
    pub reasoning: Option<f64>,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
    pub input_audio: Option<f64>,
    pub output_audio: Option<f64>,
}

/// Context window and token limits for a model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Limit {
    pub context: u64,
    pub input: Option<u64>,
    pub output: u64,
}

/// Provider-specific overrides within a model entry.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelProvider {
    pub npm: Option<String>,
    pub api: Option<String>,
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
