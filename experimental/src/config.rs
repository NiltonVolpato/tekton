use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::ConfigError;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelIdentifier {
    pub provider: String,
    pub name: String,
}

/// Maps npm package to rig API type. Determines which rig client to use.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum ApiType {
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
/// Deserialized from Pkl's `resolved_providers` map.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ResolvedProvider {
    pub name: String,
    pub api_type: ApiType,
    pub base_url: Option<String>,
    pub env: Vec<String>,
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
    pub resolved_providers: HashMap<String, ResolvedProvider>,
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
