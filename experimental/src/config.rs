use serde::Deserialize;
use std::path::Path;

use crate::error::ConfigError;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum Provider {
    OpenAI,
    OpenAICompatible,
    Anthropic,
    Gemini,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Model {
    pub provider: Provider,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub model: Model,
    pub system_prompt: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub max_turns: usize,
}

/// Load an [`AgentConfig`] from a Pkl file.
///
/// # Errors
///
/// Returns [`ConfigError::PklError`] if the file cannot be read, parsed, or
/// deserialized into an `AgentConfig`.
pub fn load_config(path: impl AsRef<Path>) -> Result<AgentConfig, ConfigError> {
    rpkl::from_config(path.as_ref()).map_err(ConfigError::from)
}
