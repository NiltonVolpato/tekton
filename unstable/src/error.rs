use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to load config: {0}")]
    PklError(#[from] rpkl::Error),
}

#[derive(Debug, Error)]
pub enum FactoryError {
    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error("failed to spawn terminal tool: {0}")]
    TerminalSpawn(#[from] tekton_terminal_tool::TerminalError),

    #[error("failed to build provider client: {0}")]
    ClientBuild(#[from] rig::http_client::Error),

    #[error("no API key for provider `{provider}`: set `api_key` in credentials or env var {env}")]
    MissingApiKey { provider: String, env: String },

    #[error("agent `{0}` not found in config")]
    UnknownAgent(String),

    #[error("provider `{0}` not found in resolved config")]
    UnknownProvider(String),
}
