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
}
