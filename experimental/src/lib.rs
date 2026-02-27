mod config;
mod error;
mod factory;
mod handle;

pub use config::{AgentConfig, Model, Provider, load_config};
pub use error::{ConfigError, FactoryError};
pub use factory::build_agent;
pub use handle::{AgentHandle, StreamEvent, TextStream};
