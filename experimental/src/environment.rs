/// Read-only view into environment variables.
///
/// Abstracts over `std::env::var` so that callers can be tested
/// without mutating the process environment.
pub(crate) trait EnvironmentView {
    /// Look up an environment variable by name.
    ///
    /// Returns `Some(value)` if the variable is set, `None` otherwise.
    fn var(&self, name: &str) -> Option<String>;
}

/// Forwards to [`std::env::var`].
pub(crate) struct RealEnvironment;

impl EnvironmentView for RealEnvironment {
    fn var(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

/// In-memory environment for tests.
#[cfg(test)]
pub(crate) struct MockEnvironment {
    vars: std::collections::HashMap<String, String>,
}

#[cfg(test)]
impl MockEnvironment {
    pub fn new() -> Self {
        Self {
            vars: std::collections::HashMap::new(),
        }
    }

    pub fn set(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.vars.insert(name.into(), value.into());
        self
    }
}

#[cfg(test)]
impl EnvironmentView for MockEnvironment {
    fn var(&self, name: &str) -> Option<String> {
        self.vars.get(name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_environment_returns_none_for_missing_var() {
        let env = MockEnvironment::new();
        assert_eq!(env.var("DOES_NOT_EXIST"), None);
    }

    #[test]
    fn mock_environment_returns_set_value() {
        let env = MockEnvironment::new().set("MY_VAR", "hello");
        assert_eq!(env.var("MY_VAR"), Some("hello".to_string()));
    }

    #[test]
    fn mock_environment_supports_multiple_vars() {
        let env = MockEnvironment::new()
            .set("A", "1")
            .set("B", "2");
        assert_eq!(env.var("A"), Some("1".to_string()));
        assert_eq!(env.var("B"), Some("2".to_string()));
        assert_eq!(env.var("C"), None);
    }

    #[test]
    fn mock_environment_last_set_wins() {
        let env = MockEnvironment::new()
            .set("KEY", "first")
            .set("KEY", "second");
        assert_eq!(env.var("KEY"), Some("second".to_string()));
    }

    #[test]
    fn real_environment_reads_known_var() {
        // PATH is always set on any Unix-like system
        let env = RealEnvironment;
        assert!(env.var("PATH").is_some());
    }

    #[test]
    fn real_environment_returns_none_for_missing_var() {
        let env = RealEnvironment;
        assert_eq!(env.var("TEKTON_TEST_DEFINITELY_NOT_SET_12345"), None);
    }
}
