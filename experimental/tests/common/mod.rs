use std::path::{Path, PathBuf};

/// The root of the `experimental` crate (resolved at compile time).
fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The `global:` directory — production Pkl schema and config files.
pub fn global_dir() -> PathBuf {
    crate_root().join("pkl")
}

/// Path to a workspace's `tekton.pkl` in testdata.
pub fn workspace_pkl(name: &str) -> PathBuf {
    crate_root()
        .join("tests/testdata/workspaces")
        .join(name)
        .join("tekton.pkl")
}
