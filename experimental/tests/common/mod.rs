use std::io::Write;

/// Returns the mock server URL.
/// Panics if TEST_SERVER_URL is not set — use `just test` to run with the mock server.
#[allow(dead_code)]
pub fn test_server_url() -> String {
    std::env::var("TEST_SERVER_URL")
        .expect("TEST_SERVER_URL not set — run `just test` to start the mock server")
}

pub fn write_pkl(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

/// Write the base schema into the temp dir so amending configs can reference it.
pub fn write_base_schema(dir: &std::path::Path) -> std::path::PathBuf {
    let schema = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("pkl/AgentConfig.pkl"),
    )
    .unwrap();
    write_pkl(dir, "AgentConfig.pkl", &schema)
}
