use std::io::Write;

/// Returns the VidaiMock URL.
/// Panics if VIDAIMOCK_URL is not set — use `just test` to run with VidaiMock.
#[allow(dead_code)]
pub fn vidaimock_url() -> String {
    std::env::var("VIDAIMOCK_URL")
        .expect("VIDAIMOCK_URL not set — run `just test` to start VidaiMock")
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
