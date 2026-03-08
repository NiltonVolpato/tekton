use std::io::Write;
use std::path::Path;

pub fn write_pkl(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

/// Copy a file into the temp dir, preserving the relative path.
fn copy_to(src: &Path, dir: &Path, relative: &str) {
    let dest = dir.join(relative);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::copy(src, dest).unwrap();
}

/// The root of the `experimental` crate (resolved at compile time).
fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Write the full Config.pkl schema chain into a temp dir.
///
/// Copies production files:
/// - `providerSchema.pkl` — from `experimental/pkl/`
/// - `Config.pkl` — from `experimental/pkl/`
///
/// Copies test fixtures:
/// - `models_dev/providersModelsDev.pkl` — test provider catalog
/// - `models_dev/providers.pkl` — amends the catalog (no custom providers)
///
/// Returns the path to `Config.pkl` (for amending in test configs).
pub fn write_config_schema(dir: &Path) -> std::path::PathBuf {
    let pkl_dir = crate_root().join("pkl");
    let testdata_dir = crate_root().join("tests/testdata");

    // Production schema and config
    copy_to(&pkl_dir.join("providerSchema.pkl"), dir, "providerSchema.pkl");
    copy_to(&pkl_dir.join("Config.pkl"), dir, "Config.pkl");

    // Test provider catalog and user override layer
    copy_to(
        &testdata_dir.join("models_dev/providersModelsDev.pkl"),
        dir,
        "models_dev/providersModelsDev.pkl",
    );
    copy_to(
        &testdata_dir.join("providers.pkl"),
        dir,
        "providers.pkl",
    );

    dir.join("Config.pkl")
}
