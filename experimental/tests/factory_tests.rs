mod common;

use tekton_experimental::{build_agent, load_config};

use common::{write_base_schema, write_pkl};

fn make_config(dir: &std::path::Path, provider: &str, model: &str) -> std::path::PathBuf {
    write_base_schema(dir);
    write_pkl(
        dir,
        "test.pkl",
        &format!(
            r#"
amends "AgentConfig.pkl"

name = "factory-test"
model {{
  provider = "{provider}"
  name = "{model}"
}}
system_prompt = "You are a test agent."
"#
        ),
    )
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn build_agent_anthropic() {
    let dir = tempfile::tempdir().unwrap();
    let pkl = make_config(dir.path(), "Anthropic", "claude-sonnet-4-20250514");
    let config = load_config(&pkl).unwrap();
    let _agent = build_agent(&config).await.unwrap();
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn build_agent_openai() {
    let dir = tempfile::tempdir().unwrap();
    let pkl = make_config(dir.path(), "OpenAI", "gpt-4o");
    let config = load_config(&pkl).unwrap();
    let _agent = build_agent(&config).await.unwrap();
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and OPENAI_BASE_URL"]
async fn build_agent_openai_compatible() {
    let dir = tempfile::tempdir().unwrap();
    let pkl = make_config(dir.path(), "OpenAICompatible", "llama-3.3-70b");
    let config = load_config(&pkl).unwrap();
    let _agent = build_agent(&config).await.unwrap();
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn build_agent_gemini() {
    let dir = tempfile::tempdir().unwrap();
    let pkl = make_config(dir.path(), "Gemini", "gemini-2.0-flash");
    let config = load_config(&pkl).unwrap();
    let _agent = build_agent(&config).await.unwrap();
}
