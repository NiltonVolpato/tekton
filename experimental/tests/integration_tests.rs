mod common;

use std::path::Path;

use tekton_experimental::{build_agent, load_config};

use common::{write_config_schema, write_pkl};

/// Write a config that points at the mock server.
/// All credentials are in the Pkl config — no env vars needed.
///
/// The catalog entry has no `api`, so the credential `base_url` is the
/// only source. This exercises the "credential base_url overrides catalog"
/// code path.
fn mock_agent_config(dir: &Path) -> std::path::PathBuf {
    write_config_schema(dir);

    // Add mock provider to the catalog — no api URL here, so
    // credentials.base_url is the sole source of the endpoint.
    write_pkl(
        dir,
        "models_dev/providers.pkl",
        r#"amends "providers_models_dev.pkl"

providers {
  ["mock"] {
    id = "mock"
    metadata {
      name = "Mock Server"
      doc = "https://example.com"
    }
    client_type = "OpenAICompatible"
    env {}
    model {
      ["openai"] {
        id = "openai"
        metadata {
          name = "OpenAI Mock"
          release_date = "2025-01"
          last_updated = "2025-01"
        }
        capabilities {}
        modalities {
          input { "text" }
          output { "text" }
        }
        limit {
          context = 128000
          output = 4096
        }
      }
    }
  }
}
"#,
    );

    write_pkl(
        dir,
        "test.pkl",
        r#"
amends "Config.pkl"

default_agent = "mock-agent"

credentials {
  ["mock"] = new {
    api_key = "fake-key"
    base_url = "http://localhost:8100/v1"
  }
}

agents {
  ["mock-agent"] = new {
    model = new { provider = "mock"; name = "openai" }
    system_prompt = "You are a test agent."
  }
}
"#,
    )
}

#[tokio::test]
async fn integration_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let pkl = mock_agent_config(dir.path());
    let config = load_config(&pkl).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();
    let response = agent.prompt("Hello").await.unwrap();
    assert!(!response.is_empty());
}

#[tokio::test]
async fn integration_chat() {
    let dir = tempfile::tempdir().unwrap();
    let pkl = mock_agent_config(dir.path());
    let config = load_config(&pkl).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();
    let response = agent.chat("Hello", vec![]).await.unwrap();
    assert!(!response.is_empty());
}

#[tokio::test]
async fn integration_stream_chat() {
    let dir = tempfile::tempdir().unwrap();
    let pkl = mock_agent_config(dir.path());
    let config = load_config(&pkl).unwrap();
    let agent = build_agent(&config, "mock-agent").await.unwrap();

    use futures::StreamExt;
    use tekton_experimental::StreamEvent;

    let mut stream = agent.stream_chat("Hello", vec![]).await;
    let mut texts = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::Text(t)) => texts.push(t),
            Ok(StreamEvent::ToolCall { .. }) => {}
            Err(e) => panic!("stream returned an error: {e}"),
        }
    }
    assert!(!texts.is_empty());
}
