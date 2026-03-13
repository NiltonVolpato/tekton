# justfile for tekton

nextest_args := "--status-level fail --show-progress none --no-output-indent --cargo-quiet"
mock_server_config_dir := justfile_directory() / "experimental/tests/testdata/config"
mock_server_pid_file := "/tmp/mock-server.pid"

# Runs all tests. Examples: just test, just test -p my-crate, just test -E 'test(foo)'
[group('test')]
test +args='--workspace': _mock-server-start && _mock-server-stop
    cargo nextest run {{ nextest_args }} {{ args }}

# Fast tests only (no server needed)
[group('test')]
test-fast +args='--workspace':
    cargo nextest run {{ nextest_args }} {{ args }} -E 'not test(integration_)'

# Test coverage (starts mock server automatically)
[group('test')]
coverage +args='--workspace': _mock-server-start && _mock-server-stop
    cargo llvm-cov --fail-under-lines 80 nextest {{ nextest_args }} {{ args }}

# Runs clippy
[group('check')]
lint:
    cargo clippy --workspace -- -D warnings

# Check for compilation errors
[group('check')]
check:
    cargo check --workspace

# Build all crates
[group('dev')]
build:
    cargo build --workspace

[group('dev')]
fmt *args:
    cargo fmt {{ args }} -- --config unstable_features=true,imports_granularity=Module,group_imports=StdExternalCrate

# Regenerate providers list in Pkl from models.dev API
[group('dev')]
generate-providers:
    cd experimental/pkl/models_dev && pkl run generate-providers.pkl -- --output providers.pkl

[group('demo')]
mock-tool-call: (_mock-server-start '1h') && _mock-server-stop
    cargo run -p tekton-experimental --example agent_factory --quiet -- \
        experimental/tests/testdata/workspaces/mock-tool-call/tekton.pkl \
        experimental/pkl

# Start mock server in background, wait for readiness
_mock-server-start mock-server-timeout='5m':
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v vidaimock &>/dev/null; then
        echo "ERROR: vidaimock not found. Install with: cargo install --git https://github.com/vidaiUK/VidaiMock" >&2
        exit 1
    fi
    cleanup() { pkill -F "{{ mock_server_pid_file }}" vidaimock 2>/dev/null || true; rm -f "{{ mock_server_pid_file }}"; }
    if command -v timeout &>/dev/null; then
        timeout "{{ mock-server-timeout }}" vidaimock --config-dir "{{ mock_server_config_dir }}" &
    else
        vidaimock --config-dir "{{ mock_server_config_dir }}" &
    fi
    echo $! > "{{ mock_server_pid_file }}"
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8100/health > /dev/null 2>&1; then
            exit 0
        fi
        sleep 0.1
    done
    echo "ERROR: mock server failed to become ready after 3 seconds" >&2
    cleanup
    exit 1

# Stop mock server if running
_mock-server-stop:
    #!/usr/bin/env bash
    if [ -f "{{ mock_server_pid_file }}" ]; then
        pkill -F "{{ mock_server_pid_file }}" vidaimock 2>/dev/null || true
        rm -f "{{ mock_server_pid_file }}"
    fi
    echo "Mock server stopped"
