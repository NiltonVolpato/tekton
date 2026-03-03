
nextest_args := "--status-level fail --show-progress none --no-output-indent --cargo-quiet"
mock_server_pid_file := "/tmp/mock-server.pid"


# Default: run all tests (starts mock server automatically)
# Examples: just test, just test -p my-crate, just test -E 'test(foo)'
test *args: _mock-server-start && _mock-server-stop
    cargo nextest run {{ nextest_args }} {{ args }}

# Fast tests only (no server needed)
test-fast *args:
    cargo nextest run {{ nextest_args }} {{ args }} -E 'not test(integration_)'

# Test coverage (starts mock server automatically)
coverage *args: _mock-server-start && _mock-server-stop
    cargo llvm-cov --fail-under-lines 80 nextest {{ nextest_args }} {{ args }}

# Build all crates
build:
    cargo build --workspace

# Lint
lint:
    cargo clippy --workspace -- -D warnings

# Regenerate models_dev_providers.pkl from models.dev api.json
generate-providers:
    python3 experimental/pkl/models_dev/generate_providers.py --fetch

# Start mock server in background, wait for readiness
_mock-server-start:
    #!/usr/bin/env bash
    set -euo pipefail
    cleanup() { kill "$(cat {{ mock_server_pid_file }})" 2>/dev/null || true; rm -f {{ mock_server_pid_file }}; }
    timeout 5m vidaimock &
    echo $! > {{ mock_server_pid_file }}
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
    if [ -f {{ mock_server_pid_file }} ]; then
        kill "$(cat {{ mock_server_pid_file }})" 2>/dev/null || true
        rm -f {{ mock_server_pid_file }}
    fi
