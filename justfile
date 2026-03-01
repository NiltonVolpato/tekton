
nextest_args := "--status-level fail --show-progress none --no-output-indent --cargo-quiet"
test_server_pid_file := shell("mktemp -t pid")

export TEST_SERVER_URL := "http://localhost:8100"
export OPENAI_API_KEY := "fake-key"
export OPENAI_BASE_URL := "http://localhost:8100/v1"

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

# Start mock server in background, wait for readiness
_mock-server-start:
    #!/usr/bin/env bash
    timeout 5m vidaimock &
    echo $! > {{ test_server_pid_file }}
    for i in $(seq 1 30); do
        curl -sf http://localhost:8100/health > /dev/null 2>&1 && break
        sleep 0.1
    done

# Stop mock server if running
_mock-server-stop:
    -kill $(cat {{ test_server_pid_file }})
