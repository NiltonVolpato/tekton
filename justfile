
nextest_args := "--status-level fail --show-progress none --no-output-indent --cargo-quiet"


# Default: run all tests (starts VidaiMock automatically)
# Examples: just test, just test -p my-crate, just test -E 'test(foo)'
test *args: _vidaimock-start
    VIDAIMOCK_URL=http://localhost:8100 \
    OPENAI_API_KEY=fake-key \
    OPENAI_BASE_URL=http://localhost:8100/v1 \
    cargo nextest {{ nextest_args }} {{ args }}
    just _vidaimock-stop

# Fast tests only (no server needed)
test-fast *args:
    cargo nextest {{ nextest_args }} {{ args }} -E 'not test(vidaimock)'

# Test coverage (starts VidaiMock automatically)
coverage *args: _vidaimock-start
    VIDAIMOCK_URL=http://localhost:8100 \
    OPENAI_API_KEY=fake-key \
    OPENAI_BASE_URL=http://localhost:8100/v1 \
    cargo llvm-cov --fail-under-lines 80 nextest {{ nextest_args }} {{ args }}
    just _vidaimock-stop

# Build all crates
build:
    cargo build --workspace

# Lint
lint:
    cargo clippy --workspace -- -D warnings

# Stop VidaiMock if running
_vidaimock-stop:
    -pkill -f vidaimock || true

# Start VidaiMock in background, wait for readiness
_vidaimock-start: _vidaimock-stop
    #!/usr/bin/env bash
    vidaimock &
    for i in $(seq 1 30); do
        curl -sf http://localhost:8100/health > /dev/null 2>&1 && break
        sleep 0.1
    done
