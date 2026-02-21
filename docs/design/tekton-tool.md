---
title: Technical Design Document
---

# Design Doc: Tekton Tool — The Terminal-Native Single-Tool LLM Interface

**Author**: @nilton

**Date**: 02-20-2026

## Introduction

### Background

Current LLM agent frameworks (Claude Code, Cursor, Aider, etc.) rely on verbose tool-calling conventions: large JSON schemas in the system prompt, multiple specialized tools (read, write, edit, grep, find, task management), and custom protocols that the model must learn per-framework. This approach creates system prompt bloat, increases token costs on every message, and forces models to operate within an abstraction that diverges from their strongest pretraining signal — the Unix command line.

Tekton is a broader agent framework; this document focuses specifically on its core primitive: the **Tekton Tool** — a single LLM tool called `type` that sends keystrokes to a persistent terminal session. The model already knows how to use a shell. Every other capability — file manipulation, web search, code execution, sub-agents — is just a command on the PATH. The model discovers available tools the same way a human developer would: by running `help` and reading `--help` output.

### Current Functionality (Prior Art)

Claude Code provides a representative example of the status quo:

- **Multiple dedicated tools**: `read`, `write`, `edit`, `grep`, `find`, `task`, `add_task`, `update_task`, `list_tasks`, all MCP tools, plus a `bash` tool as a catch-all.
- **Heavy system prompt**: Thousands of tokens of tool schemas, behavioral instructions, and constraints (e.g., "never propose a change without reading all related files" — which causes redundant file reads even when the model just read them).
- **Stateless execution**: Each bash invocation is independent — no persistent shell session, no environment variable persistence, no working directory state.
- **JSON-based tool protocol**: Tool calls use structured JSON with name, input schema, and output, requiring the model to context-switch between natural language and a framework-specific structured format.

OpenAI Codex, Gemini CLI, and similar tools follow essentially the same pattern with minor variations.

### In Scope

- The `type` tool specification: parameters, semantics, and behavior.
- PTY session management required by the tool: lifecycle, state persistence, input/output handling.
- Mechanisms for detecting command completion, handling interactive input, managing background jobs, and enforcing timeouts.
- Shell configuration required to support the tool (sentinels, traps, job control).
- System prompt design and tool discovery protocol.

### Out of Scope

- The broader Tekton framework architecture (orchestration, plugin system, etc.).
- Specific implementations of builtin commands (the actual scripts on the PATH).
- Multi-user or multi-tenant deployment.
- Authentication, authorization, and sandboxing details.
- Specific model fine-tuning or training.
- UI/frontend for the agent.
- The shell emulation layer (if used instead of a real shell) — this is a separate design.

### Assumptions & Dependencies

- The LLM supports tool calling via its native API (any provider: Anthropic, OpenAI, open-source via vLLM/llama.cpp, etc.).
- A single tool with three parameters is sufficient for all interaction.
- Models have strong pretraining on Unix shell usage and can discover capabilities via `help`/`--help` without explicit schemas.
- The host environment can allocate a PTY (or pipe pair) for the agent session.
- Bash is available as the shell.

### Terminology

- **Tekton**: The broader agent framework.
- **Tekton Tool**: The single `type` tool described in this document — the core interface between the LLM and the terminal. This is the subject of this design doc.
- **Harness**: The orchestration layer that sits between the LLM and the PTY, managing tool calls, output capture, timeouts, and notifications.
- **PTY**: Pseudoterminal — the Unix mechanism that emulates a terminal device.
- **OSC**: Operating System Command — an escape sequence format used for machine-readable metadata in terminal output (e.g., `\e]999;...\a`).
- **Sentinel**: A machine-readable marker emitted by the shell to signal events (command completion, exit codes, etc.) to the harness.

## Considerations

### Concerns

**Extra round-trips on first interaction.** Unlike frameworks that pre-load tool documentation, Tekton may require the model to run `help` and possibly `cmd --help` before it can act. Mitigation: commonly used command descriptions can be included in the system prompt, and the model quickly internalizes available commands after the first few turns.

**Model reliability with plain-text tool boundaries.** If using the model's native tool-calling API, this is not a concern — the API handles structured output. If a future iteration uses plain-text delimiters (e.g., `<shell>...</shell>`), there's a small risk of malformed output, mitigated by constrained decoding or stop sequences.

**Timeout tuning.** The default timeouts (300s non-interactive, 5s interactive) are reasonable starting points but may need adjustment per-use-case. The model can override via the `timeout` parameter, but poorly chosen values could cause premature kills or long waits. This is inherent to the design and acceptable.

**Context window pressure from long command output.** Commands that produce large output (e.g., `cat` on a big file, verbose build logs) could fill the context window. Mitigation: the harness truncates output beyond a configurable limit and appends a message indicating truncation. The model already knows patterns like piping to `head`/`tail`.

### Operational Readiness Considerations

**Deployment**: Tekton runs as a local process managing a PTY. Deployment is a single binary plus any builtin scripts on the PATH.

**Debugging**: Because the agent operates a real terminal, debugging is straightforward — attach to the PTY from another terminal and watch the session in real time.

**Recovery**: The PTY session is ephemeral. If the shell dies, the harness can start a new one. In-progress state is lost, same as closing a terminal window.

### Open Questions

- Should the harness support session persistence (save/restore shell state across agent restarts)? Probably not in v1.
- Should the model be able to request specific terminal capabilities (e.g., TERM=xterm-256color vs TERM=dumb)? Probably not — use a sensible default and let the model adapt.
- What is the maximum reasonable number of concurrent background jobs before the notification channel becomes noisy?

## Proposed Design

### Solution

Tekton Tool is the core interface primitive of the Tekton framework, built on three components:

1. **A persistent bash session** managed via a PTY, with shell hooks for machine-readable event signaling.
2. **A single LLM tool** called `type` that sends keystrokes to the terminal.
3. **A harness** that mediates between the LLM and the PTY, handling output capture, timeout enforcement, and asynchronous job notifications.

From the model's perspective, it is a user sitting at a terminal with a keyboard. It can either speak (generate natural language for the user) or type (send keystrokes to the terminal). Everything else — file I/O, web search, code execution, sub-agents — is a command.

### System Architecture

```
┌──────────────────────────────────────────────────┐
│                    User                          │
│            (chat interface)                       │
└──────────────┬───────────────────────────────────┘
               │ natural language
               ▼
┌──────────────────────────────────────────────────┐
│                    LLM                           │
│         (any model with tool calling)            │
│                                                  │
│  Outputs either:                                 │
│    • Text response (to user)                     │
│    • type(keys, interactive, timeout) tool call  │
└──────────────┬───────────────────────────────────┘
               │ tool call
               ▼
┌──────────────────────────────────────────────────┐
│               Tekton Harness                     │
│                                                  │
│  • Sends keystrokes to PTY                       │
│  • Captures output until sentinel or timeout     │
│  • Reads job notifications from named pipe       │
│  • Enforces timeout policy (kill vs return)      │
│  • Returns output to LLM as tool result          │
└─────┬──────────────────────────────┬─────────────┘
      │ keystrokes / PTY I/O         │ async notifications
      ▼                              ▼
┌─────────────────────┐    ┌────────────────────┐
│   Persistent PTY    │    │    Named Pipe       │
│   (bash session)    │    │  (job completion)   │
│                     │    │                     │
│  • PROMPT_COMMAND   │    │  • SIGCHLD trap     │
│    (OSC sentinels)  │    │    writes here      │
│  • set +m           │    │                     │
│  • Custom builtins  │    │  • Harness reads    │
│    on PATH          │    │    asynchronously   │
└─────────────────────┘    └────────────────────┘
```

### The `type` Tool

The single tool exposed to the LLM.

| Parameter     | Type   | Required | Default                                        | Description                                                                                         |
|---------------|--------|----------|------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `keys`        | string | yes      | —                                              | Keystrokes to send to the terminal. Supports escape sequences (e.g., `\n` for Enter, `\x03` for Ctrl-C). |
| `interactive` | bool   | no       | `false`                                        | Timeout behavior. `false`: kill the foreground process on timeout. `true`: return partial output to the model for further input. |
| `timeout`     | number | no       | `300` if `interactive=false`, `5` if `interactive=true` | Seconds to wait before triggering the timeout behavior. Overrides the default for either mode.         |

The model can change `interactive` on every call, even mid-command. This is not a session mode — it's a per-keystroke policy.

**Examples:**

```
# Run a build (long-running, non-interactive)
type(keys="make -j8\n", interactive=false)

# Start an interactive program
type(keys="python game.py\n", interactive=true)

# Respond to a prompt
type(keys="Nilton\n", interactive=true)

# Respond to a yes/no prompt
type(keys="y\n", interactive=true)

# Kill a hung process
type(keys="\x03", interactive=false)

# Long-running interactive program with slow responses
type(keys="analyze dataset.csv\n", interactive=true, timeout=30)
```

### Shell Session Configuration

The bash session is initialized with the following configuration:

```bash
# Sentinel for command completion — emits OSC with exit code
PROMPT_COMMAND='printf "\e]999;%d\a" $?'

# Disable job control messages ([1] Done, [1]+ Stopped, etc.)
set +m

# SIGCHLD trap — writes a signal byte to the named pipe
TEKTON_PIPE="/tmp/tekton-jobs-$$"
mkfifo "$TEKTON_PIPE"
trap '__tekton_sigchld_handler' SIGCHLD

__tekton_sigchld_handler() {
    # Signal the harness that some child's state changed.
    # No structured data: the harness determines which job(s) finished
    # by running `jobs -p` and diffing against its tracked state.
    printf 'x' > "$TEKTON_PIPE"
}
```

**Key configuration choices:**

- **`PROMPT_COMMAND` with OSC sentinel**: The harness detects `\e]999;EXIT_CODE\a` to know when a command has completed and what its exit code was. OSC escapes are invisible to normal terminal rendering and won't appear in command output, making them unambiguous.
- **`set +m`**: Disables monitor mode, suppressing bash's default `[1]+ Done  command` messages. These messages would appear asynchronously in the PTY output, interleaved with other content, and cannot be reliably filtered by text matching (the same string could appear in any program's output, e.g., a man page discussing job control). Background jobs, `jobs`, and `&` still work. SIGCHLD is still delivered. The tradeoff is losing `fg`/`bg`, but processes can be managed by PID (`kill`, `wait $pid`, `kill -STOP`, `kill -CONT`).
- **Named pipe for job notifications**: A separate channel from the PTY output stream. The SIGCHLD trap writes a single signal byte here — no structured data. It is not reliably possible to determine from inside the trap which job completed (the state may already be reaped by the time the handler runs). Instead, the harness uses the byte as a prompt to run `jobs -p` and diff the result against its tracked job state (`JobManager`). The harness has a dedicated reader task for this pipe.

**What the model sees as its prompt:**

```
[0] claude@alpha:/Users/nilton/src/tekton $
```

Format: `[EXIT_CODE] LLM_NAME@HOSTNAME:PWD $`. The exit code gives immediate feedback on the last command. The model's own name in the prompt reinforces its identity within the session.

### Harness Behavior

**Output capture flow:**

1. Model makes a `type` call.
2. Harness writes `keys` to the PTY's stdin.
3. Harness reads PTY output, buffering content.
4. Harness waits for either:
   - The OSC sentinel (command completed) → return full output + exit code to model.
   - Timeout reached with `interactive=false` → kill foreground process group, return output + error message.
   - Timeout reached with `interactive=true` → return partial output to model, model decides next action.
5. After the command completes, harness runs `jobs -p` on the PTY and syncs `JobManager` with the result. Any PIDs that were tracked but are no longer listed have completed; their callback fires out-of-band (not bundled into the tool result).

**Background job notification flow:**

The named pipe reader task runs concurrently with `call`. When it receives a signal byte:

- **If a foreground command is in progress** (`is_running = true`): ignore. The `call` will sync `JobManager` via `jobs -p` when the command finishes anyway.
- **If the tool is idle** (`is_running = false`): run `jobs -p` on the PTY, sync `JobManager`. If completions are detected, the callback fires and the harness can queue a proactive message to inject into the agent's next turn.

**Timeout enforcement (without killing the shell):**

When a non-interactive timeout is reached, the harness identifies the foreground process by iterating child processes of the shell PID, filtering out known background jobs, and sends SIGKILL (or SIGINT first, then SIGKILL) to that process or its process group. The shell itself survives because it's not in the foreground process group — standard job control behavior even with `set +m`.

**Echo is disabled.** The harness controls the PTY's termios settings and disables echo. The model doesn't need to see its own keystrokes repeated back — it knows what it typed. The output stream contains only command output and the PROMPT_COMMAND sentinel.

**Input mode for non-interactive commands**: For `interactive=false`, stdin for the spawned command is effectively closed (`< /dev/null` semantics or equivalent). This prevents commands from blocking on input unexpectedly. For `interactive=true`, stdin remains connected to the PTY so the model can type into the running program.

### System Prompt Design

The system prompt is minimal by design:

```
You're looking at a terminal. The only tool you have is `type`, which sends
keystrokes to the terminal.

[type tool schema from the LLM's native tool-calling format]

The terminal says:

Welcome back Claude! For help run the command `help`.
[0] claude@alpha:/Users/nilton/src/tekton $
```

Optionally, for coding-focused sessions, commonly used builtin commands can be summarized in the system prompt to save the initial `help` round-trip. But the minimum viable system prompt is just the above.

**Tool discovery protocol:**

- `help` — lists all available builtins with one-line descriptions.
- `command --help` — detailed usage for any specific command.
- Standard Unix commands (`ls`, `cat`, `grep`, `curl`, etc.) are available as expected.
- Custom tools are executables on the PATH. Adding a tool = deploying a script. Removing a tool = removing the script. No prompt changes, no schema updates.

**When no tools are needed:**

Many conversations require zero tool calls. "Why is the sky blue?" → the model answers from its knowledge. "Who was Rayleigh?" → same. The minimal system prompt creates no gravitational pull toward unnecessary tool use, unlike frameworks with verbose tool schemas that bias the model toward reaching for tools even when its own knowledge suffices.

### Background Jobs and Sub-Agents

Background job management uses standard Unix patterns:

```bash
# Launch a background task
long_task > /tmp/task1.log 2>&1 &
# Model sees: [1] 12345

# Check status
jobs -p

# Wait for completion (or get notified via named pipe)
wait 12345

# Sub-agent as a background job
agent "refactor the auth module" > /tmp/refactor.log 2>&1 &

# Check sub-agent output later
cat /tmp/refactor.log
```

The model already knows these patterns from pretraining. Output is redirected to files to prevent interleaving with foreground output. The harness tracks background PIDs via `JobManager` and notifies the model of completions without requiring it to poll.

### Work Required

1. **Harness (Rust)**: PTY allocation, tool call dispatch, output capture with sentinel detection, timeout enforcement, named pipe reader, process group management.
2. **Shell initialization**: Bash profile/rc with PROMPT_COMMAND, set +m, SIGCHLD trap, named pipe setup.
3. **LLM integration**: Adapter for the `type` tool to at least one model API (Anthropic as primary, with the architecture supporting any provider).
4. **Builtin commands**: Initial set of useful commands as scripts on the PATH (help, agent, etc.).
5. **Basic chat loop**: Minimal UI that accepts user input and displays model output + tool call results.

## Impact

### Performance

- **Token efficiency**: Minimal system prompt (tens of tokens vs. thousands). Tool documentation is loaded on-demand via `help`/`--help` and only when needed.
- **Latency**: One extra round-trip if the model needs to discover tools. Eliminated if common commands are summarized in the system prompt.
- **Context window**: Persistent shell state means the model doesn't re-read files or re-establish context as often. Long command output is truncated by the harness.

### Security

The agent has whatever permissions the shell session has. Sandboxing is orthogonal to this design — use containers, VMs, seccomp, or a shell emulation layer to restrict capabilities. The design is compatible with all of these; the model doesn't know or care what's behind the PTY.

## Alternatives

### Status Quo: Multiple Specialized Tools

The Claude Code / Cursor / Aider approach. Multiple tools with JSON schemas, each with specific parameters and behaviors.

**Why not**: System prompt bloat, token waste, forces the model into an unnatural abstraction, adding tools requires prompt changes, no composability between tools (can't pipe `read` into `grep`).

### Plain-Text Delimiters Instead of API Tool Calling

Using `<shell>command</shell>` in the model's text output instead of the model's native tool-calling API.

**Why not**: Works fine for a single tool, but native tool calling is already supported by all major APIs, gives cleaner separation between speech and action, and benefits from any provider-side constrained decoding. The plain-text approach is a valid fallback for models without tool-calling support.

### Stateless Command Execution

Each tool call spawns a fresh shell, executes, and exits. No persistent session.

**Why not**: Loses environment state (`cd`, env vars, `source`), can't do background jobs, can't interact with long-running processes, forces the model to repeat setup on every call.

### Zsh Instead of Bash

Zsh offers `precmd`/`preexec` hooks, `TRAPCHLD`, better globbing, and associative arrays.

**Why not**: Bash is overwhelmingly dominant in pretraining data. Stack Overflow answers, Dockerfiles, CI scripts, and tutorials are almost all bash. The model will generate more correct shell code on the first try with bash. The features zsh offers over bash for this use case (hooks, traps) have adequate bash equivalents (`PROMPT_COMMAND`, `DEBUG` trap, SIGCHLD trap).

## Looking into the Future

- **Shell emulation layer**: Instead of a real bash session, a lightweight emulator that intercepts commands, providing finer-grained control over permissions, sandboxing, and virtual filesystems — while presenting the same interface to the model.
- **Multi-agent orchestration**: The `agent` command + background jobs provide a natural primitive for multi-agent workflows. Future iterations could add shared state, inter-agent communication via pipes or shared files, and coordination primitives.
- **Session persistence**: Save and restore shell sessions across agent restarts, enabling long-running tasks that survive disconnections.
- **Model-driven terminal capability negotiation**: Let the model request specific terminal features (colors, cursor movement) for use cases that benefit from them (e.g., TUI interaction).
- **Interactive terminal observation**: Allow human users to attach to the agent's PTY in real time, watching and optionally intervening in the agent's work — true pair programming with an AI.
