#!/usr/bin/env python3
"""Generate models_dev_providers.pkl from models.dev api.json.

Fetches https://models.dev/api.json (or reads a local copy), filters to
providers whose npm package maps to a supported rig API type, and writes
a typed Pkl module.

Usage:
    python3 generate_providers.py [--fetch]

    --fetch   Download a fresh api.json from models.dev (default: use local copy)

Output: models_dev_providers.pkl (in the same directory as this script)
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# npm package → rig ApiType enum variant
NPM_TO_API_TYPE: dict[str, str] = {
    "@ai-sdk/anthropic": "Anthropic",
    "@ai-sdk/openai": "OpenAI",
    "@ai-sdk/openai-compatible": "OpenAICompatible",
    "@openrouter/ai-sdk-provider": "OpenAICompatible",
    "@ai-sdk/google": "Gemini",
    "@ai-sdk/google-vertex": "Gemini",
}


def pkl_string(s: str) -> str:
    """Escape a string for Pkl double-quoted literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def generate(api_json_path: Path) -> str:
    with open(api_json_path) as f:
        data = json.load(f)

    lines = [
        '/// Auto-generated from https://models.dev/api.json',
        '/// Do not edit — re-run generate_providers.py to update.',
        'module models_dev_providers',
        '',
        'class ProviderInfo {',
        '  name: String',
        '  api_type: "Anthropic"|"OpenAI"|"OpenAICompatible"|"Gemini"',
        '  base_url: String?',
        '  env: Listing<String>',
        '}',
        '',
        'providers: Mapping<String, ProviderInfo> = new {',
    ]

    count = 0
    for provider_id in sorted(data.keys()):
        provider = data[provider_id]
        if not isinstance(provider, dict):
            continue
        npm = provider.get("npm", "")
        api_type = NPM_TO_API_TYPE.get(npm)
        if api_type is None:
            continue

        name = pkl_string(provider.get("name", provider_id))
        base_url = provider.get("api")  # "api" field = base URL
        env_list = provider.get("env", [])

        lines.append(f'  ["{pkl_string(provider_id)}"] {{')
        lines.append(f'    name = "{name}"')
        lines.append(f'    api_type = "{api_type}"')
        if base_url:
            lines.append(f'    base_url = "{pkl_string(base_url)}"')
        lines.append("    env {")
        for var in env_list:
            lines.append(f'      "{pkl_string(var)}"')
        lines.append("    }")
        lines.append("  }")
        count += 1

    lines.append("}")
    lines.append("")  # trailing newline

    print(f"Generated {count} providers", file=sys.stderr)
    return "\n".join(lines)


def main() -> None:
    api_json = SCRIPT_DIR / "api.json"

    if "--fetch" in sys.argv:
        import urllib.request
        print("Fetching https://models.dev/api.json ...", file=sys.stderr)
        urllib.request.urlretrieve("https://models.dev/api.json", api_json)

    if not api_json.exists():
        print(f"Error: {api_json} not found. Run with --fetch or place api.json in {SCRIPT_DIR}", file=sys.stderr)
        sys.exit(1)

    output = generate(api_json)
    out_path = SCRIPT_DIR / "models_dev_providers.pkl"
    out_path.write_text(output)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
