"""Probe OpenCode Go reasoning controls for deepseek-v4-flash.

This script sends tiny chat/completions requests with different provider-specific
extra_body values and reports whether final content is returned. It does not
touch the RagMemory DB.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from openai import OpenAI

from test_llm_provider import DEFAULT_NVIDIA_MODEL, load_local_config, load_provider


CASES: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {}),
    ("reasoning_effort_low", {"reasoning_effort": "low"}),
    ("reasoning_effort_none", {"reasoning_effort": "none"}),
    ("reasoning_disabled", {"reasoning": {"enabled": False}}),
    ("thinking_false", {"thinking": False}),
    ("thinking_budget_0", {"thinking": {"budget_tokens": 0}}),
    ("thinking_budget_64", {"thinking": {"budget_tokens": 64}}),
    ("thinking_disabled_type", {"thinking": {"type": "disabled"}}),
    ("thinking_enabled_budget_64", {"thinking": {"type": "enabled", "budget_tokens": 64}}),
    ("thinking_auto_budget_64", {"thinking": {"type": "auto", "budget_tokens": 64}}),
]


def usage_value(usage: object, key: str) -> object:
    if usage is None:
        return None
    if hasattr(usage, key):
        return getattr(usage, key)
    if isinstance(usage, dict):
        return usage.get(key)
    return None


def reasoning_tokens(response) -> object:
    usage = getattr(response, "usage", None)
    details = usage_value(usage, "completion_tokens_details")
    if details is None:
        return None
    if hasattr(details, "reasoning_tokens"):
        return details.reasoning_tokens
    if isinstance(details, dict):
        return details.get("reasoning_tokens")
    return None


def run_case(client: OpenAI, model: str, max_tokens: int, name: str, extra_body: dict[str, Any]) -> None:
    print(f"=== {name} ===")
    if extra_body:
        print("extra_body:", json.dumps(extra_body, ensure_ascii=False))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Reply with exactly one word: ok. Do not explain.",
                },
                {"role": "user", "content": "Say ok."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
    except Exception as exc:
        print("error:", exc)
        print()
        return

    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    dumped_message = choice.message.model_dump()
    hidden_reasoning = dumped_message.get("reasoning_content") or dumped_message.get("reasoning")
    print("finish_reason:", choice.finish_reason)
    print("completion_tokens:", usage_value(response.usage, "completion_tokens"))
    print("reasoning_tokens:", reasoning_tokens(response))
    print("content_len:", len(content))
    print("content:", repr(content[:200]))
    if hidden_reasoning:
        print("reasoning_preview:", repr(str(hidden_reasoning)[:200]))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe OpenCode Go reasoning-control parameters.")
    parser.add_argument("--provider", default="opencode_go")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--case",
        choices=[name for name, _ in CASES],
        action="append",
        help="Run only one or more named cases. Default runs all cases.",
    )
    args = parser.parse_args(argv)

    load_local_config()
    options = load_provider(args.provider, DEFAULT_NVIDIA_MODEL)
    if not options.api_key:
        print(f"Provider failed: {options.provider} API key missing")
        return 1
    if options.provider != "opencode_go":
        print("This probe is intended for opencode_go.")
        return 1

    selected = set(args.case or [])
    cases = [(name, body) for name, body in CASES if not selected or name in selected]
    client = OpenAI(base_url=options.base_url, api_key=options.api_key)
    print(f"provider={options.provider} model={options.model} max_tokens={args.max_tokens}")
    print()
    for name, body in cases:
        run_case(client, options.model, args.max_tokens, name, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
