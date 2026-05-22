"""Smoke-test a configured RagMemory LLM provider without loading ChromaDB."""
from __future__ import annotations

import argparse
import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LLM_PROVIDER = "nvidia"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "minimaxai/minimax-m2.7"
DEFAULT_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_OPENCODE_GO_MODEL = "deepseek-v4-flash"
LLM_API_STYLE_OPENAI_CHAT = "openai_chat"


@dataclass
class ProviderOptions:
    provider: str
    api_key: str | None
    base_url: str
    model: str
    api_style: str
    extra_body: dict | None = None


def provider_env_prefix(provider: str) -> str:
    provider_key = re.sub(r"[^A-Za-z0-9]+", "_", provider.strip()).strip("_").upper()
    return f"RAGMEMORY_LLM_{provider_key}"


def load_local_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (
        repo_root / "ragmemory.local.ini",
        Path.cwd() / "ragmemory.local.ini",
        Path.home() / ".ragmemory.ini",
    ):
        if path.exists():
            load_settings_file(path)


def load_settings_file(path: Path) -> None:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8-sig")
    if parser.has_section("structured_memory"):
        section = parser["structured_memory"]
        if section.get("api_key"):
            os.environ.setdefault("NVIDIA_API_KEY", section["api_key"].strip())
            os.environ.setdefault("RAGMEMORY_LLM_NVIDIA_API_KEY", section["api_key"].strip())
        if section.get("model"):
            os.environ.setdefault("STRUCTURED_MEMORY_MODEL", section["model"].strip())
    if parser.has_section("llm"):
        section = parser["llm"]
        if section.get("compact_provider"):
            os.environ.setdefault("RAGMEMORY_COMPACT_PROVIDER", section["compact_provider"].strip())
        if section.get("structured_provider"):
            os.environ.setdefault("RAGMEMORY_STRUCTURED_PROVIDER", section["structured_provider"].strip())
    for section_name in parser.sections():
        if not section_name.startswith("llm."):
            continue
        provider = section_name.split(".", 1)[1].strip()
        if not provider:
            continue
        prefix = provider_env_prefix(provider)
        section = parser[section_name]
        for key, suffix in (
            ("api_key", "API_KEY"),
            ("base_url", "BASE_URL"),
            ("model", "MODEL"),
            ("api_style", "API_STYLE"),
            ("thinking", "THINKING"),
        ):
            if section.get(key):
                os.environ.setdefault(f"{prefix}_{suffix}", section[key].strip())


def load_provider(provider: str, fallback_model: str) -> ProviderOptions:
    provider = provider.strip().lower()
    prefix = provider_env_prefix(provider)
    api_key = os.environ.get(f"{prefix}_API_KEY")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    model = os.environ.get(f"{prefix}_MODEL")
    api_style = os.environ.get(f"{prefix}_API_STYLE", LLM_API_STYLE_OPENAI_CHAT)
    thinking = os.environ.get(f"{prefix}_THINKING", "").strip().lower()

    if provider == "nvidia":
        api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        base_url = base_url or DEFAULT_NVIDIA_BASE_URL
        model = model or fallback_model
    elif provider == "opencode_go":
        base_url = base_url or DEFAULT_OPENCODE_GO_BASE_URL
        model = model or DEFAULT_OPENCODE_GO_MODEL
    else:
        base_url = base_url or ""
        model = model or fallback_model

    return ProviderOptions(
        provider=provider,
        api_key=api_key.strip() if api_key else None,
        base_url=base_url.rstrip("/"),
        model=model.strip(),
        api_style=api_style.strip().lower(),
        extra_body=extra_body_for(provider, thinking),
    )


def extra_body_for(provider: str, thinking: str) -> dict | None:
    if provider != "opencode_go" or not thinking:
        return None
    if thinking == "disabled":
        return {"thinking": {"type": "disabled"}}
    if thinking == "enabled":
        return {"thinking": {"type": "enabled"}}
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send a tiny prompt to a RagMemory LLM provider.")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument(
        "--purpose",
        choices=("structured", "compact"),
        default="compact",
        help="Selects the fallback model if the provider section has no model.",
    )
    args = parser.parse_args(argv)

    load_local_config()
    fallback_model = os.environ.get("STRUCTURED_MEMORY_MODEL", DEFAULT_NVIDIA_MODEL)
    options = load_provider(args.provider, fallback_model)
    if not options.api_key:
        print(f"Provider failed: {options.provider} API key missing")
        return 1
    if options.api_style != LLM_API_STYLE_OPENAI_CHAT:
        print(f"Provider failed: unsupported api_style: {options.api_style}")
        return 1

    from openai import OpenAI

    client = OpenAI(base_url=options.base_url, api_key=options.api_key)
    try:
        response = client.chat.completions.create(
            model=options.model,
            messages=[
                {"role": "system", "content": "You are a health check endpoint. Reply with one short word."},
                {"role": "user", "content": "Say ok."},
            ],
            temperature=0,
            max_tokens=256,
            extra_body=options.extra_body,
        )
    except Exception as exc:
        print(f"Provider failed: {options.provider} ({exc})")
        return 1

    text = (response.choices[0].message.content or "").strip()
    if not text:
        print(f"Provider failed: {options.provider} returned empty text")
        return 1
    print(f"Provider OK: {options.provider} model={options.model}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
