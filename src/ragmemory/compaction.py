from __future__ import annotations

import os
from dataclasses import dataclass

from .llm import DEFAULT_LLM_PROVIDER, LLMProviderClient, LLMProviderOptions
from .models import EvidenceReference

DEFAULT_COMPACT_MODEL = os.environ.get(
    "STRUCTURED_MEMORY_MODEL", "minimaxai/minimax-m2.7"
)
DEFAULT_COMPACT_MIN_CHARS = 1500
DEFAULT_COMPACT_MAX_CHARS = 30000
DEFAULT_COMPACT_MAX_TOKENS = 1200
DEFAULT_COMPACT_TARGET_RATIO = 0.35
DEFAULT_COMPACT_MODE = "background"
DEFAULT_COMPACT_PROVIDER = os.environ.get("RAGMEMORY_COMPACT_PROVIDER", DEFAULT_LLM_PROVIDER)
COMPACT_STATUS_OK = "ok"
COMPACT_STATUS_FAILED = "failed"
COMPACT_STATUS_SKIPPED_SHORT = "skipped_short"
COMPACT_STATUS_TOO_LONG = "too_long"


def _strip_inline_comment(value: str) -> str:
    text = value.strip()
    for marker in (" #", "\t#", " ;", "\t;"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return _strip_inline_comment(value).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(_strip_inline_comment(value))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(_strip_inline_comment(value))
    except ValueError:
        return default


@dataclass
class MessageCompactionOptions:
    enabled: bool = False
    provider: str = DEFAULT_COMPACT_PROVIDER
    model: str = DEFAULT_COMPACT_MODEL
    min_chars: int = DEFAULT_COMPACT_MIN_CHARS
    max_chars: int = DEFAULT_COMPACT_MAX_CHARS
    max_tokens: int = DEFAULT_COMPACT_MAX_TOKENS
    target_ratio: float = DEFAULT_COMPACT_TARGET_RATIO
    mode: str = DEFAULT_COMPACT_MODE

    @classmethod
    def from_env(cls) -> "MessageCompactionOptions":
        mode = _strip_inline_comment(
            os.environ.get("RAGMEMORY_COMPACT_MODE", DEFAULT_COMPACT_MODE)
        ).lower()
        if mode not in {"background", "inline"}:
            mode = DEFAULT_COMPACT_MODE
        return cls(
            enabled=_env_bool("RAGMEMORY_COMPACT_ENABLE", False),
            provider=os.environ.get("RAGMEMORY_COMPACT_PROVIDER", DEFAULT_COMPACT_PROVIDER).strip().lower(),
            model=os.environ.get("RAGMEMORY_COMPACT_MODEL", DEFAULT_COMPACT_MODEL).strip(),
            min_chars=max(0, _env_int("RAGMEMORY_COMPACT_MIN_CHARS", DEFAULT_COMPACT_MIN_CHARS)),
            max_chars=max(0, _env_int("RAGMEMORY_COMPACT_MAX_CHARS", DEFAULT_COMPACT_MAX_CHARS)),
            max_tokens=max(1, _env_int("RAGMEMORY_COMPACT_MAX_TOKENS", DEFAULT_COMPACT_MAX_TOKENS)),
            target_ratio=max(
                0.05,
                min(_env_float("RAGMEMORY_COMPACT_TARGET_RATIO", DEFAULT_COMPACT_TARGET_RATIO), 1.0),
            ),
            mode=mode,
        )


class MessageCompactor:
    def __init__(self, options: LLMProviderOptions, max_tokens: int = DEFAULT_COMPACT_MAX_TOKENS):
        self.options = options
        self.model = options.model
        self.max_tokens = max_tokens
        self.last_error: str | None = None
        self.llm = LLMProviderClient(options)

    def compact(
        self,
        role: str,
        text: str,
        target_ratio: float,
        evidence_refs: list[EvidenceReference] | None = None,
    ) -> str | None:
        self.last_error = None
        if not text.strip():
            self.last_error = "empty message"
            return None

        compact_text = self.llm.complete_chat(
            messages=[
                {
                    "role": "system",
                    "content": "Compact noisy chat memory while preserving exact technical evidence.",
                },
                {
                    "role": "user",
                    "content": self._build_prompt(
                        role,
                        text,
                        target_ratio,
                        evidence_refs or [],
                    ),
                },
            ],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        if compact_text is None:
            self.last_error = self.llm.last_error or "unknown compaction failure"
            print(f"  [compact skipped] {self.options.provider} compaction failed: {self.last_error}")
            return None

        if not compact_text:
            self.last_error = "empty compact text"
            return None
        return compact_text

    def _build_prompt(
        self,
        role: str,
        text: str,
        target_ratio: float,
        evidence_refs: list[EvidenceReference],
    ) -> str:
        ratio_percent = int(target_ratio * 100)
        manifest = "\n".join(
            f"- {ref.marker}: {ref.preview}"
            for ref in evidence_refs
        ) or "- none"
        return f"""Role: {role}

Compact this chat message for long-term memory storage.
Target about {ratio_percent}% of the original length, but preserve important details over hitting the ratio.

Preserve verbatim:
- File paths, identifiers, URLs, config keys, and backticked tokens.
- Full error messages, exception names, tracebacks, and command output that explains a failure.
- Shell commands and flags.
- User preferences, decisions, constraints, and requested workflow rules.
- Numbers, dates, versions, ports, IDs, and model names.

Evidence references:
- For long fenced code/config/table blocks, you may use only the references listed below instead of repeating the full block.
- Do not invent references.
- Do not use references for inline file paths, URLs, errors, or commands; write those verbatim.

Available references:
{manifest}

Drop:
- Repeated IDE context blocks.
- Tool call boilerplate and "I am reading/checking" filler.
- Duplicated stack traces or logs after the first useful lines.
- Conversation filler that does not change future behavior.

Return compact text only. Do not wrap it in JSON or Markdown fences.

Message:
{text[:12000]}
"""
