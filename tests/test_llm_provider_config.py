"""Verify LLM provider configuration stays provider-agnostic."""
import os

from ragmemory.memory import (
    DEFAULT_OPENCODE_GO_BASE_URL,
    LLMProviderClient,
    LLMProviderOptions,
    MessageCompactionOptions,
    StructuredExtractionOptions,
    StructuredMemoryExtractor,
)


KEYS = [
    "RAGMEMORY_LLM_OPENCODE_GO_API_KEY",
    "RAGMEMORY_LLM_OPENCODE_GO_BASE_URL",
    "RAGMEMORY_LLM_OPENCODE_GO_MODEL",
    "RAGMEMORY_LLM_OPENCODE_GO_API_STYLE",
    "RAGMEMORY_LLM_OPENCODE_GO_THINKING",
    "RAGMEMORY_STRUCTURED_MAX_CHARS",
    "RAGMEMORY_STRUCTURED_MAX_TOKENS",
    "RAGMEMORY_COMPACT_MIN_CHARS",
    "RAGMEMORY_COMPACT_MAX_TOKENS",
    "RAGMEMORY_COMPACT_TARGET_RATIO",
]
saved = {key: os.environ.get(key) for key in KEYS}

try:
    for key in KEYS:
        os.environ.pop(key, None)
    os.environ["RAGMEMORY_LLM_OPENCODE_GO_API_KEY"] = "test-key"

    options = LLMProviderOptions.from_env("opencode_go", "fallback-model")
    assert options.provider == "opencode_go"
    assert options.api_key == "test-key"
    assert options.base_url == DEFAULT_OPENCODE_GO_BASE_URL
    assert options.model == "deepseek-v4-flash"
    assert options.api_style == "openai_chat"
    assert options.extra_body is None

    os.environ["RAGMEMORY_LLM_OPENCODE_GO_THINKING"] = "disabled"
    options = LLMProviderOptions.from_env("opencode_go", "fallback-model")
    assert options.extra_body == {"thinking": {"type": "disabled"}}

    os.environ["RAGMEMORY_LLM_OPENCODE_GO_MODEL"] = "custom-model"
    os.environ["RAGMEMORY_LLM_OPENCODE_GO_API_STYLE"] = "anthropic_messages"
    options = LLMProviderOptions.from_env("opencode_go", "fallback-model")
    client = LLMProviderClient(options)
    assert client.complete_chat([], max_tokens=1) is None
    assert client.last_error == "unsupported api_style: anthropic_messages"

    os.environ["RAGMEMORY_STRUCTURED_MAX_CHARS"] = "12 # inline comment"
    os.environ["RAGMEMORY_STRUCTURED_MAX_TOKENS"] = "34 ; inline comment"
    extraction_options = StructuredExtractionOptions.from_env()
    assert extraction_options.max_chars == 12
    assert extraction_options.max_tokens == 34

    os.environ["RAGMEMORY_COMPACT_MIN_CHARS"] = "56 # inline comment"
    os.environ["RAGMEMORY_COMPACT_MAX_TOKENS"] = "78 ; inline comment"
    os.environ["RAGMEMORY_COMPACT_TARGET_RATIO"] = "0.25 # inline comment"
    compaction_options = MessageCompactionOptions.from_env()
    assert compaction_options.min_chars == 56
    assert compaction_options.max_tokens == 78
    assert compaction_options.target_ratio == 0.25
    extractor = StructuredMemoryExtractor(
        options=LLMProviderOptions(
            provider="test",
            api_key=None,
            base_url="",
            model="test-model",
            api_style="openai_chat",
        ),
        extraction_options=extraction_options,
    )
    assert "Message:\nhello world!" in extractor._build_prompt("user", "hello world! extra")
finally:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

print("LLM provider config test passed.")
