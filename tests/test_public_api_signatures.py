"""
Verify the documented public API signatures.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_public_api_signatures.py
"""
import inspect

from ragmemory import (
    AddMessageResult,
    ContextBundle,
    ForgetPreview,
    ForgetResult,
    MemoryStore,
    SearchResult,
    format_for_prompt,
)


def names(callable_obj):
    return list(inspect.signature(callable_obj).parameters)


assert names(MemoryStore.add_message) == ["self", "role", "text", "extract_structured"]
assert names(MemoryStore.build_context_bundle) == ["self", "user_message"]
assert names(MemoryStore.search) == ["self", "query", "top_k"]
assert names(MemoryStore.forget) == [
    "self",
    "message_ids",
    "before",
    "query",
    "confirm",
    "sample_limit",
]
assert names(format_for_prompt) == ["bundle"]

for exported in (
    AddMessageResult,
    ContextBundle,
    ForgetPreview,
    ForgetResult,
    MemoryStore,
    SearchResult,
):
    assert exported.__name__

print("Public API signature test passed.")
