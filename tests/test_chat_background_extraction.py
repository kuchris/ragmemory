"""
Verify scripts/chat.py uses bounded background extraction.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_chat_background_extraction.py
"""
import ast
from pathlib import Path

CHAT_PATH = Path("scripts/chat.py")
tree = ast.parse(CHAT_PATH.read_text(encoding="utf-8"))

add_calls = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "add_message"
]
run_calls = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "run_pending_extractions"
]

user_call = next(
    call for call in add_calls
    if len(call.args) >= 2
    and isinstance(call.args[0], ast.Constant)
    and call.args[0].value == "user"
)
assistant_call = next(
    call for call in add_calls
    if len(call.args) >= 2
    and isinstance(call.args[0], ast.Constant)
    and call.args[0].value == "assistant"
)

user_mode = next(keyword.value for keyword in user_call.keywords if keyword.arg == "extract_structured")
assistant_mode = next(keyword.value for keyword in assistant_call.keywords if keyword.arg == "extract_structured")

assert isinstance(user_mode, ast.Constant)
assert user_mode.value == "background"
assert isinstance(assistant_mode, ast.Constant)
assert assistant_mode.value is False

assert len(run_calls) == 1
limit = next(keyword.value for keyword in run_calls[0].keywords if keyword.arg == "limit")
assert isinstance(limit, ast.Constant)
assert limit.value == 1

print("Chat background extraction test passed.")
