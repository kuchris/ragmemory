"""
Verify embedding provider config without loading a model.

Run:
    uv run python tests/test_embedding_options.py
"""
import os

from ragmemory.embeddings import EmbeddingOptions


OLD_ENV = {
    name: os.environ.get(name)
    for name in (
        "RAGMEMORY_EMBEDDING_PROVIDER",
        "RAGMEMORY_EMBEDDING_MODEL",
        "RAGMEMORY_EMBEDDING_DEVICE",
        "RAGMEMORY_EMBEDDING_NORMALIZE",
    )
}


def restore_env():
    for name, value in OLD_ENV.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


try:
    for name in OLD_ENV:
        os.environ.pop(name, None)

    default_options = EmbeddingOptions.from_env()
    assert default_options.label == "chroma_default"
    assert default_options.collection_name("chat_memory") == "chat_memory"

    os.environ["RAGMEMORY_EMBEDDING_PROVIDER"] = "sentence_transformers"
    os.environ["RAGMEMORY_EMBEDDING_MODEL"] = "BAAI/bge-m3"
    os.environ["RAGMEMORY_EMBEDDING_NORMALIZE"] = "true"
    m3_options = EmbeddingOptions.from_env()
    assert m3_options.label == "sentence_transformers:BAAI/bge-m3"
    assert (
        m3_options.collection_name("chat_memory")
        == "chat_memory_sentence_transformers_baai_bge_m3_norm"
    )
    assert (
        m3_options.collection_name("structured_memory")
        == "structured_memory_sentence_transformers_baai_bge_m3_norm"
    )
finally:
    restore_env()

print("Embedding options test passed.")
