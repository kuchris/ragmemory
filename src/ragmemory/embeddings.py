from __future__ import annotations

import os
import re
from dataclasses import dataclass

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


DEFAULT_EMBEDDING_PROVIDER = "chroma_default"
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "BAAI/bge-small-en-v1.5"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "default"


@dataclass(frozen=True)
class EmbeddingOptions:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    model: str = ""
    device: str = "cpu"
    normalize_embeddings: bool = True

    @classmethod
    def from_env(cls) -> "EmbeddingOptions":
        provider = os.environ.get(
            "RAGMEMORY_EMBEDDING_PROVIDER",
            DEFAULT_EMBEDDING_PROVIDER,
        ).strip()
        model = os.environ.get("RAGMEMORY_EMBEDDING_MODEL", "").strip()
        device = os.environ.get("RAGMEMORY_EMBEDDING_DEVICE", "cpu").strip() or "cpu"
        return cls(
            provider=provider,
            model=model,
            device=device,
            normalize_embeddings=_env_bool("RAGMEMORY_EMBEDDING_NORMALIZE", True),
        )

    @property
    def label(self) -> str:
        provider = self.provider.strip().lower()
        if provider in {"", "default", "chroma", "chroma_default"}:
            return "chroma_default"
        model = self.model or DEFAULT_SENTENCE_TRANSFORMERS_MODEL
        return f"{provider}:{model}"

    def collection_name(self, base_name: str) -> str:
        provider = self.provider.strip().lower()
        if provider in {"", "default", "chroma", "chroma_default"}:
            return base_name
        model = self.model or DEFAULT_SENTENCE_TRANSFORMERS_MODEL
        suffix = _slug(
            f"{provider}_{model}_{'norm' if self.normalize_embeddings else 'raw'}"
        )
        return f"{base_name}_{suffix}"


def build_embedding_function(options: EmbeddingOptions):
    provider = options.provider.strip().lower()
    if provider in {"", "default", "chroma", "chroma_default"}:
        return DefaultEmbeddingFunction()
    if provider in {"sentence_transformers", "sentence-transformers", "st"}:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        return SentenceTransformerEmbeddingFunction(
            model_name=options.model or DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
            device=options.device,
            normalize_embeddings=options.normalize_embeddings,
        )
    raise ValueError(
        "unknown embedding provider "
        f"{options.provider!r}; use chroma_default or sentence_transformers"
    )
