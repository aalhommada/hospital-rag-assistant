"""
Turning text into vectors.

An embedding model reads a piece of text and returns a fixed-length list of
numbers — 384 of them, for the default model here. Texts that mean similar
things get similar numbers, which is what lets the database find "you may not
eat for six hours beforehand" when someone asks "can I have breakfast first?".

Two details matter more than the model choice:

1. **Queries and passages are embedded differently.** A question and the
   paragraph that answers it do not look alike, so retrieval models are trained
   with an instruction prefix on the query side. Skipping this quietly costs
   several points of recall, and nothing in the system will tell you.

2. **The provider is swappable, the vector width is not.** Every provider here
   returns 384 numbers so they can share one database column. A model with a
   different width needs a migration — see README, "Changing the embedding
   model".

The default provider runs on your machine, so this whole project needs exactly
one API key: the Claude one.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

from django.conf import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """What the rest of the codebase is allowed to assume about a provider."""

    dimensions: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed text that will be stored and searched."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question someone just typed."""
        ...


class LocalEmbeddings:
    """
    Runs the model on this machine through ONNX — no API key, no network call
    after the first download, no per-query cost.

    The model is BAAI/bge-small-en-v1.5: 384 dimensions, about 130 MB, and
    close enough to the hosted models that you will not notice the difference
    on a corpus this size. `query_embed` applies the instruction prefix that
    the bge family expects; `passage_embed` does not.
    """

    dimensions = 384

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        logger.info("loading local embedding model %s", model_name)
        self.model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed([text]))).tolist()


class VoyageEmbeddings:
    """
    Voyage AI, the embedding provider Anthropic recommends alongside Claude.

    `voyage-3-lite` is used because it returns 384 dimensions, matching the
    database column. The larger Voyage models are wider and need a migration.
    """

    dimensions = 384
    _endpoint = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model_name: str = "voyage-3-lite") -> None:
        if not api_key:
            raise ValueError("EMBEDDING_PROVIDER=voyage needs VOYAGE_API_KEY to be set")
        self.api_key = api_key
        self.model_name = model_name

    def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        import httpx2 as httpx

        response = httpx.post(
            self._endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": texts, "input_type": input_type},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        # Voyage does not promise result order, so sort by the echoed index.
        return [item["embedding"] for item in sorted(payload["data"], key=lambda d: d["index"])]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts, "document") if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "query")[0]


class OpenAIEmbeddings:
    """
    OpenAI embeddings, included to show the interface is not Claude-specific.

    `text-embedding-3-small` is natively 1536-dimensional but supports
    shortening at request time, so `dimensions=384` keeps it compatible with
    the same database column.
    """

    dimensions = 384
    _endpoint = "https://api.openai.com/v1/embeddings"

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small") -> None:
        if not api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY to be set")
        self.api_key = api_key
        self.model_name = model_name

    def _call(self, texts: list[str]) -> list[list[float]]:
        import httpx2 as httpx

        response = httpx.post(
            self._endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": texts, "dimensions": self.dimensions},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in sorted(payload["data"], key=lambda d: d["index"])]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        # OpenAI uses the same encoder for both sides — no instruction prefix.
        return self._call([text])[0]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """
    Build the configured provider once per process.

    Cached because the local model takes a second or two to load and would
    otherwise be reloaded on every single request.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "local":
        instance: EmbeddingProvider = LocalEmbeddings(settings.EMBEDDING_MODEL)
    elif provider == "voyage":
        instance = VoyageEmbeddings(settings.VOYAGE_API_KEY, settings.EMBEDDING_MODEL)
    elif provider == "openai":
        instance = OpenAIEmbeddings(settings.OPENAI_API_KEY, settings.EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r}: expected local, voyage, or openai")

    from knowledge.models import EMBEDDING_DIMENSIONS

    if instance.dimensions != EMBEDDING_DIMENSIONS:
        # Failing here beats writing wrongly sized vectors into the database and
        # discovering it as a confusing Postgres error days later.
        raise ValueError(
            f"{provider} returns {instance.dimensions}-dimensional vectors but the database "
            f"column holds {EMBEDDING_DIMENSIONS}. See README, 'Changing the embedding model'."
        )
    return instance
