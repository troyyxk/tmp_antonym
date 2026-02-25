import json
import os
import urllib.error
import urllib.request
from typing import List, Protocol, Sequence


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class SentenceTransformerEmbedder:
    """
    Local embedding backend for BGE and other sentence-transformers models.
    Example model names:
    - BAAI/bge-small-en-v1.5
    - BAAI/bge-m3
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Ensure the model is available locally or network access is enabled."
            ) from exc

        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [list(vector) for vector in vectors]


class OpenAICompatibleEmbedder:
    """
    OpenAI-compatible embedding backend.
    Works with OpenAI and providers exposing POST {base_url}/embeddings.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set and no api_key was provided.")

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": list(texts)}
        url = f"{self.base_url}/embeddings"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding request failed: HTTP {exc.code}, body={body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        parsed = json.loads(body)
        if "data" not in parsed:
            raise RuntimeError(f"Unexpected embedding response: {parsed}")

        data = sorted(parsed["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]
