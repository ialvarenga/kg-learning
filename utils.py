import os
from typing import Union

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

_openai_client: OpenAI | None = None


def _get_openai_client(
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> OpenAI:
    """Return a (cached) OpenAI-compatible client pointed at Ollama."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(base_url=base_url, api_key=api_key)
    return _openai_client


def chat(
    messages: list[dict],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> str:
    """Send *messages* to the Ollama-served model and return the reply text."""
    client = _get_openai_client(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


# ── Embeddings ────────────────────────────────────────────────────────────────

_embed_model: SentenceTransformer | None = None
_embed_model_name: str | None = None


def embed(
    texts: Union[str, list[str]],
    model_name: str = "all-MiniLM-L6-v2",
):
    """Encode *texts* with a SentenceTransformer and return the embeddings.

    A single string returns a 1-D array; a list returns a 2-D array.
    The model is cached between calls as long as *model_name* stays the same.
    """
    global _embed_model, _embed_model_name
    if _embed_model is None or _embed_model_name != model_name:
        _embed_model = SentenceTransformer(model_name)
        _embed_model_name = model_name
    return _embed_model.encode(texts)


# ── Neo4j ─────────────────────────────────────────────────────────────────────


def init_neo4j_driver(
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
):
    """Initialise a Neo4j driver, verify connectivity, and return it.

    Falls back to the environment variables ``NEO4J_URI``, ``NEO4J_USERNAME``,
    and ``NEO4J_PASSWORD`` when the corresponding argument is *None*.
    """
    uri = uri or os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = password or os.environ.get("NEO4J_PASSWORD", "your_password")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    return driver
