import os
from typing import Union

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import tiktoken
from neo4j import GraphDatabase


load_dotenv()

neo4j_driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI", "neo4j://localhost:7687"),
    auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "your_password")),
    notifications_min_severity="OFF"
)


def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks


def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def tool_choice(messages, model="gpt-4o", temperature=0, tools=[], config={}):
    
    global _openai_client
    response = _openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=tools or None,
        **config,
    )
    return response.choices[0].message.tool_calls


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
