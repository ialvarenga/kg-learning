"""
Test script to inspect the raw response from the Ollama llama3.2 model,
helping diagnose the SchemaExtractionError: LLM response is not valid JSON.
"""
import asyncio
import json
import httpx
from neo4j_graphrag.llm import OllamaLLM


# ─── 1. Raw HTTP request (bypasses any neo4j_graphrag wrapper) ───────────────

def test_raw_ollama(prompt: str):
    """Call Ollama /api/generate directly and print the full response."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "format": "json",        # ask Ollama for JSON mode directly
        "options": {
            "temperature": 0
        }
    }

    print("=" * 60)
    print("[RAW HTTP] Sending request to Ollama...")
    print(f"Prompt:\n{prompt}\n")

    response = httpx.post(url, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "")

    print(f"[RAW HTTP] Full response object keys: {list(data.keys())}")
    print(f"[RAW HTTP] 'response' field content:\n{raw_text}\n")

    try:
        parsed = json.loads(raw_text)
        print("[RAW HTTP] ✅ Response is valid JSON:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print(f"[RAW HTTP] ❌ Response is NOT valid JSON. Error: {e}")

    return raw_text


# ─── 2. via OllamaLLM (same wrapper neo4j_graphrag uses) ─────────────────────

async def test_ollama_llm(prompt: str):
    """Call OllamaLLM the same way SimpleKGPipeline does."""
    llm = OllamaLLM(
        model_name="llama3.2",
        model_params={
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    )

    print("=" * 60)
    print("[OllamaLLM] Sending request via neo4j_graphrag OllamaLLM...")

    resp = await llm.ainvoke(prompt)

    # LLMResponse typically has a .content attribute
    content = getattr(resp, "content", resp)

    print(f"[OllamaLLM] Response type  : {type(resp)}")
    print(f"[OllamaLLM] .content type  : {type(content)}")
    print(f"[OllamaLLM] Raw content    :\n{content}\n")

    try:
        parsed = json.loads(content)
        print("[OllamaLLM] ✅ Content is valid JSON:")
        print(json.dumps(parsed, indent=2))
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[OllamaLLM] ❌ Content is NOT valid JSON. Error: {e}")

    return content


# ─── Prompts ──────────────────────────────────────────────────────────────────

# Minimal schema-extraction prompt (mirrors what SimpleKGPipeline sends)
SCHEMA_PROMPT = """\
You are a knowledge graph expert.
Extract entities and relationships from the text below.

Return ONLY a JSON object with this exact structure:
{
  "node_types": [
    {"label": "Person", "description": "A human individual"}
  ],
  "relationship_types": [
    {"label": "KNOWS", "description": "One person knows another"}
  ]
}

Text:
Marie Curie was a physicist who discovered radium. She worked at the University of Paris.
"""

# Simple sanity-check prompt
SIMPLE_PROMPT = 'Respond with valid JSON only. Return: {"status": "ok", "model": "llama3.2"}'


if __name__ == "__main__":
    print("\n### TEST 1 - Simple JSON prompt (raw HTTP) ###\n")
    test_raw_ollama(SIMPLE_PROMPT)

    print("\n### TEST 2 - Schema extraction prompt (raw HTTP) ###\n")
    test_raw_ollama(SCHEMA_PROMPT)

    print("\n### TEST 3 - Simple JSON prompt (OllamaLLM wrapper) ###\n")
    asyncio.run(test_ollama_llm(SIMPLE_PROMPT))

    print("\n### TEST 4 - Schema extraction prompt (OllamaLLM wrapper) ###\n")
    asyncio.run(test_ollama_llm(SCHEMA_PROMPT))
