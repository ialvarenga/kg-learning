import argparse
import asyncio
import logging
import os
from pathlib import Path
import yaml
import neo4j
from neo4j_graphrag.llm import OllamaLLM, OpenAILLM
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from graph_schema import get_graph_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password"

PROMPT_TEMPLATES: dict[str, str] = {
    "medical": "prompts/kg_extraction.yaml",
    "book": "prompts/kg_extraction_book.yaml",
}


def load_prompt_template(schema_type: str = "book") -> str:
    path = PROMPT_TEMPLATES.get(schema_type, "prompts/kg_extraction_book.yaml")
    with open(Path(__file__).parent / path, "r") as f:
        data = yaml.safe_load(f)
    return data["prompt_template"]


DEFAULTS = {
    "openai": {"model": "gpt-5.2", "embedding_model": "text-embedding-3-small"},
    "ollama": {"model": "qwen2.5:7b", "embedding_model": "nomic-embed-text"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider to use: 'openai' or 'ollama' (default: ollama)",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="LLM model name (default: gpt-4o for openai, llama3.2 for ollama)",
    )
    parser.add_argument(
        "-e", "--embedding-model",
        default=None,
        dest="embedding_model",
        help="Embedding model name (default: text-embedding-3-small for openai, nomic-embed-text for ollama)",
    )
    parser.add_argument(
        "-f", "--file",
        default="data/a-morte-de-ivan-ilitch.pdf",
        help="Path to the PDF file to process (default: data/a-morte-de-ivan-ilitch.pdf)",
    )
    parser.add_argument(
        "-s", "--schema",
        choices=["book", "medical"],
        default="book",
        help="Graph schema to use for entity extraction: 'book' or 'medical' (default: book)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        dest="chunk_size",
        help="Text splitter chunk size in characters (default: 4000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        dest="chunk_overlap",
        help="Text splitter chunk overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--no-approximate-split",
        action="store_false",
        dest="approximate",
        default=True,
        help="Disable approximate splitting (split exactly at chunk_size boundary)",
    )
    return parser.parse_args()


def get_neo4j_driver() -> neo4j.Driver:
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def get_llm_and_embedder(provider: str, model: str | None = None, embedding_model: str | None = None):
    model = model or DEFAULTS[provider]["model"]
    embedding_model = embedding_model or DEFAULTS[provider]["embedding_model"]
    if provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        llm = OpenAILLM(
            model_name=model,
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        embedder = OpenAIEmbeddings(model=embedding_model)
    else:
        llm = OllamaLLM(
            model_name=model,
            model_params={
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 4096}
            }
        )
        embedder = OllamaEmbeddings(model=embedding_model)
    return llm, embedder


def build_kg_pipeline(
    llm,
    embedder,
    driver: neo4j.Driver,
    schema_type: str = "book",
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    approximate: bool = True,
) -> SimpleKGPipeline:
    node_labels, rel_types = get_graph_schema(schema_type)
    prompt_template = load_prompt_template(schema_type)
    text_splitter = FixedSizeSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        approximate=approximate,
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        neo4j_database="neo4j",
        embedder=embedder,
        from_pdf=True,
        entities=node_labels,
        relations=rel_types,
        prompt_template=prompt_template,
        text_splitter=text_splitter,
    )


def main():
    args = parse_args()

    model = args.model or DEFAULTS[args.provider]["model"]
    embedding_model = args.embedding_model or DEFAULTS[args.provider]["embedding_model"]

    logger.info("=== KG Pipeline Configuration ===")
    logger.info("  Provider:        %s", args.provider)
    logger.info("  LLM model:       %s", model)
    logger.info("  Embedding model: %s", embedding_model)
    logger.info("  Schema:          %s", args.schema)
    logger.info("  Input file:      %s", args.file)
    logger.info("  Neo4j URI:       %s", NEO4J_URI)
    logger.info("  Chunk size:      %s", args.chunk_size)
    logger.info("  Chunk overlap:   %s", args.chunk_overlap)
    logger.info("  Approx split:    %s", args.approximate)
    logger.info("=================================")

    driver = get_neo4j_driver()
    llm, embedder = get_llm_and_embedder(args.provider, args.model, args.embedding_model)
    kg_builder = build_kg_pipeline(
        llm, embedder, driver,
        schema_type=args.schema,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        approximate=args.approximate,
    )

    logger.info("Starting pipeline run...")
    result = asyncio.run(kg_builder.run_async(file_path=args.file))
    logger.info("Pipeline finished.")
    print(result.result)


if __name__ == "__main__":
    main()