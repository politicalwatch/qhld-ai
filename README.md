# qhld-ai

Semantic search and NLP package for the QHLD platform (package `qhld-ai`,
import `qhld_ai`), extracted from `qhld-engine` so both the engine (indexing,
mention tagging) and `qhld-backend` (search endpoints) can use it.

It provides:

- **Query understanding** — LLM-based parsing of natural-language queries into
  structured filters plus a residual semantic query, with entity resolution
  against the deputies/persons catalogs and parliamentary-group aliases.
- **Retrieval** — dense embeddings, hybrid dense+BM25 search with RRF/DBSF
  fusion, and optional cross-encoder reranking over a Qdrant vector store.
- **NLP** — spaCy NER for person mentions, fuzzy person/deputy resolution,
  language detection, and passage chunking.
- **Provider adapters** — LLM and embedding factories for Anthropic, OpenAI,
  Ollama, Google and Mistral behind `domain/ports` protocols.

Configuration is env-driven via a Pydantic `Settings`
(`qhld_ai/infrastructure/config/settings.py`); data models come from
[`qhld-data`](https://github.com/politicalwatch/qhld-data).

## Development

```bash
uv sync
uv run pytest
```
