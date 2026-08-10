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
uv sync --extra dev
uv run pytest
```

**Use `--extra dev`, not a bare `uv sync`.** Everything provider- or feature-specific
lives in an optional extra (see `[project.optional-dependencies]`), and `uv sync` is
exact: it installs the core dependencies plus the `dev` dependency-group and *removes*
everything else. So a bare `uv sync` — or `uv run`, which syncs implicitly — deletes
`langchain-openai`, spaCy, `onnxruntime`, `av` and `num2words`, and the suite then fails
in a way that looks like a regression at HEAD and is not one. `dev` is the profile extra
that installs what the tests need without pulling torch.
