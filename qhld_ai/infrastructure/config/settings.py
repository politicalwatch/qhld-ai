from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # Logging (field name matches env LOGLEVEL exactly)
    loglevel: str = "INFO"

    # LLM providers.
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    mistral_api_key: str = ""
    ollama_base_url: str = "http://host.docker.internal:11434"
    vmlx_base_url: str = "http://host.docker.internal:8080"

    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # Qdrant vector store (semantic speech search). Host defaults to the
    # docker-compose service name; ":memory:" selects qdrant-client's in-process
    # mode (used by the Docker-free tests).
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False
    # Empty -> the index/search services derive a per-model collection name
    # (speeches__<provider>__<model>__<dim>); set to force a fixed name.
    qdrant_collection: str = ""

    # Speech chunking (passage granularity for embeddings). Char-budgeted rather
    # than token-based so it stays provider/tokenizer-agnostic.
    speech_chunk_chars: int = 1200
    speech_chunk_overlap: int = 150

    # Query understanding: parse a NL query into structured
    # filters + a residual semantic query. Decoupled from the main llm_* so the
    # parser can use a different model than any future answer-synthesis; empty
    # provider/model fall back to llm_provider/llm_model.
    query_parser_provider: str = "llm"
    query_parser_llm_provider: str = ""
    query_parser_llm_model: str = ""

    # Cross-encoder reranker. "noop" leaves bi-encoder order
    # untouched (the clean baseline); any other provider over-fetches
    # reranker_top_n passages and reorders them on (query, passage) relevance.
    reranker_provider: str = "noop"
    reranker_model: str = ""
    reranker_top_n: int = 50
    # Rerankers served over HTTP by a LOCAL server: the model stays loaded in
    # its own process instead of being loaded in-process by every caller. For
    # "tei" the base URL is the server root (the adapter calls its fixed
    # /rerank route); for "rerank_api" it is the FULL rerank endpoint URL,
    # because servers mount it at different paths (e.g. a local vMLX serves
    # http://127.0.0.1:11438/v1/rerank). Local servers are unauthenticated —
    # hosted vendors get dedicated providers with their own keys below.
    reranker_base_url: str = ""
    # Hosted-vendor providers ("jina", "cohere", "voyage", "novita")
    # authenticate with per-vendor keys so several vendors can coexist in one
    # .env. The cohere and voyage SDKs also read their own env vars
    # (COHERE_API_KEY / VOYAGE_API_KEY) when the setting is empty.
    jina_api_key: str = ""
    cohere_api_key: str = ""
    voyage_api_key: str = ""
    novita_api_key: str = ""
    # Relevance floor on the reranked score: results below it are dropped, so an
    # off-domain or nonsensical query returns nothing rather than the top-k
    # least-irrelevant passages (bi-encoder cosine / RRF scores don't separate
    # in-domain from off-domain, but the cross-encoder does). Only applied on the
    # reranked path, and only for topical queries — natural search skips it for
    # pure-entity and pure-filter queries, where valid brief-mention hits
    # legitimately score as low as junk. 0.0 disables it, keeping the bi-encoder
    # baseline untouched. Calibrated to the reranker model's raw scores —
    # re-check if the model changes.
    reranker_score_floor: float = 0.0

    # Sparse lexical embeddings (hybrid retrieval). "none" keeps pure dense
    # retrieval — existing collections and search behavior untouched. When set
    # (e.g. "bm25"), indexing writes a second, lexical vector per passage and
    # search fuses the dense and lexical rankings, which keeps literal tokens
    # (names, road codes, law numbers) from being blurred by the dense embedding.
    sparse_provider: str = "none"
    sparse_model: str = "Qdrant/bm25"
    # Stemmer/stopword language for BM25 tokenization; it must match between
    # indexing and querying. The corpus is multilingual, but "spanish" covers
    # the dominant blocks and proper names are barely affected by stemming.
    sparse_language: str = "spanish"
    # Hybrid fusion tuning: candidates fetched per branch (dense / lexical)
    # before fusion, and the fusion algorithm ("rrf" or "dbsf").
    hybrid_prefetch_limit: int = 50
    hybrid_fusion: str = "rrf"

    # Mention extraction (index-time NER → resolved deputies on Speech.mentions).
    # PER spans are found by spaCy over the Spanish text block, then fuzzy-matched
    # against the deputies catalog; token_set_ratio scores subset matches high, so
    # a high threshold stays both forgiving (surname-only) and precise.
    ner_provider: str = "spacy"
    ner_model: str = "es_core_news_lg"
    # Seed the NER pipeline with a gazetteer of distinctive deputy surnames so it also
    # tags the uncommon/compound ones (Catalan/Basque names, hyphenated compounds) the
    # base model misses. Off => base model only.
    ner_gazetteer: bool = True
    # Precision gate on non-person entity spans: keep a span only if it holds a proper
    # noun and no verb, so clauses and discourse fragments the model mislabels as
    # entities ("Por tanto", "Llama la atención…") are dropped at the source. Off =>
    # every non-PER span is emitted (the raw model output).
    ner_entity_pos_gate: bool = True
    # Tag the name in a role apposition the model missed or mislabelled ("El ministro
    # Albares" comes back as MISC), when the catalog records somebody of that surname
    # holding that office. Reaches the common in-vocabulary surnames the gazetteer must
    # leave alone, because the role word is what says this occurrence names a person; the
    # claimed span also stops being a named entity. Off => those mentions stay untagged.
    ner_role_apposition: bool = True
    # Tag the name a courtesy form introduces when the model missed it ("señor Cuerpo"
    # comes back as ORG). Needs no catalog gate, unlike the apposition above: a role word
    # also appears where it names nobody ("Gracias, presidenta"), while a courtesy form is
    # followed by a name essentially always, and a surname from outside the catalog simply
    # resolves to nobody. Off => those mentions stay untagged.
    ner_courtesy_form: bool = True
    mention_match_threshold: int = 90
    # Let a gendered courtesy form decide a surname two catalog people share: "la señora
    # Muñoz" is the female Muñoz, so the ambiguity guard need not drop it. Evidence is
    # pooled per speech, so one such form settles every bare occurrence of that surname in
    # it. Off => shared surnames are dropped unless a fuller form disambiguates them.
    mention_gender_gate: bool = True
    # Attach a surname that stayed ambiguous to the one tied person the same speech names
    # elsewhere by a fuller form. It never introduces a person the speech does not
    # already name — it only stops their remaining occurrences going uncounted. Off =>
    # those occurrences are dropped.
    mention_speech_coreference: bool = True
    # Let the office a speech names somebody by decide a surname two catalog people share:
    # "el presidente Sánchez" is the prime minister, not a deputy who happens to be called
    # Sánchez. Pooled per speech like the courtesy form, and it only ever narrows a tie —
    # when no candidate holds the office, the guard decides as before. Off => a surname
    # named by title is dropped like any other ambiguous one.
    mention_role_apposition: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
