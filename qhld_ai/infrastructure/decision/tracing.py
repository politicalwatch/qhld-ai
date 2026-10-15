"""Token usage of a decision call, recorded on its LangSmith run.

Each adapter's ``decide`` is traced as an ``llm`` run, and this puts the provider,
the model and the tokens the provider reported on it — the same fields LangChain
records for a chat model — so LangSmith counts a decision call's tokens and, once
the model has a price entry there, its cost, beside the parse it may have saved.
With tracing off there is no current run and this does nothing.
"""

from langsmith.run_helpers import get_current_run_tree


def record_usage(provider: str, model: str, input_tokens: int | None,
                 output_tokens: int | None, cached_tokens: int | None = None) -> None:
    run = get_current_run_tree()
    if run is None:
        return
    usage = {"input_tokens": input_tokens or 0, "output_tokens": output_tokens or 0}
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    if cached_tokens:
        usage["input_token_details"] = {"cache_read": cached_tokens}
    run.set(metadata={"ls_provider": provider, "ls_model_name": model},
            usage_metadata=usage)
