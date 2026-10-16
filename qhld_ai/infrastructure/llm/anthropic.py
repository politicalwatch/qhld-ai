from langchain_anthropic import ChatAnthropic

from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register

# Reasoning-tier models (Claude 5 family + Opus 4.7/4.8) removed the sampling
# parameters: sending `temperature`/`top_p`/`top_k` returns a 400
# ("temperature is deprecated for this model"). Determinism is steered by prompt
# and effort instead. Haiku 4.5, Sonnet 4.6, Opus 4.6 and older still accept them.
# Matched by prefix so point releases (claude-haiku-5-5, claude-opus-5-5,
# claude-fable-5-1, ...) are covered without a new entry each time.
_REASONING_FAMILIES = (
    "claude-haiku-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-mythos-5",
)


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_FAMILIES)


def _reasoning_kwargs(effort: str) -> dict:
    """Map the provider-neutral effort setting onto Claude's thinking + effort.

    Thinking is always set explicitly. Left unset, langchain's structured output
    forces the tool call, the API then turns thinking off, and every effort level
    behaves like "none" (measured 2026-10-16 on claude-haiku-5-5: zero reasoning
    tokens at "low"). "none" disables thinking and pins effort to "low", the
    configuration that was measured; it is rejected by models that cannot
    disable thinking (Opus 5.5, Sonnet 5.5, Fable).
    """
    if not effort:
        return {}
    if effort == "none":
        return {"thinking": {"type": "disabled"}, "reasoning_effort": "low"}
    return {"thinking": {"type": "adaptive"}, "reasoning_effort": effort}


@_register("anthropic")
def create(settings: Settings) -> ChatAnthropic:
    kwargs = {"model": settings.llm_model, "api_key": settings.anthropic_api_key}
    if _is_reasoning_model(settings.llm_model):
        kwargs.update(_reasoning_kwargs(settings.llm_reasoning_effort))
    elif settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatAnthropic(**kwargs)
