from langchain_openai import ChatOpenAI

from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


# Reasoning models reject any temperature but the default (a 400) unless the
# effort is exactly "none", and so do reasoning models run at their default
# effort. langchain strips temperature for these itself, but it recognises the
# family by name ("gpt-5*", not "chat"), and the list lags OpenAI: langchain
# 1.6.4 still sends temperature=0.0 to gpt-6-luna, which fails every call. So
# the adapter enforces the rule for the families listed here, the way the
# Anthropic adapter lists its own. A new family goes here before it is used.
#
# What it means in practice: "no reasoning" and "temperature actually applied"
# are the same setting, and every other effort level is non-deterministic by
# construction.
_REASONING_FAMILIES = ("gpt-5", "gpt-6")


def _sends_temperature(model: str, effort: str | None) -> bool:
    model = model.lower()
    if model.startswith(_REASONING_FAMILIES) and "chat" not in model:
        return effort == "none"
    return True


@_register("openai")
def create(settings: Settings) -> ChatOpenAI:
    kwargs = {"model": settings.llm_model, "api_key": settings.openai_api_key}
    if settings.llm_temperature is not None and _sends_temperature(
            settings.llm_model, settings.llm_reasoning_effort):
        kwargs["temperature"] = settings.llm_temperature
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    return ChatOpenAI(**kwargs)
