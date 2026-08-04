from langchain_openai import ChatOpenAI

from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register


# Reasoning effort and temperature are entangled on the gpt-5 family, and the
# entanglement is enforced by langchain rather than by us: for any gpt-5* model
# that is not a "chat" variant it DROPS temperature unless the effort is
# explicitly "none" (langchain_openai/chat_models/base.py, at field validation
# and again when building the Responses payload). It drops it silently — no
# error, no warning — so passing temperature=0.0 at any other effort is a no-op
# and the model samples at its own default.
#
# We deliberately do not re-implement that condition here: langchain already
# owns it, and a copy would drift on upgrade. It is pinned by a test instead, so
# an upgrade that changes the behaviour fails loudly. What it means in practice:
# "no reasoning" and "temperature actually applied" are the same setting, and
# every other effort level is non-deterministic by construction.
@_register("openai")
def create(settings: Settings) -> ChatOpenAI:
    kwargs = {"model": settings.llm_model, "api_key": settings.openai_api_key}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    return ChatOpenAI(**kwargs)
