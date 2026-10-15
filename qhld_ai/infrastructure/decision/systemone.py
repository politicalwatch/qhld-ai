"""Decision models behind the System One request format: TypeSafe's hosted Jev
models and a local Ollama, both through TypeSafe's SDK (the ``typesafe`` extra).

Ollama serves decision models on the same ``POST /v1/systemone`` route and body
TypeSafe defines, which is why one adapter covers both and only the base URL and
the key differ. The SDK refuses an empty key, so Ollama, which ignores the header,
is sent a placeholder. A ``Predicate`` is what the format calls a ``noul``; a
``Choice`` sends its options as ``criteria``.

The SDK retries 429 and 5xx twice by default and waits up to 10 s; both are
overridden — no retries, our timeout — because the callers of a decision model are
on the request path and must decide quickly, the same policy as the hosted
rerankers. Request compression is off: it depends on whether zstd happens to be
installed, and nothing says Ollama accepts it.

Any failure — HTTP error, timeout, a missing answer or one of a type we did not
ask for — raises ``DecisionUnavailable``. The tokens the provider reports are put
on the call's trace (see ``tracing``). The SDK client is created lazily on first
use, so importing this module does not need the SDK.
"""

from langsmith import traceable

from qhld_ai.domain.ports.decision_model import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionModelPort,
    DecisionUnavailable,
    PredicateAnswer,
    Question,
)
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register
from .tracing import record_usage

_TYPESAFE_URL = "https://api.typesafe.ai"
_OLLAMA_KEY = "ollama"


class SystemOneDecisionModel(DecisionModelPort):
    def __init__(self, model: str, base_url: str, api_key: str = "",
                 timeout: float = 3.0, provider: str = "typesafe", transport=None):
        self.model = model
        self.provider = provider
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            self._client = TypeSafeClient(
                api_key=self._api_key or _OLLAMA_KEY, model=self.model,
                base_url=self._base_url, timeout=self._timeout,
                retry=RetryPolicy(max_retries=0), compression=False,
                transport=self._transport)
        return self._client

    def _questions(self, questions: dict[str, Question]) -> dict:
        from typesafe_sdk import Choice as SdkChoice, Noul

        return {name: SdkChoice(instructions=q.instructions, criteria=dict(q.options))
                if isinstance(q, Choice) else Noul(instructions=q.instructions)
                for name, q in questions.items()}

    @traceable(name="decision_model", run_type="llm")
    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        from typesafe_sdk import TypeSafeError

        try:
            response = self.client.system_one(state=state,
                                              questions=self._questions(questions))
        except TypeSafeError as exc:
            raise DecisionUnavailable(f"{type(exc).__name__}: {exc}") from exc
        record_usage(self.provider, self.model, response.usage.input_tokens,
                     response.usage.output_tokens)
        answers = {}
        for name, question in questions.items():
            if isinstance(question, Choice) and name in response.choices:
                answer = response.choices[name]
                answers[name] = ChoiceAnswer(
                    choice=answer.choice, probabilities=dict(answer.probabilities),
                    confidence=answer.confidence)
            elif not isinstance(question, Choice) and name in response.nouls:
                answers[name] = PredicateAnswer(probability=response.nouls[name].noul)
            else:
                raise DecisionUnavailable(f"no usable answer to {name!r}")
        return answers


@_register("typesafe")
def create_typesafe(settings: Settings) -> SystemOneDecisionModel:
    return SystemOneDecisionModel(
        settings.decision_model, _TYPESAFE_URL, settings.typesafe_api_key,
        settings.decision_timeout, provider="typesafe")


@_register("ollama")
def create_ollama(settings: Settings) -> SystemOneDecisionModel:
    return SystemOneDecisionModel(
        settings.decision_model, settings.ollama_base_url,
        timeout=settings.decision_timeout, provider="ollama")
