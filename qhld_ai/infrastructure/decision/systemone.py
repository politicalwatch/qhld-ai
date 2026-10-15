"""Decision models behind the System One request format: TypeSafe's hosted Jev
models and a local Ollama.

Ollama serves decision models on the same ``POST /v1/systemone`` route and body
TypeSafe defines, which is why one adapter covers both and only the base URL and
the authentication differ. The request carries the text as ``state`` and the
questions as an object keyed by name; a ``Predicate`` is what the format calls a
``noul`` (answered with the probability of yes, under the key ``noul``), a
``Choice`` sends its options as ``criteria``.

Any failure — HTTP error, timeout, an answer of a type we did not ask for —
raises ``DecisionUnavailable``. No retry layer: the callers of a decision model
are on the request path and must decide quickly, the same policy as the hosted
rerankers. The ``httpx.Client`` is created lazily on first use.
"""

import httpx

from qhld_ai.domain.ports.decision_model import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionModelPort,
    DecisionUnavailable,
    Predicate,
    PredicateAnswer,
    Question,
)
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register

_TYPESAFE_URL = "https://api.typesafe.ai"
_ROUTE = "/v1/systemone"


def _encode(question: Question) -> dict:
    if isinstance(question, Choice):
        return {"type": "choice", "instructions": question.instructions,
                "criteria": dict(question.options)}
    return {"type": "noul", "instructions": question.instructions}


def _decode(question: Question, answer: dict) -> Answer:
    if isinstance(question, Choice) and answer.get("type") == "choice":
        return ChoiceAnswer(
            choice=answer["choice"],
            probabilities={k: float(v) for k, v in answer.get("probabilities", {}).items()},
            confidence=answer.get("confidence"))
    if isinstance(question, Predicate) and answer.get("type") == "noul":
        return PredicateAnswer(probability=float(answer["noul"]))
    raise DecisionUnavailable(f"unexpected answer of type {answer.get('type')!r}")


class SystemOneDecisionModel(DecisionModelPort):
    def __init__(self, model: str, base_url: str, api_key: str = "",
                 timeout: float = 3.0):
        self.model = model
        self._url = base_url.rstrip("/") + _ROUTE
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._client = httpx.Client(headers=headers, timeout=self._timeout)
        return self._client

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        body = {"model": self.model, "state": state,
                "questions": {name: _encode(q) for name, q in questions.items()}}
        try:
            response = self.client.post(self._url, json=body)
            response.raise_for_status()
            answers = response.json()["answers"]
            return {name: _decode(q, answers[name]) for name, q in questions.items()}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise DecisionUnavailable(f"{type(exc).__name__}: {exc}") from exc


@_register("typesafe")
def create_typesafe(settings: Settings) -> SystemOneDecisionModel:
    return SystemOneDecisionModel(
        settings.decision_model, _TYPESAFE_URL, settings.typesafe_api_key,
        settings.decision_timeout)


@_register("ollama")
def create_ollama(settings: Settings) -> SystemOneDecisionModel:
    return SystemOneDecisionModel(
        settings.decision_model, settings.ollama_base_url,
        timeout=settings.decision_timeout)
