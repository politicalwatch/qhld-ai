"""Decision models behind OpenAI's Decisions API.

``POST https://api.openai.com/v1/decisions`` with the text as ``input`` and the
questions as a LIST, each carrying its own ``name``. A ``Choice`` sends its options
as ``choices`` of ``{value, description}`` and comes back with ``probabilities``
as a list of ``{value, probability}``; a ``Predicate`` is a ``predicate`` question
answered with ``probability``. The API may answer a question with
``type: "refusal"`` instead, which is treated like any other failure.

Any failure raises ``DecisionUnavailable``; no retry layer, for the same reason
as the System One adapter. The tokens the response reports, cached ones included,
are put on the call's trace (see ``tracing``). Authenticates with
``settings.openai_api_key``. The ``httpx2.Client`` is created lazily on first use.
"""

import httpx2
from langsmith import traceable

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
from .tracing import record_usage

_DECISIONS_URL = "https://api.openai.com/v1/decisions"


def _encode(name: str, question: Question) -> dict:
    if isinstance(question, Choice):
        return {"type": "choice", "name": name, "instructions": question.instructions,
                "choices": [{"value": value, "description": description}
                            for value, description in question.options.items()]}
    return {"type": "predicate", "name": name, "instructions": question.instructions}


def _decode(question: Question, answer: dict) -> Answer:
    if isinstance(question, Choice) and answer.get("type") == "choice":
        return ChoiceAnswer(
            choice=answer["choice"],
            probabilities={row["value"]: float(row["probability"])
                           for row in answer.get("probabilities", [])},
            confidence=answer.get("confidence"))
    if isinstance(question, Predicate) and answer.get("type") == "predicate":
        return PredicateAnswer(probability=float(answer["probability"]))
    raise DecisionUnavailable(f"unexpected answer of type {answer.get('type')!r}")


class OpenAIDecisionModel(DecisionModelPort):
    def __init__(self, model: str, api_key: str, timeout: float = 3.0):
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = httpx2.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout)
        return self._client

    @traceable(name="decision_model", run_type="llm")
    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        body = {"model": self.model, "input": state,
                "questions": [_encode(name, q) for name, q in questions.items()]}
        try:
            response = self.client.post(_DECISIONS_URL, json=body)
            response.raise_for_status()
            payload = response.json()
            answers = {row.get("name"): row for row in payload["answers"]}
            decoded = {name: _decode(q, answers[name]) for name, q in questions.items()}
        except (httpx2.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise DecisionUnavailable(f"{type(exc).__name__}: {exc}") from exc
        usage = payload.get("usage") or {}
        record_usage("openai", self.model, usage.get("input_tokens"),
                     usage.get("output_tokens"),
                     (usage.get("input_tokens_details") or {}).get("cached_tokens"))
        return decoded


@_register("openai")
def create(settings: Settings) -> OpenAIDecisionModel:
    return OpenAIDecisionModel(
        settings.decision_model, settings.openai_api_key, settings.decision_timeout)
