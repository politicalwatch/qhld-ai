"""Port for a decision model: a model that reads text and answers typed questions
about it with probabilities, instead of generating text.

Backend-agnostic: a hosted API (TypeSafe's Jev, OpenAI's Decisions) or a local
server (Ollama) implements it. One call carries one piece of text — the ``state``
— and any number of NAMED questions about it, answered independently; providers
batch them in a single request, so a task that needs several verdicts on the same
text pays one round trip.

Two question shapes, which are the two every provider supports:

- ``Choice`` — pick one of several named options. The answer carries the
  probability of every option, so a caller can apply its own threshold rather
  than trust the provider's pick.
- ``Predicate`` — a yes/no question. The answer IS the probability of yes.

A provider that cannot answer (HTTP error, timeout, the model declining a
question) raises ``DecisionUnavailable``, so each caller decides whether its task
fails open or closed.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Choice:
    """Pick one of ``options`` (value -> description). An empty description is
    allowed; the value alone then has to carry the meaning."""
    instructions: str
    options: dict[str, str]


@dataclass(frozen=True)
class Predicate:
    """A yes/no question; answered with the probability of yes."""
    instructions: str


Question = Choice | Predicate


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None

    def probability(self, value: str) -> float:
        """The probability of one option, 0.0 when the provider did not report it."""
        return self.probabilities.get(value, 0.0)


@dataclass(frozen=True)
class PredicateAnswer:
    probability: float


Answer = ChoiceAnswer | PredicateAnswer


class DecisionUnavailable(Exception):
    """The decision model gave no usable answer: the request failed, timed out, or
    the model declined a question."""


class DecisionModelPort(Protocol):
    model: str

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer every question about ``state``, keyed by the caller's names.
        Raises ``DecisionUnavailable`` when any answer cannot be had."""
        ...
