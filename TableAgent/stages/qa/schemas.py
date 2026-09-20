"""The shapes every QA model call must return, and the helper that constrains them.

Each QA call asks for one JSON object, and the stage carries two repair prompts plus
regex salvage for the times a model does not deliver one. Constraining decoding removes
the failure instead of recovering from it: vLLM masks tokens that cannot continue a
valid document, so a malformed object stops being reachable.

The models here are the single source of truth. The same class both generates the schema
sent to the server and validates what comes back, so a field cannot be constrained one
way and parsed another. Validation stays deliberately lenient about *form* -- a missing
`category`, a comma-joined `depends_on` -- because guided decoding can be off,
unsupported, or degraded, and the salvage path still has to work. It is strict about
*meaning*: an unknown layer or an empty id is refused.

The call helper degrades rather than fails. A server that does not understand the
constraint answers 400; the call is retried once unconstrained, the client stops sending
it, and the parse-and-repair path still covers the output.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

SubtaskLayer = Literal["table_inspect", "inspect", "synthesis"]
SubtaskCategory = Literal["normal", "common_info"]


class PlannedSubtask(BaseModel):
    """One node of a QA plan, as the planner is allowed to write it."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    layer: SubtaskLayer = "inspect"
    category: SubtaskCategory = "normal"
    depends_on: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "description", mode="before")
    @classmethod
    def _as_trimmed_text(cls, value: Any) -> Any:
        return str(value).strip() if value is not None else value

    @field_validator("category", mode="before")
    @classmethod
    def _normalise_category(cls, value: Any) -> Any:
        return "normal" if value is None else str(value).strip().lower()

    @field_validator("depends_on", mode="before")
    @classmethod
    def _accept_joined_dependencies(cls, value: Any) -> list[str]:
        """Models write `depends_on` as a list, as a comma-joined string, or not at all."""
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("metadata", mode="before")
    @classmethod
    def _tolerate_absent_metadata(cls, value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @model_validator(mode="before")
    @classmethod
    def _promote_top_level_metadata(cls, data: Any) -> Any:
        """Accept `common_info_scope` and `target_names` written beside the other fields.

        The prompt shows them nested under `metadata`, and models routinely hoist them to
        the top level instead. Both spellings mean the same thing, so the flat one is
        folded in rather than dropped.
        """
        if not isinstance(data, dict):
            return data
        promoted = dict(data)
        metadata = dict(promoted.get("metadata") or {}) if isinstance(
            promoted.get("metadata"), dict
        ) else {}
        for key in ("common_info_scope", "target_names"):
            if promoted.get(key) and key not in metadata:
                metadata[key] = promoted[key]
        promoted["metadata"] = metadata
        return promoted


class QAPlan(BaseModel):
    """The planner's whole reply."""

    model_config = ConfigDict(extra="ignore")

    subtasks: list[PlannedSubtask] = Field(min_length=1)

    @model_validator(mode="after")
    def _common_info_needs_a_scope(self) -> "QAPlan":
        """A non-synthesis common-info subtask must say what it describes.

        Without a scope the common-info route has nothing to aim at and quietly
        describes the wrong object, so this is refused rather than defaulted.
        """
        for subtask in self.subtasks:
            if subtask.category != "common_info" or subtask.layer == "synthesis":
                continue
            scope = str(subtask.metadata.get("common_info_scope") or "").strip().lower()
            if scope not in {"workbook", "sheet", "table"}:
                raise ValueError(
                    f"Common-info subtask {subtask.id!r} requires "
                    "metadata.common_info_scope to be workbook, sheet, or table."
                )
            subtask.metadata["common_info_scope"] = scope
        return self


class GeneratedCode(BaseModel):
    """A code-generation reply: the reasoning, the cell, and what it does."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = Field(min_length=1)
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)

    @field_validator("reasoning", "code", "description", mode="before")
    @classmethod
    def _as_trimmed_text(cls, value: Any) -> Any:
        return str(value).strip() if value is not None else value


class _ScoredVerdict(BaseModel):
    """Shared leniency for the two reviewer replies.

    A reviewer that omits the score, or writes one outside the range, has still said the
    thing that matters -- whether it accepts. Rejecting the whole reply over the number
    would turn a usable verdict into a failed attempt, so the score is defaulted from
    `accepted` and clamped instead.
    """

    model_config = ConfigDict(extra="ignore")

    accepted: bool
    score: Optional[float] = None
    feedback: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _tolerate_unusable_score(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    @field_validator("feedback", mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @model_validator(mode="after")
    def _default_score_from_verdict(self):
        if self.score is None:
            self.score = 1.0 if self.accepted else 0.0
        return self

    def resolved_feedback(self) -> str:
        return self.feedback.strip() or ("Accepted." if self.accepted else "Rejected.")


class SubtaskReview(_ScoredVerdict):
    """A reviewer's verdict on one attempt."""


class FinalAnswerReview(_ScoredVerdict):
    """The final verdict, which separates a wrong answer from a weak derivation.

    `answer_wrong` decides whether the answer is discarded and the plan rewritten;
    `accepted` covers the attempt as a whole. Keeping them apart stops a complaint about
    the derivation from throwing away an answer the reviewer agrees with.
    """

    answer_wrong: Optional[bool] = None

    @model_validator(mode="after")
    def _default_answer_verdict(self) -> "FinalAnswerReview":
        if self.answer_wrong is None:
            # An older reviewer gives no separate verdict, so a rejection means what it
            # meant before the split: discard the answer.
            self.answer_wrong = not self.accepted
        return self


def json_schema(
    model: type[BaseModel],
    *,
    require: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Render `model` as a self-contained JSON Schema for a guided-decoding backend.

    Pydantic factors nested models into `$defs` and points at them with `$ref`. Backends
    differ in how well they resolve those, and a schema this small loses nothing by being
    inlined, so references are expanded and the presentation-only `title` keys dropped to
    keep the grammar the server compiles small.

    `require` tightens what generation must produce beyond what parsing will accept, as a
    map from a dotted path (`""` for the root, `subtasks.items` for an array's element)
    to the field names to mark required. The two differ on purpose: a field with a Python
    default parses fine when absent, but letting the model omit it means the default gets
    applied silently -- a synthesis subtask that never says `layer` would quietly become
    an inspect subtask. Generation is made to state it; parsing stays forgiving for the
    outputs that arrive unconstrained.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def expand(node: Any, seen: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, list):
            return [expand(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.rsplit("/", 1)[-1]
            if name in seen:
                # A self-referential model cannot be inlined; leave an unconstrained
                # object rather than recursing forever.
                return {"type": "object"}
            target = expand(definitions.get(name, {"type": "object"}), seen | {name})
            overrides = {key: value for key, value in node.items() if key != "$ref"}
            return {**target, **expand(overrides, seen)}
        return {key: expand(value, seen) for key, value in node.items() if key != "title"}

    expanded = expand(schema)

    for path, names in (require or {}).items():
        node = expanded
        for step in [piece for piece in path.split(".") if piece]:
            node = node.get("properties", {}).get(step, node) if step != "items" else node.get("items", node)
        required = list(node.get("required", []))
        for name in names:
            if name not in required and name in node.get("properties", {}):
                required.append(name)
        node["required"] = required
    return expanded


PLAN_SCHEMA: dict[str, Any] = json_schema(
    QAPlan,
    require={"subtasks.items": ["layer", "category", "depends_on"]},
)
CODE_SCHEMA: dict[str, Any] = json_schema(GeneratedCode)
REVIEW_SCHEMA: dict[str, Any] = json_schema(
    SubtaskReview,
    require={"": ["score", "feedback"]},
)
FINAL_REVIEW_SCHEMA: dict[str, Any] = json_schema(
    FinalAnswerReview,
    require={"": ["answer_wrong", "feedback"]},
)
# `answer_wrong` is optional to the parser so a reviewer that predates the split still
# validates, but a constrained generation has no such excuse: a null verdict there would
# silently fall back to "discard the answer", which is the behaviour the split exists to
# avoid. Generation must commit to a boolean.
FINAL_REVIEW_SCHEMA["properties"]["answer_wrong"] = {"type": "boolean"}
# Same reasoning for the score on both reviewers: optional to the parser so a sloppy
# reply still yields a verdict, but a constrained generation states a number in range.
_SCORE_PROPERTY = {"type": "number", "minimum": 0.0, "maximum": 1.0}
REVIEW_SCHEMA["properties"]["score"] = dict(_SCORE_PROPERTY)
FINAL_REVIEW_SCHEMA["properties"]["score"] = dict(_SCORE_PROPERTY)


def validation_message(error: ValidationError) -> str:
    """Flatten a pydantic error into the one line a retry prompt can act on."""
    parts = []
    for item in error.errors():
        location = ".".join(str(piece) for piece in item.get("loc", ())) or "<root>"
        parts.append(f"{location}: {item.get('msg', 'invalid')}")
    return "; ".join(parts) or str(error)


def _accepts_response_schema(client: Any) -> bool:
    """Whether this client's `generate` takes a `response_schema` keyword.

    Clients in other stages and in tests implement the two-argument contract, so the
    keyword only goes where it is understood. Deliberately uncached: keying a cache on
    `id(client)` is wrong because CPython reuses the id of a collected object, which
    hands one client the answer computed for an unrelated one. Signature inspection
    costs microseconds against a network round trip.
    """
    generate = getattr(client, "generate", None)
    if not callable(generate):
        return False
    try:
        parameters = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        return False
    return "response_schema" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def schema_if_enabled(env: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    """`schema` when this run asked for constrained decoding, otherwise None.

    Constrained decoding is opt-in because a grammar is a property of the serving stack,
    not of this code: the first backend it reached emitted JSON strings with no `\n`
    escape at all, which silently collapsed a generated Python cell onto one line. A
    switch keeps that a one-variable experiment instead of something every run inherits.
    """
    return schema if bool(getattr(env, "qa_structured_output", False)) else None


def generate_json(
    client: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
    schema: dict[str, Any] | None = None,
) -> Any:
    """Call `client.generate`, constraining the output to `schema` where supported."""
    if schema is not None and _accepts_response_schema(client):
        return client.generate(prompt, system_prompt=system_prompt, response_schema=schema)
    return client.generate(prompt, system_prompt=system_prompt)


__all__ = [
    "CODE_SCHEMA",
    "FINAL_REVIEW_SCHEMA",
    "PLAN_SCHEMA",
    "REVIEW_SCHEMA",
    "FinalAnswerReview",
    "GeneratedCode",
    "PlannedSubtask",
    "QAPlan",
    "SubtaskReview",
    "ValidationError",
    "generate_json",
    "schema_if_enabled",
    "json_schema",
    "validation_message",
]
