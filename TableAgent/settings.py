"""Settings of the QA agent, read from one config mapping (the `table_agent` section of a pipeline config)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

QA_CHECKS = ("reverse", "independent", "debate")  # see src/checker.py
CHECKER_VIEWS = ("grid", "image", "frame")
START_VIEWS = ("text", "none")
EXPLORATIONS = ("progressive", "none")
SYNTHESES = ("code", "direct")
SWITCHABLE_TOOLS = ("find", "headers", "describe")  # Inspector tools that disabled_tools may switch off
ALL_TOOLS = "all"  # disabled_tools: [all] leaves the Inspector only the openpyxl workbook


@dataclass(frozen=True)
class QASettings:
    """How a question is answered and how cell ranges are rendered when the model looks at them.

    plan_count: how many starting plans are written independently; above 1, one more call reconciles them.
    checker, checker_view, checker_sees_value: the Checkers that test each found need (src/checker.py).
    synthesis_check: a Checker reviews how the answer was computed before the final review.
    inspector_max_turns: turns of one inspection, before the turn that only reports (src/inspector.py MAX_TURNS).
    max_returns: times the answer review may send the loop back (src/ledger.py MAX_RETURNS); 0 skips the review.
    disabled_tools: Inspector tools switched off, among find, headers and describe (headers and describe together);
        [all] switches off every tool: the Inspector reads the openpyxl workbook in its own code.
    start_view: what the starting plans are shown of the workbook: the text preview, or none (sheet names and sizes
        only). The Checkers keep the text preview.
    exploration: progressive (reasoning steps, Inspectors and Checkers read the needs before synthesis), or none (an
        ablation: after the plan, the Synthesizer's code reads the workbook itself through openpyxl, and the final
        review sees that code); none takes no checker, synthesis_check or direct synthesis.
    synthesis: code (the answer parts are computed in code from the evidence), or direct (an ablation: the model
        writes them from the evidence without code).
    save_artifacts: write each question's trace, ledger, events and code cells.
    generation_max_tokens: the generation cap set on the answer and vision models (None keeps the clients' own).
    max_context_chars: table text given to a fallback answer, and each evidence value given to a direct synthesis,
        is cut to this length.
    The rest configure LibreOffice rendering for `sheet.look`.
    """

    plan_count: int = 2
    checker: tuple[str, ...] = ()
    checker_view: str = "grid"
    checker_sees_value: bool = True
    synthesis_check: bool = False
    inspector_max_turns: int = 12
    max_returns: int = 1
    disabled_tools: tuple[str, ...] = ()
    start_view: str = "text"
    exploration: str = "progressive"
    synthesis: str = "code"
    save_artifacts: bool = True
    generation_max_tokens: int | None = None
    max_context_chars: int = 240000
    libreoffice_path: Path | None = None
    image_resolution: int = 192
    render_timeout_seconds: float = 60.0
    show_coordinates: bool = True
    max_cell_width: int = 50
    max_workers: int = 1  # questions answered at once; above 1, rendered PDF pages are rasterized in a subprocess

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "QASettings":
        """Settings from a mapping; a key left out keeps its default."""
        config = config or {}
        unknown = sorted(set(config) - set(cls.__dataclass_fields__) - {"llm_provider", "vlm_provider"})
        if unknown:
            raise ValueError(f"Unknown table_agent settings: {', '.join(unknown)}")
        defaults = cls()
        settings = cls(
            plan_count=_positive_int(config.get("plan_count", defaults.plan_count), "plan_count"),
            checker=_checks(config.get("checker", defaults.checker)),
            checker_view=_choice(config.get("checker_view", defaults.checker_view), "checker_view", CHECKER_VIEWS),
            checker_sees_value=_bool(config.get("checker_sees_value", defaults.checker_sees_value)),
            synthesis_check=_bool(config.get("synthesis_check", defaults.synthesis_check)),
            inspector_max_turns=_positive_int(
                config.get("inspector_max_turns", defaults.inspector_max_turns), "inspector_max_turns",
            ),
            max_returns=_count(config.get("max_returns", defaults.max_returns), "max_returns"),
            disabled_tools=_tools(config.get("disabled_tools", defaults.disabled_tools)),
            start_view=_choice(config.get("start_view", defaults.start_view), "start_view", START_VIEWS),
            exploration=_choice(config.get("exploration", defaults.exploration), "exploration", EXPLORATIONS),
            synthesis=_choice(config.get("synthesis", defaults.synthesis), "synthesis", SYNTHESES),
            save_artifacts=_bool(config.get("save_artifacts", defaults.save_artifacts)),
            generation_max_tokens=None if config.get("generation_max_tokens") is None else int(config["generation_max_tokens"]),
            max_context_chars=int(config.get("max_context_chars", defaults.max_context_chars)),
            libreoffice_path=None if config.get("libreoffice_path") is None else Path(str(config["libreoffice_path"])),
            image_resolution=int(config.get("image_resolution", defaults.image_resolution)),
            render_timeout_seconds=float(config.get("render_timeout_seconds", defaults.render_timeout_seconds)),
            show_coordinates=_bool(config.get("show_coordinates", defaults.show_coordinates)),
            max_cell_width=_positive_int(config.get("max_cell_width", defaults.max_cell_width), "max_cell_width"),
            max_workers=_positive_int(config.get("max_workers", defaults.max_workers), "max_workers"),
        )
        if settings.synthesis == "direct" and settings.synthesis_check:
            raise ValueError("table_agent.synthesis_check reviews the synthesis code: it needs synthesis: code")
        if settings.exploration == "none" and (settings.checker or settings.synthesis_check or settings.synthesis != "code"):
            raise ValueError(
                "table_agent.exploration: none reads no evidence to check or answer from: "
                "it needs checker: [], synthesis_check: false and synthesis: code"
            )
        return settings


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _positive_int(value: Any, key: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"table_agent.{key} must be a positive integer")
    return parsed


def _count(value: Any, key: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"table_agent.{key} must be 0 or more")
    return parsed


def _tools(value: Any) -> tuple[str, ...]:
    """Inspector tools to switch off: a list or a comma-separated string among find, headers, describe; or all."""
    names = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    tools = tuple(str(name).strip().lower() for name in names if str(name).strip())
    if tools == (ALL_TOOLS,):
        return tools
    if any(name not in SWITCHABLE_TOOLS for name in tools) or len(set(tools)) != len(tools):
        raise ValueError(
            f"table_agent.disabled_tools must list distinct tools among: {', '.join(SWITCHABLE_TOOLS)}, or be [all]"
        )
    if ("headers" in tools) != ("describe" in tools):
        raise ValueError("table_agent.disabled_tools: headers and describe are switched off together")
    return tools


def _checks(value: Any) -> tuple[str, ...]:
    """The QA checks, in order: a list or a comma-separated string of reverse, independent, debate."""
    names = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    checks = tuple(str(name).strip().lower() for name in names if str(name).strip())
    unknown = [name for name in checks if name not in QA_CHECKS]
    if unknown or len(set(checks)) != len(checks):
        raise ValueError(f"table_agent.checker must list distinct checks among: {', '.join(QA_CHECKS)}")
    return checks


def _choice(value: Any, key: str, choices: tuple[str, ...]) -> str:
    choice = str(value).strip().lower()
    if choice not in choices:
        raise ValueError(f"table_agent.{key} must be one of: {', '.join(choices)}")
    return choice
