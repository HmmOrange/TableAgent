from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

from TableAgent.QA.actions.base_action import (
    BaseCodeGenerationAction,
    CodeGenerationRequest,
    CodeGenerationResult,
)
from TableAgent.QA.prompts.react_prompts import (
    REACT_SYSTEM_PROMPT,
    REACT_USER_PROMPT_TEMPLATE,
    REVISION_USER_PROMPT_TEMPLATE,
)
from TableAgent.QA.prompts.synthesis_prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT_TEMPLATE,
)

CODE_JSON_REPAIR_SYSTEM_PROMPT = """You are a strict JSON code-generation formatter.
Your only job is to produce one valid JSON object for the spreadsheet notebook agent.

Return exactly one JSON object or exactly one ```json fenced JSON object.
Do not include prose before or after it.
The JSON object must contain three non-empty string fields:
- "reasoning"
- "code"
- "description"

The "code" field must contain executable Python code. Put any inspection or computation steps in code, not in prose.
"""


def _clip_text(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def get_structure_summary(env: Any, table_id: str) -> str:
    struct = env.get_table_structure(table_id)
    if not struct:
        return f"Table '{table_id}' not found."

    summary_lines = []
    summary_lines.append(f"Table ID: {table_id}")
    summary_lines.append(f"Table Name: {struct.get('name')}")
    summary_lines.append(f"Description: {struct.get('description')}")
    summary_lines.append(f"Sheet: {struct.get('sheet')}")
    summary_lines.append("Headers:")

    def format_header(h, indent=2):
        space = " " * indent
        line = f"{space}- ID: {h.id}, Label: {h.label}, Description: {h.description}, Orientation: {h.orientation}"
        summary_lines.append(line)
        for sub in h.sub_headers:
            format_header(sub, indent + 2)

    for h in struct.get("headers", []):
        format_header(h)

    return "\n".join(summary_lines)


def get_table_catalog_summary(env: Any) -> str:
    table_ids = env.operators.list_tables() if hasattr(env, "operators") else list(getattr(env, "structures", {}).keys())
    if not table_ids:
        return "No tables are available."

    lines = []
    for table_id in table_ids:
        struct = env.get_table_structure(table_id)
        headers = struct.get("headers", []) if struct else []
        header_bits = []
        for header in headers[:20]:
            label = getattr(header, "label", "")
            h_id = getattr(header, "id", "")
            desc = getattr(header, "description", "")
            bit = f"{label} ({h_id})" if h_id else str(label)
            if desc:
                bit += f": {desc}"
            header_bits.append(bit)
        if len(headers) > 20:
            header_bits.append(f"... {len(headers) - 20} more headers")
        lines.append(
            "\n".join([
                f"- table_id: {table_id}",
                f"  name: {struct.get('name', '') if struct else ''}",
                f"  description: {struct.get('description', '') if struct else ''}",
                f"  sheet: {struct.get('sheet', '') if struct else ''}",
                f"  headers: {'; '.join(header_bits) if header_bits else '(none)'}",
            ])
        )
    return "\n".join(lines)


def get_selected_structure_summary(env: Any, table_ids: list[str]) -> str:
    return "\n\n".join(get_structure_summary(env, table_id) for table_id in table_ids)


def get_prior_outcomes(env: Any) -> str:
    if not hasattr(env, "notebook") or not env.notebook.cells:
        return "No code has been executed yet."
    if hasattr(env, "get_history"):
        return env.get_history(last_n=5, include_output=True, max_code_len=900, max_output_len=800)
    return env.notebook.get_history(last_n=5, include_output=True, max_code_len=900, max_output_len=800)


def get_operator_catalog(env: Any) -> str:
    if hasattr(env, "operators") and hasattr(env.operators, "operator_catalog"):
        return env.operators.operator_catalog()
    return "No operator catalog is available."


def parse_model_output(content: str) -> Tuple[str, str, str]:
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    payload = json_match.group(1) if json_match else content.strip()
    try:
        data = json.loads(payload)
    except Exception as exc:
        raise ValueError("Code generation output must be valid JSON or a ```json code block.") from exc

    if not isinstance(data, dict):
        raise ValueError("Code generation JSON must be an object.")

    reasoning = str(data.get("reasoning", "")).strip()
    code = str(data.get("code", "")).strip()
    description = str(data.get("description", "")).strip()

    missing = [
        field_name
        for field_name, value in (
            ("reasoning", reasoning),
            ("code", code),
            ("description", description),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Code generation JSON is missing non-empty fields: {missing}")

    return reasoning, code, description


class LLMCodeGenerationAction(BaseCodeGenerationAction):
    """LLM-backed action that writes executable Python code for a QA subtask."""
    name = "llm_code_generation"
    desc = "Generate notebook Python code for an inspect or synthesis subtask."

    def __init__(self, llm_client: Any, env: Optional[Any] = None, output_format_retries: int = 2):
        self.llm_client = llm_client
        self.env = env
        self.output_format_retries = max(0, int(output_format_retries))

    def run(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        if not self.env:
            raise ValueError("Environment not set on LLMCodeGenerationAction.")
        operator_catalog = get_operator_catalog(self.env)

        if request.layer == "table_inspect":
            if request.round_num == 1:
                table_catalog = get_table_catalog_summary(self.env)
                available_vars = list(self.env.notebook.namespace.keys())
                available_vars = [
                    v for v in available_vars
                    if not v.startswith("__") and v not in {"pd", "openpyxl", "env", "operators", "Cell", "CellRange", "AxisSelection", "Header", "np"}
                ]
                prior_outcomes = get_prior_outcomes(self.env)
                formatted_experience = self.env.experience_pool.format()
                prompt = REACT_USER_PROMPT_TEMPLATE.format(
                    question=request.question,
                    subtask_description=(
                        f"Subtask: {request.subtask_id}.\n"
                        "Choose the relevant table_id or table_ids from this catalog.\n"
                        "Your code must set `selected_table_ids` to a non-empty list of valid table_id strings. "
                        "Also print a compact explanation of the selection.\n\n"
                        f"Table catalog:\n{table_catalog}\n\n"
                        f"Experience:\n{formatted_experience}"
                    ),
                    available_variables=", ".join(available_vars) if available_vars else "None",
                    prior_outcomes=prior_outcomes,
                )
                system_prompt = REACT_SYSTEM_PROMPT.format(operator_catalog=operator_catalog)
            else:
                last_cell = self.env.notebook.cells[-1] if self.env.notebook.cells else None
                failed_code = last_cell.code if last_cell else ""
                if last_cell:
                    observation = self.env.notebook.observation_for_cell(last_cell)
                    error_msg = observation.format()
                else:
                    error_msg = "Unknown execution error"
                prompt = REVISION_USER_PROMPT_TEMPLATE.format(
                    question=request.question,
                    subtask_description=(
                        f"Subtask: {request.subtask_id}\n"
                        "Revise the table selection code. It must set `selected_table_ids` to a non-empty list of valid table_id strings.\n"
                        f"Previous attempts and reasoning:\n{self.env.experience_pool.format()}"
                    ),
                    failed_code=failed_code,
                    error_message=error_msg,
                )
                system_prompt = REACT_SYSTEM_PROMPT.format(operator_catalog=operator_catalog)
        elif request.layer == "inspect":
            if request.round_num == 1:
                table_ids = None
                if request.subtask and hasattr(request.subtask, "metadata") and isinstance(request.subtask.metadata, dict):
                    table_ids = request.subtask.metadata.get("table_ids")
                    if not table_ids and request.subtask.metadata.get("table_id"):
                        table_ids = [request.subtask.metadata.get("table_id")]
                if not table_ids:
                    table_ids = self.env.execution_namespace.get("selected_table_ids")
                if isinstance(table_ids, str):
                    table_ids = [table_ids]
                if not table_ids:
                    table_ids = [self.env.default_table_id()]

                struct_summary = get_selected_structure_summary(self.env, [str(table_id) for table_id in table_ids])
                available_vars = list(self.env.notebook.namespace.keys())
                available_vars = [
                    v for v in available_vars
                    if not v.startswith("__") and v not in {"pd", "openpyxl", "env", "operators", "Cell", "CellRange", "AxisSelection", "Header", "np"}
                ]

                prior_outcomes = get_prior_outcomes(self.env)
                formatted_experience = self.env.experience_pool.format()

                prompt = REACT_USER_PROMPT_TEMPLATE.format(
                    question=request.question,
                    subtask_description=(
                        f"Subtask: {request.subtask_id}.\n"
                        f"Table structure:\n{struct_summary}\n\n"
                        f"Experience:\n{formatted_experience}"
                    ),
                    available_variables=", ".join(available_vars) if available_vars else "None",
                    prior_outcomes=prior_outcomes,
                )
                system_prompt = REACT_SYSTEM_PROMPT.format(operator_catalog=operator_catalog)
            else:
                last_cell = self.env.notebook.cells[-1] if self.env.notebook.cells else None
                failed_code = last_cell.code if last_cell else ""
                if last_cell:
                    observation = self.env.notebook.observation_for_cell(last_cell)
                    error_msg = observation.format()
                else:
                    error_msg = "Unknown execution error"

                prompt = REVISION_USER_PROMPT_TEMPLATE.format(
                    question=request.question,
                    subtask_description=(
                        f"Subtask: {request.subtask_id}\n"
                        f"Previous attempts and reasoning:\n{self.env.experience_pool.format()}"
                    ),
                    failed_code=failed_code,
                    error_message=error_msg,
                )
                system_prompt = REACT_SYSTEM_PROMPT.format(operator_catalog=operator_catalog)
        elif request.layer == "synthesis":
            available_vars = list(self.env.notebook.namespace.keys())
            available_vars = [
                v for v in available_vars
                if not v.startswith("__") and v not in {"pd", "openpyxl", "env", "operators", "Cell", "CellRange", "AxisSelection", "Header", "np"}
            ]
            prior_outcomes = get_prior_outcomes(self.env)

            prompt = SYNTHESIS_USER_PROMPT_TEMPLATE.format(
                question=request.question,
                available_variables=", ".join(available_vars) if available_vars else "None",
                prior_outcomes=prior_outcomes,
            )
            system_prompt = SYNTHESIS_SYSTEM_PROMPT.format(operator_catalog=operator_catalog)
        else:
            raise ValueError(f"Unknown subtask layer: {request.layer}")

        self.env.logger.log_event("generate_call", {
            "action": self.name,
            "layer": request.layer,
            "round_num": request.round_num,
            "prompt": prompt,
            "system_prompt": system_prompt,
        })

        response = self.llm_client.generate(prompt, system_prompt=system_prompt)

        self.env.logger.log_event("generate_response", {
            "content": response.content,
        })

        reasoning, code, description = self._parse_or_repair_response(
            content=response.content,
            request=request,
            prompt=prompt,
            system_prompt=system_prompt,
        )

        self.env.logger.log_event("generate_parsed", {
            "reasoning": reasoning,
            "code": code,
            "description": description,
        })

        return CodeGenerationResult(code=code, description=description, reasoning=reasoning)

    def _parse_or_repair_response(
        self,
        content: str,
        request: CodeGenerationRequest,
        prompt: str,
        system_prompt: str,
    ) -> Tuple[str, str, str]:
        current_content = content
        last_error: ValueError | None = None

        for attempt in range(self.output_format_retries + 1):
            try:
                parsed = parse_model_output(current_content)
                if attempt > 0:
                    self.env.logger.log_event("generate_repair_parsed", {
                        "repair_attempt": attempt,
                    })
                return parsed
            except ValueError as exc:
                last_error = exc
                self.env.logger.log_event("generate_parse_error", {
                    "repair_attempt": attempt,
                    "error": str(exc),
                    "content": current_content,
                })
                if attempt >= self.output_format_retries:
                    raise

                repair_prompt = self._build_repair_prompt(
                    request=request,
                    original_prompt=prompt,
                    original_system_prompt=system_prompt,
                    invalid_response=current_content,
                    parse_error=str(exc),
                    repair_attempt=attempt + 1,
                )
                self.env.logger.log_event("generate_repair_call", {
                    "repair_attempt": attempt + 1,
                    "prompt": repair_prompt,
                    "system_prompt": CODE_JSON_REPAIR_SYSTEM_PROMPT,
                })
                repair_response = self.llm_client.generate(
                    repair_prompt,
                    system_prompt=CODE_JSON_REPAIR_SYSTEM_PROMPT,
                )
                current_content = repair_response.content
                self.env.logger.log_event("generate_repair_response", {
                    "repair_attempt": attempt + 1,
                    "content": current_content,
                })

        raise last_error or ValueError("Code generation output could not be parsed as JSON.")

    def _build_repair_prompt(
        self,
        request: CodeGenerationRequest,
        original_prompt: str,
        original_system_prompt: str,
        invalid_response: str,
        parse_error: str,
        repair_attempt: int,
    ) -> str:
        subtask_description = ""
        if request.subtask is not None:
            subtask_description = getattr(request.subtask, "description", "")

        return (
            "The previous response for a TableAgent code-generation action was invalid.\n"
            f"Repair attempt: {repair_attempt}\n"
            f"Parse error: {parse_error}\n\n"
            f"User question: {request.question}\n"
            f"Subtask id: {request.subtask_id}\n"
            f"Layer: {request.layer}\n"
            f"Subtask description: {subtask_description}\n\n"
            "Original task prompt:\n"
            f"{_clip_text(original_prompt)}\n\n"
            "Original system instructions:\n"
            f"{_clip_text(original_system_prompt)}\n\n"
            "Invalid response:\n"
            f"{_clip_text(invalid_response, limit=4000)}\n\n"
            "Now produce a fresh valid JSON response. Do not merely explain what should be done. "
            "Write executable Python in the code field.\n"
            "Required shape:\n"
            "```json\n"
            "{\"reasoning\":\"concise reasoning\",\"code\":\"executable python code\",\"description\":\"short description\"}\n"
            "```"
        )
