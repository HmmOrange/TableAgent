from __future__ import annotations

from datasets.base import EvalSample

from TableAgent.config import TableAgentConfig
from TableAgent.pipeline.common import SourceCandidate, is_siflex


class PromptBuilder:
    def __init__(self, settings: TableAgentConfig, templates: object):
        self.settings = settings
        self.templates = templates

    def layout_prompt(self, sample: EvalSample, table_context: str, feedback: str) -> str:
        feedback_block = f"\nPrevious verification feedback:\n{feedback}\n" if feedback else ""
        return self.templates.layout_user_prompt_template.format(
            table_context=table_context,
            question=sample.question,
            feedback_block=feedback_block,
        )

    def verification_prompt(self, sample: EvalSample, table_context: str, structure_text: str) -> str:
        return self.templates.verification_user_prompt_template.format(
            table_context=table_context,
            question=sample.question,
            structure_text=structure_text,
        )

    def source_layout_prompt(self, table_context: str, feedback: str) -> str:
        return self.templates.layout_user_prompt_template.format(
            table_context=table_context,
            question="Analyze the structure, rows, columns, headers, and metadata of this table.",
            feedback_block=f"\nPrevious verification feedback:\n{feedback}\n" if feedback else "",
        )

    def source_verification_prompt(self, table_context: str, structure_text: str) -> str:
        return self.templates.verification_user_prompt_template.format(
            table_context=table_context,
            question="Analyze the structure, rows, columns, headers, and metadata of this table.",
            structure_text=structure_text,
        )

    def answer_prompt(self, sample: EvalSample, table_context: str, structure_text: str) -> str:
        if is_siflex(sample):
            return self._siflex_answer_prompt(sample, table_context, structure_text)
        return self.templates.answer_user_prompt_template.format(
            question=sample.question,
            structure_text=structure_text,
            table_context=table_context,
        )

    def candidate_prompt_text(self, candidates: list[SourceCandidate], fit_context) -> str:
        lines = []
        for index, candidate in enumerate(candidates):
            preview = fit_context(candidate.sheet_text)[: self.settings.retrieval_candidate_max_chars]
            lines.append(
                f"Candidate {index}:\n"
                f"workbook: {candidate.workbook_path.name}\n"
                f"sheet: {candidate.sheet_name}\n"
                f"lexical_score: {candidate.score}\n"
                f"structure.yaml:\n{candidate.structure_text}\n"
                f"text preview:\n{preview}"
            )
        return "\n\n".join(lines)

    def _siflex_answer_prompt(self, sample: EvalSample, table_context: str, structure_text: str) -> str:
        answer_type = sample.raw.get("answer_type", "") if isinstance(sample.raw, dict) else ""
        if answer_type == "table":
            format_instructions = (
                "CRITICAL EXPECTED FORMAT: TABLE\n"
                "Format your final answer as a markdown table and preserve the table relations."
            )
        elif answer_type == "list":
            format_instructions = (
                "CRITICAL EXPECTED FORMAT: LIST\n"
                "Format your final answer as a bulleted list with one item per answer unit."
            )
        elif answer_type == "form":
            format_instructions = (
                "CRITICAL EXPECTED FORMAT: FORM/DOCUMENT\n"
                "Organize your final answer in a clear document structure."
            )
        else:
            format_instructions = "CRITICAL EXPECTED FORMAT: STRUCTURED RESPONSE\nAnswer clearly in the question language."
        return (
            f"Question: {sample.question}\n\n"
            f"Verified structure.yaml:\n{structure_text}\n\n"
            f"Table content:\n{table_context}\n\n"
            f"FORMAT INSTRUCTIONS:\n{format_instructions}\n\n"
            "Answer:"
        )
