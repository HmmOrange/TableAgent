from __future__ import annotations

from TableAgent.schema import EvalSample

from TableAgent.configs import TableAgentConfig
from TableAgent.pipeline.common import SourceCandidate


class PromptBuilder:
    def __init__(self, settings: TableAgentConfig, templates: object):
        self.settings = settings
        self.templates = templates

    def answer_prompt(self, sample: EvalSample, table_context: str, structure_text: str) -> str:
        prompt = self.templates.answer_user_prompt_template.format(
            question=sample.question,
            structure_text=structure_text,
            table_context=table_context,
        )
        sample_path = str(sample.sample_path or "").lower()
        answer_type = str(sample.raw.get("answer_type", "")).strip().lower() if isinstance(sample.raw, dict) else ""
        if "siflex" not in sample_path or answer_type not in {"table", "list", "form"}:
            return prompt
        format_instructions = {
            "table": (
                "CRITICAL EXPECTED FORMAT: TABLE\n"
                "Format your final answer as a markdown table."
            ),
            "list": (
                "CRITICAL EXPECTED FORMAT: LIST\n"
                "Format your final answer as a bulleted list."
            ),
            "form": (
                "CRITICAL EXPECTED FORMAT: FORM/DOCUMENT\n"
                "Organize your final answer in a clear document structure."
            ),
        }[answer_type]
        return f"{prompt}\n\nFORMAT INSTRUCTIONS:\n{format_instructions}"

    def candidate_prompt_text(self, candidates: list[SourceCandidate], fit_context) -> str:
        lines = []
        for index, candidate in enumerate(candidates):
            card = candidate.retrieval_card or fit_context(candidate.sheet_text)[: self.settings.retrieval_candidate_max_chars]
            card = card[: self.settings.retrieval_candidate_max_chars]
            lines.append(
                f"Candidate {index}:\n"
                f"workbook: {candidate.workbook_path.name}\n"
                f"sheet: {candidate.sheet_name}\n"
                f"retrieval_type: {candidate.retrieval_type}\n"
                f"retrieval_level: {candidate.retrieval_level}\n"
                f"table_id: {candidate.table_id}\n"
                f"table_name: {candidate.table_name}\n"
                f"score: {candidate.score}\n"
                f"lexical_score: {candidate.lexical_score}\n"
                f"embedding_score: {candidate.embedding_score}\n"
                f"embedding_used: {candidate.embedding_used}\n"
                f"entity_score: {candidate.entity_score}\n"
                f"matched_terms: {list(candidate.matched_terms)}\n"
                f"missing_terms: {list(candidate.missing_terms)}\n"
                f"retrieval_card:\n{card}"
            )
        return "\n\n".join(lines)
