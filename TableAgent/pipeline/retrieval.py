from __future__ import annotations

import json
from pathlib import Path

from datasets.base import EvalSample
from utils.llm.base import BaseLLM, LLMResponse

from TableAgent.config import TableAgentConfig
from TableAgent.perception.structure import _is_valid_structure, _parse_yaml_mapping
from TableAgent.pipeline.common import SourceCandidate, is_siflex
from TableAgent.utils.table_text import _lexical_overlap_score


class SourceRetriever:
    def __init__(self, settings: TableAgentConfig, llm: BaseLLM, templates: object, prompt_builder: object):
        self.settings = settings
        self.llm = llm
        self.templates = templates
        self.prompt_builder = prompt_builder

    def select(self, sample: EvalSample, responses: list[LLMResponse], fit_context) -> SourceCandidate | None:
        if not is_siflex(sample) or not sample.table_path:
            return None
        candidates = self.load_candidates(sample)
        if not candidates:
            return None
        if not self.settings.retrieval_rerank_with_llm or len(candidates) == 1:
            return candidates[0]

        top_candidates = candidates[: max(1, self.settings.retrieval_top_k)]
        prompt = self.templates.reranker_user_prompt_template.format(
            question=sample.question,
            candidates_text=self.prompt_builder.candidate_prompt_text(top_candidates, fit_context),
        )
        response = self.llm.generate(prompt=prompt, system_prompt=self.templates.reranker_system_prompt)
        responses.append(response)
        return self._choose_from_reranker(response.content, top_candidates)

    def load_candidates(self, sample: EvalSample) -> list[SourceCandidate]:
        artifact_dir = self.settings.source_artifact_dir or self.settings.artifact_dir
        source_dirs = artifact_dir / "sources"
        if not source_dirs.is_dir():
            return []
        allowed_paths = {str(Path(value.strip()).resolve()) for value in str(sample.table_path).split(";") if value.strip()}
        candidates: list[SourceCandidate] = []

        for source_dir in source_dirs.iterdir():
            candidate = self._candidate_from_dir(source_dir, allowed_paths, sample.question)
            if candidate is not None:
                candidates.append(candidate)
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _candidate_from_dir(self, source_dir: Path, allowed_paths: set[str], query: str) -> SourceCandidate | None:
        if not source_dir.is_dir():
            return None
        metadata_path = source_dir / "metadata.json"
        structure_path = source_dir / "structure.yaml"
        sheet_text_path = source_dir / "sheet_text.txt"
        image_path = source_dir / "table.png"
        if not (metadata_path.is_file() and structure_path.is_file() and sheet_text_path.is_file() and image_path.is_file()):
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        workbook_path = Path(metadata.get("workbook_path", ""))
        if allowed_paths and str(workbook_path.resolve()) not in allowed_paths:
            return None

        structure_text = structure_path.read_text(encoding="utf-8")
        if not _is_valid_structure(structure_text):
            return None
        sheet_text = sheet_text_path.read_text(encoding="utf-8")
        return SourceCandidate(
            directory=source_dir,
            workbook_path=workbook_path,
            sheet_name=str(metadata.get("sheet_name", "")),
            image_path=image_path,
            html_path=source_dir / "table.html",
            structure_text=structure_text,
            sheet_text=sheet_text,
            score=_lexical_overlap_score(query, sheet_text),
        )

    @staticmethod
    def _choose_from_reranker(content: str, top_candidates: list[SourceCandidate]) -> SourceCandidate:
        parsed = _parse_yaml_mapping(content)
        try:
            selected_index = int(parsed.get("selected_index"))
        except (TypeError, ValueError):
            selected_index = -1

        if 0 <= selected_index < len(top_candidates):
            chosen = top_candidates[selected_index]
            object.__setattr__(chosen, "reranker_selected_index", selected_index)
            object.__setattr__(chosen, "reranker_rationale", str(parsed.get("rationale", "")))
            object.__setattr__(chosen, "fallback_used", False)
            return chosen

        chosen = top_candidates[0]
        object.__setattr__(chosen, "reranker_selected_index", None)
        object.__setattr__(chosen, "reranker_rationale", "")
        object.__setattr__(chosen, "fallback_used", True)
        return chosen
