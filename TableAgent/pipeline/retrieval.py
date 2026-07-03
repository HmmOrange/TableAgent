from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any
import numpy as np
import requests

from datasets.base import EvalSample
from utils.llm.base import BaseLLM, LLMResponse

from TableAgent.config import TableAgentConfig
from configs.config import load_config
from configs.embedding_config import resolve_embedding_config
from TableAgent.perception.structure import _is_valid_structure, _parse_yaml_mapping
from TableAgent.pipeline.common import SourceCandidate, is_siflex
from TableAgent.utils.table_text import _lexical_overlap_score


class MockEmbeddingModel:
    """A simple deterministic embedding model for offline testing without VPN/model."""
    def __init__(self, dim: int = 128):
        self.dim = dim

    async def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = re.findall(r"\w+", text.lower())
            if not words:
                words = [""]
            for word in words:
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)

    async def batch_encode(self, texts: str | list[str], **kwargs) -> np.ndarray:
        return await self.encode(texts)


class OpenAICompatibleEmbeddingClient:
    """Minimal OpenAI-compatible embedding client used by TableAgent retrieval."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float | None = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    @classmethod
    def from_config(cls, provider: str | None = None, config_path: str | Path = "configs/config.yaml"):
        config = load_config(config_path)
        provider_name, embedding_config = resolve_embedding_config(
            config,
            provider or "embedding",
        )
        backend = str(embedding_config.get("provider", "")).lower()
        if backend not in {"openai", "openai_compatible"}:
            raise ValueError(
                f"Unsupported retrieval embedding backend for '{provider_name}': {backend}. "
                "Only OpenAI-compatible embeddings are supported."
            )
        base_url = embedding_config.get("base_url") or embedding_config.get("endpoint")
        model = embedding_config.get("model") or embedding_config.get("model_name")
        if not base_url or not model:
            raise ValueError(
                f"Embedding provider '{provider_name}' must configure base_url/endpoint and model/model_name."
            )
        timeout = embedding_config.get("timeout", embedding_config.get("timeout_seconds", 120))
        return cls(
            base_url=str(base_url),
            model=str(model),
            api_key=embedding_config.get("api_key"),
            timeout=float(timeout) if timeout is not None else None,
        )

    async def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = sorted(payload.get("data", []), key=lambda row: row.get("index", 0))
        return np.array([row["embedding"] for row in rows], dtype=np.float32)

    async def batch_encode(self, texts: str | list[str], **kwargs) -> np.ndarray:
        return await self.encode(texts)


def _extract_headers_text(headers_list: list, *, limit: int = 40) -> list[str]:
    parts = []
    for header in headers_list:
        if len(parts) >= limit:
            break
        if not isinstance(header, dict):
            continue
        h_id = header.get("id", "")
        h_label = header.get("label", "")
        h_desc = header.get("description", "")
        h_orient = header.get("orientation", "")
        header_text = f"{h_label}"
        if h_id:
            header_text += f" ({h_id})"
        if h_desc:
            header_text += f": {h_desc}"
        if h_orient:
            header_text += f" [{h_orient}]"
        parts.append(header_text)
        if "sub_headers" in header and isinstance(header["sub_headers"], list):
            remaining = limit - len(parts)
            parts.extend(_extract_headers_text(header["sub_headers"], limit=remaining))
    return parts


def _build_retrieval_card(workbook_path: Path, sheet_name: str, structure_text: str, sheet_text: str) -> str:
    import yaml
    table_parts = []
    try:
        structure_data = yaml.safe_load(structure_text)
    except Exception:
        structure_data = {}

    if isinstance(structure_data, dict):
        top_headers = structure_data.get("headers")
        if isinstance(top_headers, list):
            headers_text = "; ".join(_extract_headers_text(top_headers))
            if headers_text:
                table_parts.append(f"Headers: {headers_text}")

        for table_key, table_val in structure_data.items():
            if not isinstance(table_val, dict):
                continue
            t_id = table_val.get("id", table_key)
            t_name = table_val.get("name", "")
            t_desc = table_val.get("description", "")
            table_parts.append(f"Table: {t_name or t_id} ({t_id})")
            if t_desc:
                table_parts.append(f"Description: {t_desc}")
            headers = table_val.get("headers", [])
            if isinstance(headers, list):
                headers_text = "; ".join(_extract_headers_text(headers))
                if headers_text:
                    table_parts.append(f"Headers: {headers_text}")

    sheet_preview = sheet_text[:500] if sheet_text else ""

    parts = [
        f"Workbook: {workbook_path.name}",
        f"Sheet: {sheet_name}",
    ]
    parts.extend(table_parts)
    if sheet_preview:
        parts.append(f"Sheet preview: {sheet_preview}")

    return "\n".join(parts)


class SourceRetriever:
    def __init__(self, settings: TableAgentConfig, llm: BaseLLM, templates: object, prompt_builder: object, embedding_client: Any = None):
        self.settings = settings
        self.llm = llm
        self.templates = templates
        self.prompt_builder = prompt_builder
        self.embedding_client = embedding_client

        if self.embedding_client is None:
            provider = getattr(self.settings, "retrieval_embedding_provider", None)
            if provider == "mock":
                self.embedding_client = MockEmbeddingModel()
            elif provider in {"default", "live", "openai_embedding"}:
                try:
                    provider_name = None if provider in {"default", "live"} else provider
                    self.embedding_client = OpenAICompatibleEmbeddingClient.from_config(provider_name)
                except Exception as exc:
                    import sys
                    print(f"Embedding client initialization failed: {exc}", file=sys.stderr)
                    self.embedding_client = None
            else:
                self.embedding_client = None

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

        if not candidates:
            return []

        embedding_client = self.embedding_client
        embedding_used = False

        if embedding_client is not None:
            try:
                query = sample.question
                docs = [c.retrieval_card for c in candidates]

                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                async def get_embeddings():
                    return await embedding_client.encode([query] + docs)

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(asyncio.run, get_embeddings())
                        vectors = future.result()
                else:
                    vectors = asyncio.run(get_embeddings())

                query_vec = vectors[0]
                doc_vecs = vectors[1:]

                query_norm = np.linalg.norm(query_vec)

                updated_candidates = []
                for idx, c in enumerate(candidates):
                    doc_vec = doc_vecs[idx]
                    doc_norm = np.linalg.norm(doc_vec)
                    if query_norm == 0 or doc_norm == 0:
                        emb_score = 0.0
                    else:
                        emb_score = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))

                    updated_c = SourceCandidate(
                        directory=c.directory,
                        workbook_path=c.workbook_path,
                        sheet_name=c.sheet_name,
                        image_path=c.image_path,
                        html_path=c.html_path,
                        structure_text=c.structure_text,
                        sheet_text=c.sheet_text,
                        score=c.score,
                        lexical_score=c.lexical_score,
                        embedding_score=emb_score,
                        embedding_used=True,
                        retrieval_card=c.retrieval_card,
                    )
                    updated_candidates.append(updated_c)
                candidates = updated_candidates
                embedding_used = True
            except Exception as e:
                import sys
                print(f"Embedding generation failed: {e}", file=sys.stderr)

        lexical_weight = getattr(self.settings, "retrieval_lexical_weight", 0.5)
        embedding_weight = getattr(self.settings, "retrieval_embedding_weight", 0.5)

        max_lex = max((c.lexical_score for c in candidates), default=1.0)
        min_lex = min((c.lexical_score for c in candidates), default=0.0)
        lex_range = max_lex - min_lex

        final_candidates = []
        for c in candidates:
            norm_lex = (c.lexical_score - min_lex) / lex_range if lex_range > 0 else 1.0

            if embedding_used:
                hybrid_score = lexical_weight * norm_lex + embedding_weight * c.embedding_score
            else:
                hybrid_score = c.lexical_score

            final_c = SourceCandidate(
                directory=c.directory,
                workbook_path=c.workbook_path,
                sheet_name=c.sheet_name,
                image_path=c.image_path,
                html_path=c.html_path,
                structure_text=c.structure_text,
                sheet_text=c.sheet_text,
                score=hybrid_score,
                lexical_score=c.lexical_score,
                embedding_score=c.embedding_score,
                embedding_used=embedding_used,
                retrieval_card=c.retrieval_card,
            )
            final_candidates.append(final_c)

        return sorted(final_candidates, key=lambda candidate: candidate.score, reverse=True)

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

        retrieval_card = _build_retrieval_card(workbook_path, str(metadata.get("sheet_name", "")), structure_text, sheet_text)
        lexical_score = _lexical_overlap_score(query, retrieval_card)

        return SourceCandidate(
            directory=source_dir,
            workbook_path=workbook_path,
            sheet_name=str(metadata.get("sheet_name", "")),
            image_path=image_path,
            html_path=source_dir / "table.html",
            structure_text=structure_text,
            sheet_text=sheet_text,
            score=lexical_score,
            lexical_score=lexical_score,
            embedding_score=0.0,
            embedding_used=False,
            retrieval_card=retrieval_card,
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
