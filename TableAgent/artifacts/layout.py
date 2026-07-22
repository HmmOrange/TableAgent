from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

from TableAgent.pipeline.common import safe_name


def workbook_artifact_dir(
    root: Path,
    workbook_name: str,
    sha256: str | None = None,
    *,
    sources: bool = True,
) -> Path:
    """Return the nested workbook directory used by cache or job artifacts."""
    base = safe_name(workbook_name)
    if sha256:
        base = f"{base}_{sha256[:8]}"
    return root / "sources" / base if sources else root / base


def sheet_artifact_dir(workbook_dir: Path, sheet_name: str) -> Path:
    return workbook_dir / safe_name(sheet_name)


def legacy_sheet_dir(root: Path, workbook_filename: str, sheet_name: str) -> Path:
    return root / "sources" / f"{safe_name(workbook_filename)}_{safe_name(sheet_name)}"


def iter_sheet_artifact_dirs(source_root: Path) -> Iterator[Path]:
    """Discover both nested and legacy sheet directories by their required files."""
    if not source_root.is_dir():
        return
    seen: set[Path] = set()
    for structure_path in sorted(source_root.rglob("structure.yaml")):
        directory = structure_path.parent
        if not (directory / "metadata.json").is_file():
            continue
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield directory


def copy_artifact_tree(source: Path, target: Path) -> None:
    """Copy a sheet artifact directory without copying unrelated workbook files."""
    if source.resolve() == target.resolve():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)
