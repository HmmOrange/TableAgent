from __future__ import annotations

# Compatibility shim: implementation now lives under TableAgent/.
from TableAgent.configs import TableAgentConfig, TableAgentSettings  # noqa: F401
from TableAgent.structure.layout.parsing import (  # noqa: F401
    _extract_yaml_text,
    _is_valid_structure,
    _parse_yaml_mapping,
    extract_strict_structure,
)
from TableAgent.pipeline import TableAgentPipeline  # noqa: F401
from TableAgent.rendering.image_utils import (  # noqa: F401
    compute_viewport_and_scale,
    _generate_image_tiles,
    _resize_image_file_to_fit,
)
from TableAgent.utils.table_text import (  # noqa: F401
    _lexical_overlap_score,
    _rows_to_markdown_simple,
)

__all__ = [
    "TableAgentPipeline",
    "TableAgentConfig",
    "TableAgentSettings",
    "compute_viewport_and_scale",
    "_extract_yaml_text",
    "extract_strict_structure",
    "_generate_image_tiles",
    "_is_valid_structure",
    "_lexical_overlap_score",
    "_parse_yaml_mapping",
    "_resize_image_file_to_fit",
    "_rows_to_markdown_simple",
]

