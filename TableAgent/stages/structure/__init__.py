"""Structure stage: workbook ingestion, layout extraction, and verification."""

from importlib import import_module

_EXPORTS = {
    "Direction": (".traversal", "Direction"),
    "DirectionQueue": (".traversal", "DirectionQueue"),
    "ExStructMetadataExtractor": (".metadata", "ExStructMetadataExtractor"),
    "LayoutAgent": (".layout.agent", "LayoutAgent"),
    "SheetMetadata": (".metadata", "SheetMetadata"),
    "SourcePreparer": (".source_preparer", "SourcePreparer"),
    "StructureCache": (".cache", "StructureCache"),
    "StructureCacheRecord": (".cache", "StructureCacheRecord"),
    "StructureInput": (".contracts", "StructureInput"),
    "StructureOutput": (".contracts", "StructureOutput"),
    "StructurePipelineMixin": (".pipeline", "StructurePipelineMixin"),
    "StructureStage": (".stage", "StructureStage"),
    "TableLayoutWorkflow": (".layout.workflow", "TableLayoutWorkflow"),
    "TraversalTask": (".traversal", "TraversalTask"),
    "Viewport": (".traversal", "Viewport"),
    "corner_viewports": (".traversal", "corner_viewports"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    return getattr(import_module(f"{__name__}{module_name}"), attribute)


__all__ = list(_EXPORTS)
