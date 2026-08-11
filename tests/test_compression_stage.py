from TableAgent.stages.compression import (
    SheetCompressor,
    SheetFormater,
    SheetHelper,
    SheetImager,
)


def test_compression_stage_exports_public_helpers():
    assert SheetCompressor.__module__ == "TableAgent.stages.compression.compressor"
    assert SheetFormater.__module__ == "TableAgent.stages.compression.formater"
    assert SheetHelper.__module__ == "TableAgent.stages.compression.helper"
    assert SheetImager.__module__ == "TableAgent.stages.compression.imager"
