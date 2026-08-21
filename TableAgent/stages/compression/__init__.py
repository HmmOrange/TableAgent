"""Workbook row compression and coordinate mapping."""

from .compressor import CompressionConfig, RowMap, SheetCompressor, SheetCompressionResult
from .formater import SheetFormater
from .helper import SheetHelper
try:
    from .imager import SheetImager
except ModuleNotFoundError:
    class SheetImager:
        """Optional legacy image adapter; image dependencies are not needed for compression."""
        @staticmethod
        def to_image(*args, **kwargs):
            raise RuntimeError("Sheet imaging requires the optional pymupdf dependency")
__all__ = [
    "CompressionConfig",
    "RowMap",
    "SheetCompressor",
    "SheetCompressionResult",
    "SheetFormater",
    "SheetHelper",
    "SheetImager",
]
