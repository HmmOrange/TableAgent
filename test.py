from TableAgent.stages.compression import SheetCompressor, SheetImager

sheet_compressor = SheetCompressor()

file_path = "economy-table48.xlsx"
sheet_name = "DELETE template"

sheet_compressor.vertical_compress(
    file_path=file_path,
    sheet_name=sheet_name,
    output_path="abc.xlsx"
)

SheetImager.to_image(
    file_path="abc.xlsx",
    sheet_name=sheet_name,
    output_path="abc.jpg"
)
