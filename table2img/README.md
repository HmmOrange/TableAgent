# table2img

Render table inputs to high-resolution PNG images through a headless browser.

The implementation follows the ST-Raptor table-image path conceptually, without importing or editing ST-Raptor code:

1. Normalize table input into HTML.
2. Preserve merged cells when the input exposes them, such as HiTab JSON `merged_regions` or XLSX merged ranges.
3. Add non-breaking cell padding and simple table borders, matching ST-Raptor's `sheet2html_file` style intent.
4. Render with Chrome/Chromium/Edge in headless mode at a configurable device scale factor.
5. Crop excess white viewport around the captured table.

## Usage

From the repo root:

```powershell
python -m table2img data\HiTab\tables\raw\100.json -o table2img\outputs\100.png --scale 2
python -m table2img data\MultiHiertt\dev.json -o table2img\outputs\mulhi.png --json-index 0 --table-index 0 --scale 2
python -m table2img path\to\table.md -o table2img\outputs\table.png
python -m table2img path\to\table.xlsx -o table2img\outputs\table.png
```

If Chrome is not auto-detected, pass `--browser C:\path\to\chrome.exe` or set `TABLE2IMG_BROWSER`.

## Smoke

```powershell
python -m table2img.smoke_semitab --project-root . --output-dir table2img\outputs\smoke --scale 2
```

This writes PNG and matching HTML files for two HiTab samples and one MulHi sample.

## ISE Table Pipeline Cache

The ISE Table `table2img_vlm` pipeline stores deterministic artifacts under:

```text
outputs/table2img_vlm/{hitab|mulhi}/{table_id}/{table_hash}/
```

Each table folder contains:

- `table.png`: the rendered table image reused by later runs.
- `table.html`: the intermediate HTML used for browser rendering.
- `table2img_metadata.json`: render metadata.
- `responses/*.json`: cached VLM answers keyed by model, question, and prompt.

With `reuse_images: true` and `cache_vlm_responses: true`, repeated runs on the same table/question reuse both the PNG and the VLM answer instead of calling the vision model again.
