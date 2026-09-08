# Plan: Remove Image Shifting from Structure Generation

## Branch

- Requested label: `[4] remove image shifting`.
- Git branch created: `4-remove-image-shifting`.
- The literal label is not a valid Git ref because Git rejects brackets and
  spaces, so the branch uses the closest valid normalized name.

## Approved direction

Remove viewport shifting and traversal directions completely from TableAgent's
structure generation. The workflow will render the sheet's complete used range
once and send that same full-sheet image to the layout VLM. It will not create a
queue, move a viewport, ask a direction VLM where to move, accept directions in
layout output, or retain viewport/shift configuration.

The generated structure must still pass the existing deterministic verification
and downstream group-enrichment flow. Verification retries may call the layout
VLM again with correction feedback, but every retry must use the same full-sheet
image and used range.

## Current flow (as scanned)

1. `TableLayoutWorkflow.run()` creates a `DirectionQueue`.
2. `corner_viewports()` seeds the queue with up to four sheet corners.
3. Each task renders a clipped range through
   `WorkbookRenderer.source_viewport_to_image()`.
4. `DirectionAgent` inspects the image and suggests remaining directions.
5. `LayoutAgent` receives the viewport image and incrementally merges YAML into
   the accumulated structure.
6. Verification failures retry the same viewport; successful/changed viewports
   call `_enqueue_shift()`, which uses `Viewport.shifted()` and `shift_cells`.
7. The workflow writes per-iteration artifacts/events and then optionally runs
   the existing full-used-range group-enrichment pass.

Relevant files identified:

- `layout/workflow.py`: queue orchestration, viewport rendering, direction
  calls, retry logic, shift scheduling, iteration artifacts.
- `traversal.py`: `DirectionQueue`, `TraversalTask`, `Viewport.shifted()`,
  corner/frontier helpers.
- `layout/direction_agent.py` and `layout/direction_prompts.py`: traversal
  direction VLM call, likely redundant for the new flow.
- `layout/agent.py` and `structure_prompts.py`: layout VLM contract currently
  describes a viewport and movement direction; prompt wording must describe a
  complete used-range image instead.
- `rendering/workbook.py`: existing `source_viewport_to_image()` can render a
  full range when passed `metadata.used_range`; `source_to_image()` also
  supports a whole-sheet render but has a different return contract.
- `cache.py`: recipe key currently includes viewport and shift settings, and
  workflow/cache version fields identify generated artifacts.
- `configs/table_agent.py` and root `config.yaml`: viewport/shift settings are
  currently parsed/configured.
- Existing structure tests in `TableAgent/tests/test_table_agent_mas.py` and
  `TableAgent/tests/test_table_agent_pipeline.py` assert queue ordering,
  viewport image names, and traversal behavior.

## Required changes

### Remove

- Delete `structure/traversal.py`, including `Direction`, `Viewport`,
  `TraversalTask`, `DirectionQueue`, `initial_viewport()`,
  `corner_viewports()`, and `frontier_directions()`.
- Delete `layout/direction_agent.py` and `layout/direction_prompts.py`.
- Remove the traversal symbols from `structure/__init__.py` lazy exports.
- Remove `direction_agent` construction/injection from `TableLayoutWorkflow`.
- Remove `_enqueue_shift()`, `_push_if_new()`, `_intersects()`,
  `_range_fully_covered()`, and `_has_enough_data()` if its only purpose remains
  filtering candidate shifted ranges.
- Remove queue/traversal state from the workflow: `queue`, `queued_ranges`,
  `successful_ranges`, `successful_viewports`, `direction_cache`,
  `zero_change_runs`, and viewport-keyed retry/feedback dictionaries.
- Remove direction fields from `LayoutAgent.run()`, `LayoutResult`, agent memory
  metadata, event records, artifact directory names, and progress fields.
- Remove direction extraction and compatibility handling from
  `layout/parsing.py`, including `remaining_directions`, `directions`, and the
  direction member in `LayoutParseResult`/`extract_layout_structure()`.
- Remove `viewport_rows`, `viewport_columns`, and `shift_cells` from
  `TableAgentConfig`, config loading, example/default configuration, cache recipe
  hashing, and tests.
- Remove tests that exist only for queue priority, viewport movement, corner
  seeding, direction prompts/parsing, or shifted-range traversal.
- Do not remove generic image tiling/post-processing or group enrichment:
  those have independent callers and are not part of image shifting.

### Change

- Refactor `TableLayoutWorkflow.run()` to:
  1. use `metadata.used_range` as the render range (or the renderer's existing
     whole-sheet behavior when the used range is empty),
  2. render one stable `table.png` full-sheet artifact rather than a new
     `viewport.png` for each iteration,
  3. call `LayoutAgent` with the full-sheet range and no direction input,
  4. run deterministic verification and the existing retry/nullification logic
     against that result, reusing the same image for correction retries,
  5. preserve `LayoutWorkflowResult`, final `structure.yaml`, changelog,
     events, response collection, progress callbacks, and `image_path`.
- Update `LayoutAgent.run()` and the layout prompt contract so the VLM is told
  it sees the complete sheet used range, should inspect all visible tables,
  and must not output or reason about traversal directions.
- Replace layout-specific `viewport_range` naming with `sheet_range` or
  `used_range` in the workflow, layout-agent API, prompts, events, and artifacts.
  Do not rename generic renderer methods or group-enrichment parameters solely
  for style when they still correctly accept an arbitrary cell range.
- Replace queue-oriented event fields with a full-sheet event containing the
  iteration/retry number, sheet range, layout-change state, token-cap state,
  and verification result.
- Update the cache recipe/workflow version so old shifted-view results cannot be
  silently reused as results from the new full-image algorithm. Remove viewport
  and shift settings from the new recipe key and include the updated prompt hash.
- Preserve `GroupEnrichmentStage`, which already renders `metadata.used_range`
  and is a separate post-layout VLM pass.

### Add

- Do not add a new renderer abstraction unless required. Reuse
  `source_viewport_to_image()` with `metadata.used_range`, since it already
  renders an exact cell range and is also used independently by other stages.
- Add a clear full-sheet prompt/template field (for example, `sheet_range` or
  `used_range`) and ensure metadata supplies the exact range shown to the VLM.
- Add only targeted updates to existing tests that validate the new single-call
  behavior, full-range image passed to the VLM, retry/verification behavior,
  artifact compatibility, and cache-version separation. Do not add broad or
  redundant test suites.

## Implementation sequence (after approval)

1. Confirm all callers of traversal, direction parsing, and removed configuration
   fields with CodeGraph before deletion.
2. Refactor workflow rendering/calling to one full used-range image while
   preserving output artifacts and progress/error handling.
3. Simplify `LayoutAgent` and layout parsing, then update structure prompts for
   full-sheet inspection with no direction contract.
4. Delete traversal and direction modules, exports, configuration, imports, and
   obsolete tests after all production callers are removed.
5. Update cache/workflow versioning and configuration examples.
6. Update existing meaningful tests, then run the focused structure and pipeline
   tests plus an end-to-end structure-generation run with a stub VLM/renderer.
7. Inspect generated artifacts (`structure.yaml`, events, image metadata,
   changelog) and verify one image is rendered, reused for retries, and no
   image-shift queue or direction call/artifact exists.

## Explicit non-goals

- No changes to QA retrieval, answer generation, compression, or unrelated
  rendering/tiling behavior.
- No redesign of structure YAML semantics or deterministic verification rules.
- No removal of group enrichment or source-preparation workflows.
- No implementation in this planning update; this file is the only codebase
  change in this step.

## Acceptance criteria

- Structure generation renders exactly one image for the worksheet used range.
- All layout attempts and verifier retries use that same image.
- No traversal/direction production modules, exports, prompts, parser fields,
  configuration fields, cache inputs, events, or artifacts remain.
- The final structure, verifier behavior, group enrichment, cache artifacts,
  progress reporting, and downstream QA inputs remain functional.
- Do not change other stages
