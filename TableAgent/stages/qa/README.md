# QA stage

Answers one question about a workbook by exploring it:
every value is read from the cells.

## Flow (`orchestrator.py`)

The Orchestrator holds the ledger (`ledger.py`) and runs the loop. Its three abstract reasoning steps
are model calls made by `Reasoner`; each reads the ledger, never an Inspector's or a Checker's transcript.
A **need** is one fact to read from the workbook, described by its meaning (entity, measure, period,
condition, unit), never by a cell; its value may be one value or a list or dict when it covers a set of rows.

1. **reason_start**. Reads the question and an A1 preview of the workbook (a merged cell written once, with
   its range). It writes the answer slots, each tied to the phrase of the question that asks for it, the
   needs to read, and a hypothesis of how the needs combine, with no values. The plan is written twice,
   independently, and a third call settles where the two differ. A plan with more than 3 needs is sent back
   once to merge the needs read from one row, column or block. The preview shows no formats: when the workbook
   has fill-colour legends, one line names them, and a need may be described by a format ("the cells filled
   green in the Price column"); which cells carry it is for the inspection to find.
2. **reason_step** ("is there enough information?"), after each inspection. It rules on every check that
   refuted a need (`keep` the evidence, or `reject` it and reopen the need), then chooses one action:
   `inspect` a need, `add_need`, `revise_need`, `drop_need`, `revise_hypothesis` (once), or `synthesize`.
   Code enforces the rules: no synthesis while a need is open or a refuting check is not ruled on, at most
   3 inspections per need since its last revision, a revised need is inspected before it is revised again,
   and the last need cannot be dropped. An action refused twice in a row for the same reason is called out.
3. **Inspect** (`inspector.py`). An Inspector establishes one need in up to 12 turns of code (in a block after its JSON reply) in a shared, sandboxed notebook, using the `sheet` tools
   (`tools/sheet.py`). It looks first and extracts after,
   then sets `evidence_<need>` = {value, cells, where, note}. Errors are observations for the next turn;
   the last turn asks for the evidence, and when the turns are used up without it, one report turn sets it
   from what was printed. An attempt that found nothing hands its last thoughts to the next attempt. A number
   read from one cell is reported as the cell's number format shows it.
4. **Check** (`checker.py`, settings below). Checkers test the evidence of a found need. A refuting check
   is shown in full to the next reasoning step, which judges it; once ruled on, it shows in one line.
5. **Synthesize** (`synthesizer.py`). Code sets `answer_parts`, one entry per slot. The runner checks the
   count and joins the parts into the answer. After a return, it is told why.
6. **reason_end**. Checks the answer against the question and the evidence. It may send the loop back to
   step 2 once.

Limits per question are in `ledger.py`: 20 steps, 80 model calls, 1 hypothesis revision, and 1 return.

## Tools (`tools/`)

`sheet` offers `sheets`, `find`, `read`, `frame`, `around` (the cells around one cell),
`headers` and `describe`, `formats` and `cells`, and `look` when a vision model is configured.

Formats reach the model through code, not through the preview (`tools/formats.py`). `sheet.cells(range)`
returns a DataFrame with one row per filled cell (a merged cell once): value, merged range, bold, italic,
fill and font colour with a colour name ("red"), indent, formula; `sheet.cells("C:C")` takes a whole column,
so cells are selected by format in code. The notebook also holds two variables found over every cell:
`legends` ({label: fill} for each fill-colour legend set apart from the data) and `formats` (every fill,
font colour, bold and italic in use, with how many cells carry it and in which columns). `sheet.formats`
lists the same per cell as text.

`sheet.headers(cell)` (`tools/headers.py`) walks from a cell up to its column headers and left to its row
labels: the label cells above the data (a row of numbers typed plain above formatted data is a header row
too); keys written once and carried down (a column of numbers written once per group of rows is a key
column); label-only section rows above, each parent found by position, indent or bold; the block
of rows shaped like the cell's row and why it starts and ends (a blank or label row, a total by formula, a
rare format, a change of the kind of data). When the walk is unsure (no header, no label), it asks the vision model about an image of the area, at most 3 times per question.

## Settings (config `table_agent.qa_*`)

- `qa_checker`: the checks of each found need, in order; a later one runs only when the earlier ones did not
  confirm. None by default.
  - `reverse`: from the cells back to their meaning. The runner walks from each cell to its headers, reads
    the cells, lists the filled cells of the same sections that were left out and the cells shared with
    other needs; one model call compares this with the need.
  - `independent`: re-derives the need in its own notebook with other tools. `qa_checker_view` picks them:
    `grid` (raw cells with `sheet.formats` for their format and formulas),
    `image` (images through the vision model) or `frame` (DataFrames only); `grid` and `frame` also get the
    workbook preview the first reasoning step saw, on the first turn only. The check's notebook holds `cells`
    (as `sheet.cells`), `legends` and `formats`. With `qa_checker_sees_value` (default true) it sees the claim and gives the verdict; blind, it
    reports its own value and the runner compares the two.
  - `debate`: argues against the evidence in its own notebook; each objection must name the cells that show it.
- `qa_synthesis_check`: before reason_end, a Checker reviews how the synthesis code computed the answer
  (operation, filters, form of each part).

Their model calls count toward the same limit of 80.

If the exploration fails, `workbook_qa.py` answers with one direct model call. It uses the needs that were
found and that no check refuted (or whose refuting check was ruled wrong) when there are any; otherwise it
uses the source context.

## Artifacts of a run (`utils/trace.py`)

Each question gets one directory:

- `trace.md` is the readable story: the abstract reasoning at the start, every step with its rulings and its
  inspections (turns, evidence, checks), the synthesis with its method check, and the abstract reasoning at the end.
- `ledger.json` is the full ledger.
- `events.jsonl` lists every event in order, with prompts and model calls.
- `events.log` shows the same events to read: a line per model call, a block per other event, prompts left out.
- `result.json` holds the outcome.
- `cells/` holds each notebook cell's code, with what it printed as comments. Checker cells are labelled
  `<need>_attempt<n>_<check>_turn<t>`.
- `images/` holds the rendered ranges looked at.
