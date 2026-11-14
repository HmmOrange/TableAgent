"""The formats in use in a workbook, found in code over every cell: fill-colour legends and format marks.

The preview shows no formats, so they reach the model through code: a short prompt line that says they exist,
and the notebook variables `legends` and `formats`; cells are then selected by format with `sheet.cells`.
"""

from __future__ import annotations

from typing import Any

from TableAgent.src.tools.headers import WHITE_FILLS, cell_format, colour_name, font_colour, legend_cells, text_of


def legends(sheets: list[Any]) -> list[dict[str, Any]]:
    """The fill-colour legends set apart from the data: sheet, label, cell, fill, and per column the number cells with that fill."""
    found = []
    for ws in sheets:
        labels = legend_cells(ws)
        columns: dict[str, dict[str, int]] = {fill: {} for fill in labels}
        for row in ws.iter_rows():
            for cell in row:
                if _is_number(cell.value):
                    fill = cell_format(cell)[1]
                    if fill in columns:
                        columns[fill][cell.column_letter] = columns[fill].get(cell.column_letter, 0) + 1
        for fill, label in labels.items():
            beside = [ws.cell(label.row, column).value for column in (label.column - 1, label.column + 1) if column >= 1]
            if any(value is not None for value in beside):
                continue  # a label inside a table row is a shaded row or column, not a legend set apart from the data
            found.append({"sheet": ws.title, "label": text_of(label.value), "cell": label.coordinate, "fill": fill, "columns": columns[fill]})
    return found


def legend_fills(found: list[dict[str, Any]]) -> dict[str, str]:
    """The legends as the notebook variable `legends`: {label: fill}."""
    return {item["label"]: item["fill"] for item in found}


def legend_names(found: list[dict[str, Any]]) -> str:
    """The legend labels with their cells, for a prompt line."""
    return ", ".join(f'"{item["label"]}" ({item["cell"]})' for item in found)


def legend_notes(found: list[dict[str, Any]]) -> str:
    """The legends as prompt lines."""
    return "\n".join(
        f"- sheet '{item['sheet']}': \"{item['label']}\" ({item['cell']}) has fill {item['fill']}; number cells with this fill, by column: "
        + ", ".join(f"{letter} ({count})" for letter, count in sorted(item["columns"].items(), key=lambda pair: (len(pair[0]), pair[0])))
        for item in found
    )


def format_marks(sheets: list[Any]) -> list[dict[str, Any]]:
    """The fills, font colours, bold and italic in use on each sheet, as the notebook variable `formats`: sheet, kind,
    value, colour (a basic colour word for an RGB value), numbers and labels (cells carrying it), columns, and legend
    (the label set apart that names it)."""
    named = {(item["sheet"], item["fill"]): f'{item["label"]} ({item["cell"]})' for item in legends(sheets)}
    marks: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for ws in sheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.value in (None, ""):
                    continue
                bold, fill, _ = cell_format(cell)
                kinds = (
                    ("fill", None if fill in WHITE_FILLS else fill), ("font", font_colour(cell)),
                    ("bold", bold or None), ("italic", bool(cell.font and cell.font.i) or None),
                )
                for kind, value in kinds:
                    if value is None:
                        continue
                    mark = marks.setdefault((ws.title, kind, value), {
                        "sheet": ws.title, "kind": kind, "value": value, "colour": colour_name(value), "numbers": 0, "labels": 0,
                        "columns": [], "legend": named.get((ws.title, value)) if kind == "fill" else None,
                    })
                    mark["numbers" if _is_number(cell.value) else "labels"] += 1
                    if cell.column_letter not in mark["columns"]:
                        mark["columns"].append(cell.column_letter)
    return list(marks.values())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
