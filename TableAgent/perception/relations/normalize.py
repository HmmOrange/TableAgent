from __future__ import annotations

import re
from openpyxl.utils.cell import column_index_from_string, range_boundaries

ref_regex = re.compile(
    r"((?:'(?:[^']|'')+'!|[A-Za-z0-9_]+!)?\$?[A-Z]+\$?[0-9]+:\$?[A-Z]+\$?[0-9]+|"
    r"(?:'(?:[^']|'')+'!|[A-Za-z0-9_]+!)?\$?[A-Z]+\$?[0-9]+)"
)


def parse_reference(ref_str: str) -> tuple[str | None, int, int, int, int]:
    if "!" in ref_str:
        sheet_part, cell_part = ref_str.split("!", 1)
        sheet_name = sheet_part.strip("'")
    else:
        sheet_name = None
        cell_part = ref_str

    min_col, min_row, max_col, max_row = range_boundaries(cell_part)
    min_col = min_col if min_col is not None else 1
    min_row = min_row if min_row is not None else 1
    max_col = max_col if max_col is not None else 16384
    max_row = max_row if max_row is not None else 1048576
    return sheet_name, min_col, min_row, max_col, max_row


def parse_cell_absolute(cell_str: str) -> tuple[str, bool, int, bool]:
    col_part = "".join(c for c in cell_str if c.isalpha() or c == "$")
    row_part = "".join(c for c in cell_str if c.isdigit() or c == "$")
    col_abs = col_part.startswith("$")
    row_abs = row_part.startswith("$")
    col_letter = col_part.replace("$", "")
    row_num = int(row_part.replace("$", ""))
    return col_letter, col_abs, row_num, row_abs


def get_normalized_formula(formula_raw: str, r_cell: int, c_cell: int) -> str:
    strings = []
    def replace_str(match):
        strings.append(match.group(0))
        return f"__STR_{len(strings)-1}__"
    formula_no_str = re.sub(r'"[^"]*"', replace_str, formula_raw)

    matches = list(ref_regex.finditer(formula_no_str))
    parts = list(formula_no_str)

    for m in reversed(matches):
        ref_str = m.group(0)
        sheet_name, min_col, min_row, max_col, max_row = parse_reference(ref_str)
        cell_part = ref_str.split("!", 1)[1] if "!" in ref_str else ref_str
        if ":" in cell_part:
            c1_str, c2_str = cell_part.split(":", 1)
            col1_letter, col1_abs, row1_num, row1_abs = parse_cell_absolute(c1_str)
            col2_letter, col2_abs, row2_num, row2_abs = parse_cell_absolute(c2_str)

            c1_col_norm = f"${col1_letter}" if col1_abs else f"col_offset={column_index_from_string(col1_letter) - c_cell}"
            c1_row_norm = f"${row1_num}" if row1_abs else f"row_offset={row1_num - r_cell}"
            c2_col_norm = f"${col2_letter}" if col2_abs else f"col_offset={column_index_from_string(col2_letter) - c_cell}"
            c2_row_norm = f"${row2_num}" if row2_abs else f"row_offset={row2_num - r_cell}"

            norm_str = f"[{c1_col_norm},{c1_row_norm}]:[{c2_col_norm},{c2_row_norm}]"
        else:
            col_letter, col_abs, row_num, row_abs = parse_cell_absolute(cell_part)
            col_norm = f"${col_letter}" if col_abs else f"col_offset={column_index_from_string(col_letter) - c_cell}"
            row_norm = f"${row_num}" if row_abs else f"row_offset={row_num - r_cell}"
            norm_str = f"[{col_norm},{row_norm}]"

        if sheet_name:
            norm_str = f"{sheet_name}!{norm_str}"

        start, end = m.span()
        parts[start:end] = list(norm_str)

    normalized_formula_no_str = "".join(parts)
    for i, s in enumerate(strings):
        normalized_formula_no_str = normalized_formula_no_str.replace(f"__STR_{i}__", s)
    return normalized_formula_no_str


def get_pattern(formula_raw: str, r_cell: int, c_cell: int, is_vertical: bool) -> str:
    formula_body = formula_raw.lstrip("=")
    strings = []
    def replace_str(match):
        strings.append(match.group(0))
        return f"__STR_{len(strings)-1}__"
    formula_no_str = re.sub(r'"[^"]*"', replace_str, formula_body)

    matches = list(ref_regex.finditer(formula_no_str))
    parts = list(formula_no_str)

    for m in reversed(matches):
        ref_str = m.group(0)
        sheet_name, min_col, min_row, max_col, max_row = parse_reference(ref_str)
        cell_part = ref_str.split("!", 1)[1] if "!" in ref_str else ref_str
        if ":" in cell_part:
            c1_str, c2_str = cell_part.split(":", 1)
            col1_letter, col1_abs, row1_num, row1_abs = parse_cell_absolute(c1_str)
            col2_letter, col2_abs, row2_num, row2_abs = parse_cell_absolute(c2_str)

            if is_vertical:
                pat1 = f"{col1_letter}{'{row}'}" if not row1_abs else c1_str
                pat2 = f"{col2_letter}{'{row}'}" if not row2_abs else c2_str
            else:
                pat1 = f"{'{col}'}{row1_num}" if not col1_abs else c1_str
                pat2 = f"{'{col}'}{row2_num}" if not col2_abs else c2_str
            replacement = f"{pat1}:{pat2}"
        else:
            col_letter, col_abs, row_num, row_abs = parse_cell_absolute(cell_part)
            if is_vertical:
                replacement = f"{col_letter}{'{row}'}" if not row_abs else cell_part
            else:
                replacement = f"{'{col}'}{row_num}" if not col_abs else cell_part

        if sheet_name:
            replacement = f"{sheet_name}!{replacement}"

        start, end = m.span()
        parts[start:end] = list(replacement)

    pattern_body = "".join(parts)
    for i, s in enumerate(strings):
        single_quoted = s.replace('"', "'")
        pattern_body = pattern_body.replace(f"__STR_{i}__", single_quoted)

    from openpyxl.utils.cell import get_column_letter
    col_letter = get_column_letter(c_cell)
    if is_vertical:
        lhs = f"{col_letter}{'{row}'}"
    else:
        lhs = f"{'{col}'}{r_cell}"
    return f"{lhs} = {pattern_body}"


def normalize_expression_spacing(expr: str) -> str:
    strings = []
    def replace_str(match):
        strings.append(match.group(0))
        return f"__STR_{len(strings)-1}__"
    expr_no_str = re.sub(r"'[^']*'", replace_str, expr)

    for op in [">=", "<=", ">", "<", "=", "*", "-", "+", "/"]:
        expr_no_str = expr_no_str.replace(op, f" {op} ")

    expr_no_str = expr_no_str.replace(",", ", ")

    expr_no_str = re.sub(r"\s+", " ", expr_no_str)
    expr_no_str = expr_no_str.replace(" > = ", " >= ")
    expr_no_str = expr_no_str.replace(" < = ", " <= ")
    expr_no_str = expr_no_str.replace(" (", "(").replace("( ", "(")
    expr_no_str = expr_no_str.replace(" )", ")").replace(") ", ")")
    expr_no_str = expr_no_str.replace(" , ", ", ").replace(" ,", ", ").replace(", ", ", ")

    expr_no_str = re.sub(r"\b(SUM|AVERAGE|MIN|MAX|COUNT|COUNTA|SUBTOTAL|IF)\s*\(", r"\1(", expr_no_str, flags=re.IGNORECASE)

    for i, s in enumerate(strings):
        expr_no_str = expr_no_str.replace(f"__STR_{i}__", s)
    return expr_no_str.strip()
