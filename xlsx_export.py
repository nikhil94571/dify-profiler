import io
import json
from typing import Any, Dict, List, Tuple, Union

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation



def _normalize_rows(obj: Any) -> Tuple[List[str], List[List[Any]]]:
    """
    Accepts:
      - list[dict]  -> headers from keys (deterministic order)
      - dict with {"columns": [...], "rows": [...]}
      - list[list]  -> assumes first row is header if all strings, else generates Col1..ColN
    Returns (headers, rows)
    """
    if isinstance(obj, dict) and "columns" in obj and "rows" in obj:
        cols = [str(c) for c in obj["columns"]]
        rows = obj["rows"]
        if not isinstance(rows, list):
            raise ValueError("rows must be a list")
        return cols, [list(r) if isinstance(r, (list, tuple)) else [r] for r in rows]

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # deterministic header order: keys in first row, then any new keys in later rows sorted
        first_keys = list(obj[0].keys())
        extra_keys = sorted({k for r in obj[1:] for k in (r.keys() if isinstance(r, dict) else [])} - set(first_keys))
        headers = [str(k) for k in (first_keys + extra_keys)]
        rows = [[r.get(h, "") for h in headers] for r in obj]
        return headers, rows

    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        # header detection
        first = obj[0]
        if all(isinstance(x, str) for x in first):
            headers = [str(x) for x in first]
            rows = obj[1:]
        else:
            headers = [f"Col{i+1}" for i in range(len(first))]
            rows = obj
        return headers, rows

    # empty or unknown
    return [], []

HEADER_FIELDS: Dict[str, str] = {
    "dataset_notes": (
        "FREE TEXT (recommended): Describe the dataset and intended structure.\n"
        "Examples:\n"
        "- 'Columns 1–12 represent Questions 1–12 of the OCQ scale.'\n"
        "- 'Each participant has multiple visits; visit columns are labelled Visit1/Visit2.'\n"
        "- 'These are survey waves; long format should use (participant_id, wave) as keys.'\n"
    ),
    "global_override_instructions": (
        "OPTIONAL FREE TEXT: Any global rename/reshape instruction.\n"
        "Examples:\n"
        "- 'Treat baseline/week4/week8 as timepoints.'\n"
        "- 'Do NOT pivot demographic columns.'\n"
    ),
    "group_definitions": (
        "OPTIONAL: Define families of variables (blocks/scales) in free text.\n"
        "Examples:\n"
        "- 'OCQ: Q1–Q12; total in ocq_total'\n"
        "- 'PHQ9: phq_1..phq_9; total in phq_total'\n"
    ),
    "group_regex_rules": (
        "OPTIONAL (advanced): Provide regex rules for repeated patterns.\n"
        "Examples:\n"
        "- 'OCQ_Q(\\d+)' or 'ocq_(\\d+)' or 'Q(\\d+)'\n"
        "Keep small; the next step will validate/interpret.\n"
    ),
}






def build_xlsx_bytes(rows_table_json: str, sheet_name: str = "Template") -> bytes:
    parsed = json.loads(rows_table_json)
    headers, rows = _normalize_rows(parsed)

    wb = Workbook()

    # -----------------------
    # Sheet 1: Overrides
    # -----------------------
    ws_over = wb.active
    ws_over.title = "Overrides"

    ws_over.append(["key", "instructions", "your_input"])
    for c in ws_over[1]:
        c.font = Font(bold=True)

    start_row = 2
    for i, (k, help_text) in enumerate(HEADER_FIELDS.items()):
        r = start_row + i
        ws_over.cell(row=r, column=1, value=k)
        ws_over.cell(row=r, column=2, value=help_text).alignment = Alignment(wrap_text=True, vertical="top")
        ws_over.cell(row=r, column=3, value="").alignment = Alignment(wrap_text=True, vertical="top")
        ws_over.row_dimensions[r].height = 120  # big free-text space

    ws_over.freeze_panes = "A2"
    ws_over.column_dimensions["A"].width = 28
    ws_over.column_dimensions["B"].width = 80
    ws_over.column_dimensions["C"].width = 60

    # -----------------------
    # Sheet 2: Contract table
    # -----------------------
    ws = wb.create_sheet(title=(sheet_name[:31] if sheet_name else "Template"))

    if headers:
        # Header row (Row 1)
        ws.append(headers)
        for cell in ws[1]:
            cell.font = Font(bold=True)  # (E) header styling
            cell.alignment = Alignment(wrap_text=True, vertical="top")

        # Field guide row:
        # Only add our generic guide if the incoming data does NOT already include a FIELD GUIDE row.
        incoming_has_guide = False
        if rows:
            first_row = list(rows[0]) if isinstance(rows[0], (list, tuple)) else [rows[0]]
            if first_row and isinstance(first_row[0], str) and first_row[0].strip().upper().startswith("FIELD GUIDE"):
                incoming_has_guide = True

        if not incoming_has_guide:
            guide = ["FIELD GUIDE (row 2): fill rows 3+; use dropdowns where present; keep values consistent."]
            if len(headers) > 1:
                guide += [""] * (len(headers) - 1)
            ws.append(guide)
            ws[2][0].alignment = Alignment(wrap_text=True, vertical="top")

        # Freeze header + the (real) guide row
        ws.freeze_panes = "A3"

        # (C) Autofilter on header row
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
    else:
        # No headers -> just dump rows; no formatting/filters
        ws.freeze_panes = None

    # (A) Remove pandas index artifact rows if we can detect them
    if headers and "original_column" in headers:
        oc_idx = headers.index("original_column")
        filtered_rows = []
        for r in rows:
            rr = list(r) if isinstance(r, (list, tuple)) else [r]
            oc_val = "" if oc_idx >= len(rr) else str(rr[oc_idx]).strip()
            if oc_val == "Unnamed: 0" or oc_val.startswith("Unnamed:"):
                continue
            filtered_rows.append(rr)
        rows = filtered_rows

    # Write data starting row 3 (after header + guide)
    for r in rows:
        ws.append(list(r) if isinstance(r, (list, tuple)) else [r])

    # (D) Dropdown validation for boolean-ish fields + include
    if headers:
        col_letter = {h: get_column_letter(i + 1) for i, h in enumerate(headers)}
        last_row = max(ws.max_row, 3)

        bool_fields = ["is_primary_id", "is_time", "reshape_exclude", "is_multi_select"]
        if any(f in col_letter for f in bool_fields):
            dv_bool = DataValidation(type="list", formula1='"TRUE,FALSE"', allow_blank=True)
            ws.add_data_validation(dv_bool)
            for f in bool_fields:
                if f in col_letter:
                    L = col_letter[f]
                    dv_bool.add(f"{L}3:{L}{last_row}")

        if "include" in col_letter:
            dv_yesno = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
            ws.add_data_validation(dv_yesno)
            L = col_letter["include"]
            dv_yesno.add(f"{L}3:{L}{last_row}")

    # light autosize (cap width to avoid silly columns)
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        max_len = 0
        for cell in ws[letter]:
            v = "" if cell.value is None else str(cell.value)
            if len(v) > max_len:
                max_len = len(v)
        ws.column_dimensions[letter].width = min(max(10, max_len + 2), 60)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

