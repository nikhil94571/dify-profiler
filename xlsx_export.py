import io
import json
from typing import Any, Dict, List, Tuple, Union

from openpyxl import Workbook
from openpyxl.utils import get_column_letter


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


def build_xlsx_bytes(rows_table_json: str, sheet_name: str = "Template") -> bytes:
    parsed = json.loads(rows_table_json)
    headers, rows = _normalize_rows(parsed)

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name[:31] if sheet_name else "Template"

    if headers:
        ws.append(headers)
        ws.freeze_panes = "A2"

    for r in rows:
        ws.append(list(r) if isinstance(r, (list, tuple)) else [r])

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
