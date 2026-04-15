from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple
import io
import os
import hashlib
import time
import json
import uuid
import logging
import re
import copy
from pathlib import Path
from collections import defaultdict, deque, Counter
from itertools import combinations
from datetime import timedelta
from uuid import uuid4
from urllib.parse import urlencode
from manifest_export import upload_and_sign_text
from canonical_contract_invariants import (
    FAMILY_DEFAULT_ALLOWED_MISSINGNESS_DISPOSITIONS,
    POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS as SHARED_POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS,
    find_row_invariant_errors as _find_canonical_row_invariant_errors,
    normalize_missingness_decision as _normalize_canonical_missingness_decision,
)

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from google.cloud import storage
import google.auth
from google.auth import impersonated_credentials


from xlsx_export import (
    DEFAULT_OVERRIDE_FIELDS,
    DEFAULT_SCALE_MAPPING_HEADERS,
    build_light_contract_xlsx_bytes,
    build_xlsx_bytes,
    parse_light_contract_xlsx_bytes,
)


try:
    from dateutil import parser as dateparser  # optional fallback
except Exception:
    dateparser = None

try:
    from pypdf import PdfReader  # optional for codebook extraction
except Exception:
    PdfReader = None

app = FastAPI()
bearer = HTTPBearer(auto_error=True)

# --- Config ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))  # 20MB default

RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))          # requests
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))    # seconds

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

logger = logging.getLogger("profiler")

# Stage logger: goes to uvicorn stderr / Cloud Run logs
stage_logger = logging.getLogger("uvicorn.error")
stage_logger.setLevel(logging.INFO)

# ip -> timestamps
_req_times = defaultdict(deque)

# -----------------------------
# Profiler v2 deterministic limits (Step 0)
# -----------------------------
SAMPLE_SIZE_HEAD = int(os.getenv("SAMPLE_SIZE_HEAD", "10"))
SAMPLE_SIZE_TAIL = int(os.getenv("SAMPLE_SIZE_TAIL", "10"))
SAMPLE_SIZE_RANDOM = int(os.getenv("SAMPLE_SIZE_RANDOM", "30"))
TOP_K_FREQUENT = int(os.getenv("TOP_K_FREQUENT", "20"))
MAX_STRING_LEN = int(os.getenv("MAX_STRING_LEN", "80"))

# Deterministic cap for expensive scans (extra, but important for service stability)
PATTERN_SCAN_MAX_ROWS = int(os.getenv("PATTERN_SCAN_MAX_ROWS", "50000"))

# Deterministic hybrid scan: head + tail + seeded random from the remainder
# (Decision) Default head/tail are small relative to max to preserve speed while catching tail-only patterns.
# Why it matters: avoids missing tail-end formatting changes without introducing nondeterminism.
PATTERN_SCAN_HEAD_ROWS = int(os.getenv("PATTERN_SCAN_HEAD_ROWS", "2000"))
PATTERN_SCAN_TAIL_ROWS = int(os.getenv("PATTERN_SCAN_TAIL_ROWS", "2000"))


# Limits for examples emitted per pattern category
EXAMPLES_PER_PATTERN = int(os.getenv("EXAMPLES_PER_PATTERN", "5"))
RARE_EXAMPLES_MAX = int(os.getenv("RARE_EXAMPLES_MAX", "10"))
EXTREMES_N = int(os.getenv("EXTREMES_N", "5"))

# -----------------------------
# Association/Dependency evidence limits (Dify-safe)
# -----------------------------
ASSOC_MAX_ROWS = int(os.getenv("ASSOC_MAX_ROWS", "50000"))                 # cap rows scanned
ASSOC_TOP_K_PAIRS = int(os.getenv("ASSOC_TOP_K_PAIRS", "40"))              # top pairs returned
ASSOC_MAX_NUMERIC_COLS = int(os.getenv("ASSOC_MAX_NUMERIC_COLS", "30"))    # cap numeric cols for pairwise corr
ASSOC_MAX_CAT_COLS = int(os.getenv("ASSOC_MAX_CAT_COLS", "30"))            # cap categorical cols for pairwise assoc
ASSOC_MAX_CAT_CARD = int(os.getenv("ASSOC_MAX_CAT_CARD", "50"))            # max unique values for categorical assoc
ASSOC_MIN_OVERLAP = int(os.getenv("ASSOC_MIN_OVERLAP", "40"))              # min non-null overlap for pairwise stats
ASSOC_TOL_ABS = float(os.getenv("ASSOC_TOL_ABS", "0.01"))                  # tolerance for arithmetic identity checks
ASSOC_TOL_REL = float(os.getenv("ASSOC_TOL_REL", "0.005"))                 # relative tolerance for arithmetic checks


# Missing-like tokens (Step 2)
MISSING_LIKE_TOKENS = [
    "N/A", "NA", "null", "None", "", "-", "—", "?", "Unknown"
]


def require_token(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> None:
    expected = os.getenv("PROFILER_API_KEY")
    token = creds.credentials
    if not expected or token != expected:
        raise HTTPException(status_code=401, detail="Invalid token")


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# -----------------------------
# Request logging middleware
# -----------------------------
@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.time()
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid

    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-Id"] = rid
        return response

    except HTTPException as exc:
        status = exc.status_code
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
        response.headers["X-Request-Id"] = rid
        return response

    except Exception as exc:
        status = 500
        logger.exception(json.dumps({
            "event": "request_exception",
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }))
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )
        response.headers["X-Request-Id"] = rid
        return response

    finally:
        duration_ms = int((time.time() - start) * 1000)
        logger.info(json.dumps({
            "event": "request",
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": duration_ms,
        }))



@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail", "/evidence_associations", "/export/light-contract-xlsx", "/light-contracts/xlsx", "/full-bundle") and request.method.upper() == "POST":
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload too large")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Content-Length")
    return await call_next(request)


@app.middleware("http")
async def basic_rate_limit(request: Request, call_next):
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail", "/evidence_associations", "/export/light-contract-xlsx", "/light-contracts/xlsx", "/full-bundle") and request.method.upper() == "POST":
        ip = _client_ip(request)
        now = time.time()
        q = _req_times[ip]

        cutoff = now - RATE_LIMIT_WINDOW
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= RATE_LIMIT_MAX:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        q.append(now)

    return await call_next(request)


# -----------------------------
# Profiling helpers
# -----------------------------
def infer_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "text"


def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _stable_int_seed(dataset_sha256: str, col: str) -> int:
    h = hashlib.sha256(f"{dataset_sha256}::{col}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit seed


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 1] + "…"


def _stringify_value(v: Any) -> Optional[str]:
    # Preserve missing as None (not the string "nan")
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v)
    return _truncate(s, MAX_STRING_LEN)


def _non_null_series(s: pd.Series) -> pd.Series:
    return s[~s.isna()]


def _as_str_series(s: pd.Series) -> pd.Series:
    # Convert to pandas StringDtype to avoid "nan" string artifacts.
    # Missing stays <NA>.
    return s.astype("string")


def _cap_scan(
    s: pd.Series,
    max_rows: int,
    *,
    seed: int,
    head_rows: int,
    tail_rows: int,
) -> pd.Series:
    """
    Deterministic hybrid sampler:
      - take head chunk
      - take tail chunk
      - take seeded random sample from the middle remainder
    Preserves reproducibility via a stable seed.
    """
    n = len(s)
    if n <= max_rows:
        return s

    # Guardrails: keep within [0, max_rows] and avoid negative middle sizes
    head_n = max(0, min(int(head_rows), max_rows))
    tail_n = max(0, min(int(tail_rows), max_rows - head_n))

    # If max_rows is tiny, the head already fills it
    if head_n >= max_rows:
        return s.iloc[:max_rows]

    # Split into head / middle / tail
    head = s.iloc[:head_n]
    tail = s.iloc[-tail_n:] if tail_n > 0 else s.iloc[:0]

    mid_start = head_n
    mid_end = max(mid_start, n - tail_n)
    middle = s.iloc[mid_start:mid_end]

    remaining = max_rows - head_n - tail_n
    if remaining <= 0 or len(middle) == 0:
        out = pd.concat([head, tail])
    else:
        rnd_n = min(remaining, len(middle))
        rnd = middle.sample(n=rnd_n, random_state=seed)
        out = pd.concat([head, tail, rnd])

    # Deterministic de-dupe while preserving first-seen order
    out = out[~out.duplicated()]

    # Final safety cap (should already be <= max_rows)
    if len(out) > max_rows:
        out = out.iloc[:max_rows]

    return out



# -----------------------------
# Pattern regexes (Step 3)
# -----------------------------
RE_STRICT_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
RE_CURRENCY_PREFIX = re.compile(r"^[€$£]\s*\d")
RE_PERCENT = re.compile(r"\d+(?:\.\d+)?%")
RE_THOUSANDS = re.compile(r"\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?")
RE_PARENS_NEG = re.compile(r"^\(\s*\d")

# Multipliers are a distinct concept from measurement units
MULTIPLIER_SUFFIXES = {"k", "m", "b"}
RE_SUFFIX_MULT = re.compile(r"(?i)\b-?\d+(?:\.\d+)?\s*([KMB])\b")

# Measurement-unit suffixes (first-class, separate from multipliers)
UNIT_SUFFIXES = {
    # weight
    "kg", "kgs", "kilogram", "kilograms",
    "g", "gram", "grams",
    "mg", "milligram", "milligrams",
    "lb", "lbs", "pound", "pounds",

    # height / length
    "mm", "cm", "m",
    "ft", "feet", "in", "inch", "inches",

    # time / duration (psych: reaction time, exposure duration, session length)
    "ms", "msec", "msecs", "millisecond", "milliseconds",
    "s", "sec", "secs", "second", "seconds",
    "min", "mins", "minute", "minutes",
    "h", "hr", "hrs", "hour", "hours",
    "d", "day", "days",
    "wk", "wks", "week", "weeks",

    # age / longitudinal survey units
    "yr", "yrs", "year", "years",
    "mo", "mos", "month", "months",
}


# numeric + optional whitespace + unit
UNIT_SUFFIX_RE = re.compile(
    r"^\s*(?P<num>-?\d+(?:\.\d+)?)\s*(?P<unit>("
    + "|".join(sorted(UNIT_SUFFIXES, key=len, reverse=True))
    + r"))\s*$",
    re.IGNORECASE
)

# URL / identifier strong text signals
RE_URL = re.compile(r"(?i)\bhttps?://")
RE_EMAIL = re.compile(r"(?i)^[^@\s]+@[^@\s]+\.[^@\s]+$")
RE_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")

# Bool-like tokens (for elimination scoring)
BOOL_LIKE = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"}

RE_ISO_DATE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}")
RE_DMY_MDY = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
RE_YEAR_ONLY = re.compile(r"^\d{4}$")
RE_MONTH_NAME = re.compile(
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b",
    re.IGNORECASE
)
RE_YEAR_RANGE = re.compile(r"^\s*\d{4}\s*~\s*\d{4}\s*$")
RE_NUM_RANGE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*-\s*-?\d+(?:\.\d+)?\s*$")



def _pct(n: int, d: int) -> float:
    return round((n / d * 100.0) if d else 0.0, 6)

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _take_examples(masked_values: pd.Series, limit: int) -> List[str]:
    out: List[str] = []
    for v in masked_values.head(limit).tolist():
        sv = _stringify_value(v)
        if sv is not None:
            out.append(sv)
    return out


# -----------------------------
# Parsing helpers for profiling (Step 5)
# -----------------------------
def _try_parse_numeric_for_profile(s_stripped: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (parsed_numeric_series, success_mask)

    Profiling-only numeric parser:
    - preserves existing behaviors (currency, %, thousands, parens-negative, K/M/B)
    - adds measurement-unit suffix recognition as a first-class parse mode:
        e.g. "72kg", "185lbs", "1.82m", "182 cm", "6ft", "72 in"
    NOTE: this returns a numeric value in the *original unit scale* (not converted),
    because quantiles/extremes are still useful, and canonical conversion is handled
    via unit_plan (separate).
    """
    x = s_stripped.copy()

    # Normalize: parens negative (123) -> -123
    paren_mask = x.str.match(r"^\(\s*.*\)\s*$", na=False)
    x = x.where(~paren_mask, "-" + x.str.replace(r"^\(\s*|\)\s*$", "", regex=True))

    # Strip currency prefix symbols
    x = x.str.replace(r"^[€$£]\s*", "", regex=True)

    # Strip trailing percent
    x = x.str.replace(r"%\s*$", "", regex=True)

    # Measurement-unit suffix parse mode (extract numeric part if unit suffix)
    # We deliberately do NOT convert here; conversion is driven by unit_plan.
    unit_extract = x.str.extract(UNIT_SUFFIX_RE)
    unit_has = unit_extract["num"].notna() & unit_extract["unit"].notna()
    unit_num = pd.to_numeric(unit_extract["num"], errors="coerce").astype("float64")

    # Capture suffix multiplier K/M/B (separate from measurement units)
    mult_extract = x.str.extract(r"(?i)^\s*(-?\d+(?:\.\d+)?)\s*([KMB])\s*$")
    has_mult = mult_extract[0].notna() & mult_extract[1].notna()
    base_num = pd.to_numeric(mult_extract[0], errors="coerce").astype("float64")
    mult = mult_extract[1].str.upper().map({"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}).astype("float64")
    mult_values = (base_num * mult).astype("float64")

    # For non-suffix values, remove thousands separators between digit groupings
    x2 = x.str.replace(r"(?<=\d)[,\s](?=\d{3}\b)", "", regex=True)
    parsed_plain = pd.to_numeric(x2, errors="coerce").astype("float64")

    # Priority: unit-suffix numeric > multiplier numeric > plain numeric
    parsed = parsed_plain.where(~has_mult, mult_values)
    parsed = parsed.where(~unit_has, unit_num)

    success = parsed.notna()
    return parsed.astype("float64"), success



def _try_parse_date_for_profile(s_stripped: pd.Series) -> Tuple[pd.Series, pd.Series, str]:
    """
    Returns (parsed_datetime_series, success_mask, policy_used)

    Uses pandas to_datetime with two deterministic policies:
      - dayfirst=False
      - dayfirst=True (only if it yields more parses)
    """
    parsed_a = pd.to_datetime(s_stripped, errors="coerce", dayfirst=False)
    success_a = parsed_a.notna()

    parsed_b = pd.to_datetime(s_stripped, errors="coerce", dayfirst=True)
    success_b = parsed_b.notna()

    if success_b.sum() > success_a.sum():
        return parsed_b, success_b, "dayfirst_true"
    return parsed_a, success_a, "dayfirst_false"

def _run_full_profile(
    request: Request,
    file: UploadFile,
    dataset_id: str,
    max_categorical_cardinality: int,
) -> Dict[str, Any]:
    """
    Runs the full profiler and returns the complete data_profile dict.
    Used internally by all profile endpoints.
    """

# -----------------------------
# Shared runner (used by all endpoints)
# -----------------------------
async def _run_full_profile(
    request: Request,
    file: UploadFile,
    dataset_id: str,
    max_categorical_cardinality: int,
    *,
    include_samples: bool = True,
    include_pattern_examples: bool = True,
    include_deep_profiles: bool = True,
) -> Dict[str, Any]:
    logger.info(json.dumps({
        "event": "profile_start",
        "request_id": getattr(request.state, "request_id", None),
        "dataset_id": dataset_id,
    }))

    try:
        raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload too large")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    dataset_sha256 = sha256_hex(raw_bytes)

    # --- Stage logging: upload + read_csv ---
    rid = getattr(request.state, "request_id", None)
    filename = file.filename or "unknown"
    size_bytes = len(raw_bytes)

    stage_logger.info(
        "stage=upload_received request_id=%s bytes=%s filename=%s",
        rid, size_bytes, filename
    )

    t = time.time()
    stage_logger.info("stage=read_csv_start request_id=%s", rid)

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    except Exception as e:
        stage_logger.exception("stage=read_csv_error request_id=%s", rid)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    stage_logger.info(
        "stage=read_csv_end request_id=%s secs=%.2f shape=%s mem_mb=%.1f",
        rid,
        time.time() - t,
        df.shape,
        df.memory_usage(deep=True).sum() / 1e6
    )

    n_rows, n_cols = df.shape
    total_missing = int(df.isna().sum().sum())

    out: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "source_file": {
            "filename": file.filename,
            "content_type": file.content_type,
        },
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "column_names": list(df.columns),
        "summary": {
            "total_missing_cells": total_missing,
            "any_missing": bool(total_missing > 0),
        },
        "profiler_config": {
            "max_categorical_cardinality": int(max_categorical_cardinality),
            "sample_size_head": SAMPLE_SIZE_HEAD,
            "sample_size_tail": SAMPLE_SIZE_TAIL,
            "sample_size_random": SAMPLE_SIZE_RANDOM,
            "top_k_frequent": TOP_K_FREQUENT,
            "max_string_len": MAX_STRING_LEN,

            # Pattern scan sampling policy (Decision: deterministic hybrid scan).
            # Why it matters: makes confidence/pattern rates auditable and reproducible across runs.
            "pattern_scan_max_rows": PATTERN_SCAN_MAX_ROWS,
            "pattern_scan_head_rows": PATTERN_SCAN_HEAD_ROWS,
            "pattern_scan_tail_rows": PATTERN_SCAN_TAIL_ROWS,

            "examples_per_pattern": EXAMPLES_PER_PATTERN,
            "rare_examples_max": RARE_EXAMPLES_MAX,
            "extremes_n": EXTREMES_N,
            "missing_like_tokens": MISSING_LIKE_TOKENS,
        },

        "columns": {},
    }

    max_cat = int(max_categorical_cardinality)

    for col in df.columns:
        stage_logger.info("stage=col_start request_id=%s col=%s", rid, col)
        try:
            # --- EVERYTHING inside your current per-column loop stays the same ---
            # Keep your existing logic here unchanged:
            # - infer type
            # - samples
            # - missing-like tokens
            # - patterns
            # - candidates
            # - numeric_profile/date_profile
            #
            # At the end, you must set:
            # out["columns"][col] = col_info

            s = df[col]
            missing_count = int(s.isna().sum())
            missing_pct = (missing_count / n_rows * 100.0) if n_rows else 0.0
            unique_count = int(s.nunique(dropna=True))
            t = infer_type(s)

            col_info: Dict[str, Any] = {
                "inferred_type": t,
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 6),
                "unique_count": unique_count,
            }

            # Base numeric min/max for numeric dtype columns (existing behavior)
            if t in ("integer", "float"):
                non_na = s.dropna()
                col_info["min"] = float(non_na.min()) if len(non_na) else None
                col_info["max"] = float(non_na.max()) if len(non_na) else None

            # Levels for low-cardinality columns (existing behavior)
            if 0 < unique_count <= max_cat:
                vals = s.dropna().unique().tolist()
                if t in ("integer", "float"):
                    levels = sorted(float(v) for v in vals)
                    if t == "integer":
                        levels = [int(v) for v in levels]
                else:
                    levels = sorted(str(v) for v in vals)
                col_info["levels"] = levels

            # -----------------------------
            # Step 1 — Value sampling per column (optional)
            # -----------------------------
            if include_samples:
                s_non_na = _non_null_series(s)
                s_non_na_str = _as_str_series(s_non_na)

                # Deterministic head/tail of non-null values
                head_vals = s_non_na_str.head(SAMPLE_SIZE_HEAD)
                tail_vals = s_non_na_str.tail(SAMPLE_SIZE_TAIL)

                # Deterministic random sample (seeded by dataset hash + col name)
                seed = _stable_int_seed(dataset_sha256, col)
                if len(s_non_na_str) > 0:
                    rnd_n = min(SAMPLE_SIZE_RANDOM, len(s_non_na_str))
                    random_vals = s_non_na_str.sample(n=rnd_n, random_state=seed)
                else:
                    random_vals = s_non_na_str

                # Top frequent values (counts) from non-null stringified values
                vc = s_non_na_str.value_counts(dropna=False)
                top_vc = vc.head(TOP_K_FREQUENT)

                top_frequent = [
                    {"value": _stringify_value(idx), "count": int(cnt)}
                    for idx, cnt in top_vc.items()
                ]

                # Rare examples: values that occur once
                rare_vals = vc[vc == 1].index.to_series()
                rare_examples = [_stringify_value(v) for v in rare_vals.head(RARE_EXAMPLES_MAX).tolist()]
                rare_examples = [v for v in rare_examples if v is not None]

                # String length stats
                if len(s_non_na_str) > 0:
                    lens = s_non_na_str.str.len()
                    p95 = float(lens.quantile(0.95)) if len(lens) else None
                    col_info["string_length_stats"] = {
                        "min": int(lens.min()) if len(lens) else None,
                        "median": float(lens.median()) if len(lens) else None,
                        "p95": p95,
                        "max": int(lens.max()) if len(lens) else None,
                    }
                else:
                    col_info["string_length_stats"] = {"min": None, "median": None, "p95": None, "max": None}

                col_info["samples"] = {
                    "head": [_stringify_value(v) for v in head_vals.tolist()],
                    "tail": [_stringify_value(v) for v in tail_vals.tolist()],
                    "random": [_stringify_value(v) for v in random_vals.tolist()],
                    "top_frequent": top_frequent,
                    "rare_examples": rare_examples,
                }

            # -----------------------------
            # Step 2 — Missing-like token detection
            # -----------------------------
            # Work on full column string view (including empty strings) but excluding true NaN
            s_full_str = _as_str_series(s)
            s_full_str_non_na = s_full_str[~s_full_str.isna()]

            s_stripped = s_full_str_non_na.str.strip()

            missing_like_counts: Dict[str, int] = {}
            missing_like_total = 0

            # whitespace-only strings (after strip => "")
            whitespace_only_count = int((s_stripped == "").sum())
            if whitespace_only_count > 0:
                missing_like_counts["<WHITESPACE_ONLY>"] = whitespace_only_count
                missing_like_total += whitespace_only_count

            # explicit tokens (case-sensitive exact match after strip, except "" handled above)
            for tok in MISSING_LIKE_TOKENS:
                if tok == "":
                    continue
                cnt = int((s_stripped == tok).sum())
                if cnt > 0:
                    missing_like_counts[tok] = cnt
                    missing_like_total += cnt

            col_info["missing_like_tokens"] = missing_like_counts
            col_info["missing_like_pct"] = round((missing_like_total / n_rows * 100.0) if n_rows else 0.0, 6)

            # -----------------------------
            # Step 3 — Pattern classifiers (rule-based)
            # -----------------------------
            # Use deterministic hybrid scan for performance + tail coverage
            seed = _stable_int_seed(dataset_sha256, col)
            scan_series = _cap_scan(
                s_stripped,
                PATTERN_SCAN_MAX_ROWS,
                seed=seed,
                head_rows=PATTERN_SCAN_HEAD_ROWS,
                tail_rows=PATTERN_SCAN_TAIL_ROWS,
            )
            scan_n = int(len(scan_series))

            patterns: Dict[str, Any] = {}

            if scan_n == 0:
                # Still emit empty pattern blocks
                patterns["numeric_like"] = {
                    "strict_pct": 0.0,
                    "currency_pct": 0.0,
                    "suffix_pct": 0.0,
                    "percent_pct": 0.0,
                    "thousands_sep_pct": 0.0,
                    "parens_neg_pct": 0.0,
                }
                patterns["numeric_like_examples"] = {}
                patterns["date_like"] = {
                    "iso_pct": 0.0,
                    "month_name_pct": 0.0,
                    "parse_success_pct": 0.0,
                    "parse_policy": None,
                    "parse_failure_examples": [],
                }
                patterns["multi_value"] = {
                    "delimiter": None,
                    "multi_token_pct": 0.0,
                    "avg_tokens": 0.0,
                    "max_tokens": 0,
                    "token_vocab_topk": [],
                }
                patterns["range_like"] = {
                    "year_range_pct": 0.0,
                    "numeric_range_pct": 0.0,
                    "examples": [],
                }
            else:
                # Numeric-like
                strict_mask = scan_series.str.match(RE_STRICT_NUM, na=False)
                currency_mask = scan_series.str.match(RE_CURRENCY_PREFIX, na=False)

                # Multiplier suffix (K/M/B)
                suffix_mult_mask = scan_series.str.contains(RE_SUFFIX_MULT, na=False)

                # Measurement-unit suffix (kg/lb/cm/m/etc.)
                unit_match = scan_series.str.extract(UNIT_SUFFIX_RE)
                unit_suffix_mask = unit_match["num"].notna() & unit_match["unit"].notna()

                # suffix_pct counts EITHER multiplier suffix OR measurement-unit suffix
                suffix_any_mask = suffix_mult_mask | unit_suffix_mask

                percent_mask = scan_series.str.contains(RE_PERCENT, na=False)
                thousands_mask = scan_series.str.contains(RE_THOUSANDS, na=False)
                parens_mask = scan_series.str.match(RE_PARENS_NEG, na=False)

                # bool-like (for elimination-based text scoring)
                bool_mask = scan_series.str.lower().isin(BOOL_LIKE)

                numeric_like = {
                    "strict_pct": _pct(int(strict_mask.sum()), scan_n),
                    "currency_pct": _pct(int(currency_mask.sum()), scan_n),

                    # keep one suffix_pct for backwards compatibility + include breakdown
                    "suffix_pct": _pct(int(suffix_any_mask.sum()), scan_n),
                    "suffix_multiplier_pct": _pct(int(suffix_mult_mask.sum()), scan_n),
                    "suffix_unit_pct": _pct(int(unit_suffix_mask.sum()), scan_n),

                    "percent_pct": _pct(int(percent_mask.sum()), scan_n),
                    "thousands_sep_pct": _pct(int(thousands_mask.sum()), scan_n),
                    "parens_neg_pct": _pct(int(parens_mask.sum()), scan_n),
                    "bool_like_pct": _pct(int(bool_mask.sum()), scan_n),
                }

                # unit breakdown counts (for unit_plan + auditability)
                units_detected: Dict[str, int] = {}
                if scan_n > 0 and int(unit_suffix_mask.sum()) > 0:
                    # canonicalize unit tokens (lbs -> lb, kilograms -> kg, etc.)
                    raw_units = unit_match.loc[unit_suffix_mask, "unit"].astype("string").str.lower().tolist()

                    def _canon_unit(u: str) -> str:
                        u = (u or "").strip().lower()

                        # weight
                        if u in ("kgs", "kilogram", "kilograms"):
                            return "kg"
                        if u in ("g", "gram", "grams"):
                            return "g"
                        if u in ("mg", "milligram", "milligrams"):
                            return "mg"
                        if u in ("lbs", "pound", "pounds"):
                            return "lb"

                        # length
                        if u in ("feet",):
                            return "ft"
                        if u in ("inch", "inches"):
                            return "in"

                        # time
                        if u in ("msec", "msecs", "millisecond", "milliseconds"):
                            return "ms"
                        if u in ("sec", "secs", "second", "seconds"):
                            return "s"
                        if u in ("min", "mins", "minute", "minutes"):
                            return "min"
                        if u in ("hr", "hrs", "hour", "hours"):
                            return "h"
                        if u in ("day", "days"):
                            return "d"
                        if u in ("wk", "wks", "week", "weeks"):
                            return "wk"

                        # age / longitudinal
                        if u in ("yr", "yrs", "year", "years"):
                            return "yr"
                        if u in ("mo", "mos", "month", "months"):
                            return "mo"

                        return u

                    for u in raw_units:
                        cu = _canon_unit(str(u))
                        units_detected[cu] = units_detected.get(cu, 0) + 1

                if include_pattern_examples:
                    numeric_like_examples = {
                        "strict": _take_examples(scan_series[strict_mask], EXAMPLES_PER_PATTERN),
                        "currency_prefixed": _take_examples(scan_series[currency_mask], EXAMPLES_PER_PATTERN),

                        "suffix_multiplier": _take_examples(scan_series[suffix_mult_mask], EXAMPLES_PER_PATTERN),
                        "suffix_unit": _take_examples(scan_series[unit_suffix_mask], EXAMPLES_PER_PATTERN),

                        "percent": _take_examples(scan_series[percent_mask], EXAMPLES_PER_PATTERN),
                        "thousands_sep": _take_examples(scan_series[thousands_mask], EXAMPLES_PER_PATTERN),
                        "parens_negative": _take_examples(scan_series[parens_mask], EXAMPLES_PER_PATTERN),
                        "bool_like": _take_examples(scan_series[bool_mask], EXAMPLES_PER_PATTERN),
                    }
                else:
                    numeric_like_examples = {}

                # Date-like patterns and parse attempt
                iso_mask = scan_series.str.match(RE_ISO_DATE, na=False)
                month_mask = scan_series.str.contains(RE_MONTH_NAME, na=False)

                parsed_dt, success_dt, policy = _try_parse_date_for_profile(scan_series)
                parse_success_pct = _pct(int(success_dt.sum()), scan_n)

                if include_pattern_examples:
                    failures = scan_series[~success_dt]
                    parse_failure_examples = _take_examples(failures, EXAMPLES_PER_PATTERN)
                else:
                    parse_failure_examples = []

                date_like = {
                    "iso_pct": _pct(int(iso_mask.sum()), scan_n),
                    "month_name_pct": _pct(int(month_mask.sum()), scan_n),
                    "parse_success_pct": parse_success_pct,
                    "parse_policy": policy,
                    "parse_failure_examples": parse_failure_examples,
                }

                # Multi-value delimited detection
                # IMPORTANT: do NOT treat whitespace as a delimiter (would misclassify normal text).
                # Instead, treat separators with optional surrounding whitespace as the delimiter.
                delimiter_specs = [
                    {"name": "comma", "pattern": r"\s*,\s*"},
                    {"name": "semicolon", "pattern": r"\s*;\s*"},
                    {"name": "pipe", "pattern": r"\s*\|\s*"},
                    {"name": "slash", "pattern": r"\s*/\s*"},
                ]

                best_spec: Optional[Dict[str, str]] = None
                best_multi_pct = -1.0
                best_tokens_stats = (0.0, 0)  # avg_tokens, max_tokens
                best_vocab: List[Dict[str, Any]] = []

                for spec in delimiter_specs:
                    pat = spec["pattern"]

                    # Regex split so ", " and ",    " behave identically
                    token_lists = scan_series.str.split(pat, regex=True)

                    # Count non-empty tokens after strip
                    token_counts = token_lists.apply(
                        lambda lst: len([t for t in lst if str(t).strip() != ""]) if isinstance(lst, list) else 1
                    )

                    multi_mask = token_counts > 1
                    multi_pct = _pct(int(multi_mask.sum()), scan_n)

                    if multi_pct > best_multi_pct:
                        best_multi_pct = multi_pct
                        best_spec = spec

                        avg_tokens = float(token_counts.mean()) if scan_n else 0.0
                        max_tokens = int(token_counts.max()) if scan_n else 0
                        best_tokens_stats = (avg_tokens, max_tokens)

                        # Build vocab top-k from multi-valued rows only (optional)
                        # When include_pattern_examples=False (e.g., /profile_summary), skip this to reduce runtime.
                        if include_pattern_examples:
                            tokens_flat: List[str] = []
                            for lst in token_lists[multi_mask].tolist():
                                if not isinstance(lst, list):
                                    continue
                                for tok in lst:
                                    tok2 = str(tok).strip()
                                    if tok2:
                                        tokens_flat.append(tok2)

                            if tokens_flat:
                                tok_series = pd.Series(tokens_flat, dtype="string")
                                tok_vc = tok_series.value_counts().head(TOP_K_FREQUENT)
                                best_vocab = [{"token": _truncate(str(k), MAX_STRING_LEN), "count": int(v)} for k, v in
                                              tok_vc.items()]
                            else:
                                best_vocab = []
                        else:
                            best_vocab = []

                # Token vocab size + delimiter presence + token-shape evidence for the chosen delimiter.
                # This is critical for fields like Positions, but it's expensive. For /profile_summary, skip it.
                token_vocab_size = 0
                delimiter_presence_pct = 0.0
                token_shape_pct = 0.0

                if best_spec is not None:
                    best_pat = best_spec["pattern"]

                    # How often the delimiter pattern appears at all (not the same as multi_token_pct)
                    delim_present_mask = scan_series.str.contains(best_pat, regex=True, na=False)
                    delimiter_presence_pct = _pct(int(delim_present_mask.sum()), scan_n)

                    if include_pattern_examples:
                        best_token_lists = scan_series.str.split(best_pat, regex=True)

                        tokens_flat_all: List[str] = []
                        for lst in best_token_lists.tolist():
                            if not isinstance(lst, list):
                                continue
                            for tok in lst:
                                tok2 = str(tok).strip()
                                if tok2:
                                    tokens_flat_all.append(tok2)

                        token_vocab_size = len(set(tokens_flat_all)) if tokens_flat_all else 0

                        # Token shape: percent of tokens that look like short codes (e.g., ST, RW, CAM)
                        if tokens_flat_all:
                            tok_series_all = pd.Series(tokens_flat_all, dtype="string")
                            shape_mask = tok_series_all.str.match(r"^[A-Za-z]{1,4}$", na=False)
                            token_shape_pct = _pct(int(shape_mask.sum()), int(len(tok_series_all)))

                multi_value = {
                    "delimiter": best_spec["name"] if best_spec is not None else None,
                    "delimiter_pattern": best_spec["pattern"] if best_spec is not None else None,
                    "multi_token_pct": round(best_multi_pct, 6) if best_multi_pct >= 0 else 0.0,
                    "delimiter_presence_pct": round(float(delimiter_presence_pct), 6),
                    "avg_tokens": round(best_tokens_stats[0], 6),
                    "max_tokens": best_tokens_stats[1],
                    "token_vocab_topk": best_vocab,
                    "token_vocab_size": int(token_vocab_size),
                    "token_shape_pct": round(float(token_shape_pct), 6),
                }




                # Range-like patterns
                year_range_mask = scan_series.str.match(RE_YEAR_RANGE, na=False)
                num_range_mask = scan_series.str.match(RE_NUM_RANGE, na=False)
                if include_pattern_examples:
                    range_examples = _take_examples(scan_series[year_range_mask | num_range_mask], EXAMPLES_PER_PATTERN)
                else:
                    range_examples = []

                range_like = {
                    "year_range_pct": _pct(int(year_range_mask.sum()), scan_n),
                    "numeric_range_pct": _pct(int(num_range_mask.sum()), scan_n),
                    "examples": range_examples,
                }

                patterns["numeric_like"] = numeric_like
                patterns["numeric_like_examples"] = numeric_like_examples
                patterns["date_like"] = date_like
                patterns["multi_value"] = multi_value
                patterns["range_like"] = range_like

                # -----------------------------
                # Unit plan (for cleaner executor)
                # -----------------------------
                UNIT_PLAN_DEFAULTS = {
                    # deterministic baseline (override later via env/config if desired)
                    "weight": "kg",
                    "length": "cm",
                    "time": "ms",  # psych default for reaction times
                    "age": "yr",
                }
                UNIT_KIND_BY_UNIT = {
                    # weight
                    "kg": "weight", "lb": "weight", "g": "weight", "mg": "weight",

                    # length
                    "mm": "length", "cm": "length", "m": "length", "ft": "length", "in": "length",

                    # time
                    "ms": "time", "s": "time", "min": "time", "h": "time", "d": "time", "wk": "time",

                    # age / longitudinal (kept as separate “age” kind to avoid accidental normalization to time)
                    "yr": "age", "mo": "age",
                }
                UNIT_CONVERSIONS = {
                    # weight
                    ("lb", "kg"): {"factor": 0.45359237, "formula": "kg = lb * 0.45359237"},
                    ("kg", "lb"): {"factor": 1.0 / 0.45359237, "formula": "lb = kg / 0.45359237"},
                    ("g", "kg"): {"factor": 0.001, "formula": "kg = g * 0.001"},
                    ("mg", "kg"): {"factor": 0.000001, "formula": "kg = mg * 0.000001"},
                    ("kg", "g"): {"factor": 1000.0, "formula": "g = kg * 1000"},
                    ("kg", "mg"): {"factor": 1_000_000.0, "formula": "mg = kg * 1000000"},

                    # length
                    ("mm", "cm"): {"factor": 0.1, "formula": "cm = mm * 0.1"},
                    ("cm", "mm"): {"factor": 10.0, "formula": "mm = cm * 10"},
                    ("m", "cm"): {"factor": 100.0, "formula": "cm = m * 100"},
                    ("cm", "m"): {"factor": 0.01, "formula": "m = cm * 0.01"},
                    ("in", "cm"): {"factor": 2.54, "formula": "cm = in * 2.54"},
                    ("cm", "in"): {"factor": 1.0 / 2.54, "formula": "in = cm / 2.54"},
                    ("ft", "in"): {"factor": 12.0, "formula": "in = ft * 12"},
                    ("in", "ft"): {"factor": 1.0 / 12.0, "formula": "ft = in / 12"},
                    ("ft", "cm"): {"factor": 30.48, "formula": "cm = ft * 30.48"},
                    ("cm", "ft"): {"factor": 1.0 / 30.48, "formula": "ft = cm / 30.48"},

                    # time (canonical default is ms)
                    ("s", "ms"): {"factor": 1000.0, "formula": "ms = s * 1000"},
                    ("min", "ms"): {"factor": 60_000.0, "formula": "ms = min * 60000"},
                    ("h", "ms"): {"factor": 3_600_000.0, "formula": "ms = h * 3600000"},
                    ("d", "ms"): {"factor": 86_400_000.0, "formula": "ms = d * 86400000"},
                    ("wk", "ms"): {"factor": 604_800_000.0, "formula": "ms = wk * 604800000"},
                    ("ms", "s"): {"factor": 0.001, "formula": "s = ms * 0.001"},

                    # age (canonical default is yr)
                    ("mo", "yr"): {"factor": 1.0 / 12.0, "formula": "yr = mo / 12"},
                    ("yr", "mo"): {"factor": 12.0, "formula": "mo = yr * 12"},
                }

                def _pick_canonical_unit(units: Dict[str, int], threshold: float = 0.60) -> Optional[Dict[str, Any]]:
                    if not units:
                        return None
                    parseable_count = int(sum(units.values()))
                    if parseable_count <= 0:
                        return None

                    # Determine "kind" by majority (weight vs length)
                    kind_counts: Dict[str, int] = {}
                    for u, c in units.items():
                        knd = UNIT_KIND_BY_UNIT.get(u)
                        if knd:
                            kind_counts[knd] = kind_counts.get(knd, 0) + int(c)
                    if not kind_counts:
                        return None

                    kind = max(kind_counts.items(), key=lambda kv: kv[1])[0]
                    default_unit = UNIT_PLAN_DEFAULTS.get(kind)
                    if not default_unit:
                        return None

                    winner_unit, winner_count = max(units.items(), key=lambda kv: kv[1])
                    winner_share = float(winner_count) / float(parseable_count)

                    if winner_share >= threshold:
                        canonical = winner_unit
                        selection_rule = f"majority>={threshold:.2f}"
                    else:
                        canonical = default_unit
                        selection_rule = f"majority<{threshold:.2f} -> default"

                    conversions: List[Dict[str, Any]] = []
                    for u, c in sorted(units.items(), key=lambda kv: (-kv[1], kv[0])):
                        if u == canonical:
                            continue
                        conv = UNIT_CONVERSIONS.get((u, canonical))
                        if conv:
                            conversions.append({
                                "from": u,
                                "to": canonical,
                                "factor": float(conv["factor"]),
                                "formula": conv["formula"],
                                "count": int(c),
                            })

                    return {
                        "units_detected": units,
                        "parseable_count": int(parseable_count),
                        "canonical_unit_recommended": canonical,
                        "selection_rule": selection_rule,
                        "conversion": conversions,  # may be empty if no mapping exists
                        "exceptions": {
                            "unit_missing_count": None,
                            "range_count": None,
                            "unparseable_examples": [],
                        },
                    }

                unit_plan = _pick_canonical_unit(units_detected) if "units_detected" in locals() else None
                if unit_plan is not None:
                    col_info["unit_plan"] = unit_plan

                # -----------------------------
                # Unit plan exceptions (auditability + cleaner routing)
                # -----------------------------
                if isinstance(col_info.get("unit_plan"), dict):
                    # Values that look numeric but have no unit when the column contains unit-suffixed values
                    unit_missing_count = int(strict_mask.sum()) if int(unit_suffix_mask.sum()) > 0 else 0

                    # Range-like count (year or numeric ranges)
                    range_count = int((year_range_mask | num_range_mask).sum())

                    # Unparseable examples (neither strict numeric nor unit-suffixed numeric)
                    unparsable_mask = ~(strict_mask | unit_suffix_mask)
                    unparseable_examples = _take_examples(scan_series[unparsable_mask], EXAMPLES_PER_PATTERN)

                    col_info["unit_plan"]["exceptions"]["unit_missing_count"] = int(unit_missing_count)
                    col_info["unit_plan"]["exceptions"]["range_count"] = int(range_count)
                    col_info["unit_plan"]["exceptions"]["unparseable_examples"] = unparseable_examples

            col_info["patterns"] = patterns



            # -----------------------------
            # Step 4 — Candidate interpretations
            # -----------------------------
            candidates: List[Dict[str, Any]] = []

            # Use scan_n and computed pattern pcts
            num_like = patterns.get("numeric_like", {})
            date_like = patterns.get("date_like", {})
            multi_like = patterns.get("multi_value", {})
            range_like = patterns.get("range_like", {})

            strict_pct = float(num_like.get("strict_pct", 0.0))
            currency_pct = float(num_like.get("currency_pct", 0.0))
            suffix_pct = float(num_like.get("suffix_pct", 0.0))

            # Keep Step 4 aligned with Step 3 numeric_like outputs
            suffix_multiplier_pct = float(num_like.get("suffix_multiplier_pct", 0.0))
            suffix_unit_pct = float(num_like.get("suffix_unit_pct", 0.0))

            percent_pct = float(num_like.get("percent_pct", 0.0))
            thousands_pct = float(num_like.get("thousands_sep_pct", 0.0))
            month_pct = float(date_like.get("month_name_pct", 0.0))
            iso_pct = float(date_like.get("iso_pct", 0.0))
            date_parse_pct = float(date_like.get("parse_success_pct", 0.0))
            multi_token_pct = float(multi_like.get("multi_token_pct", 0.0))
            year_range_pct = float(range_like.get("year_range_pct", 0.0))
            num_range_pct = float(range_like.get("numeric_range_pct", 0.0))

            def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
                return max(lo, min(hi, x))

            # Simple deterministic ranking rules
            def add_candidate(score: float, obj: Dict[str, Any]) -> None:
                obj["confidence"] = round(float(score), 6)
                candidates.append(obj)

            # ---------------------------------------------------------
            # Numeric-coded detection (binary codes, Likert-ish scales)
            # ---------------------------------------------------------
            # (Decision) Treat low-cardinality, strict-integer columns as potentially coded categories.
            # Why it matters: prevents columns like sex=1/2 or Likert 1..5 from becoming "100% numeric".
            coded_kind: Optional[str] = None  # "binary" | "scale" | None
            coded_levels: Optional[List[int]] = None  # sorted integer levels if detected
            coded_conf_cap_numeric = 1.0  # numeric confidence cap if coded

            # Only attempt if column is small-cardinality AND strongly strict numeric AND not obviously "transform" numeric.
            # We exclude currency/percent/thousands/suffix-unit because those are genuinely numeric transforms.
            coded_candidate_gate = (
                    (unique_count > 0 and unique_count <= max_cat) and
                    (strict_pct >= 95.0) and
                    ((currency_pct + suffix_pct) < 10.0) and
                    (percent_pct < 10.0) and
                    (thousands_pct < 10.0) and
                    (suffix_unit_pct < 10.0)
            )

            if coded_candidate_gate:
                # Build integer levels robustly (handles integer dtype, float dtype with .0, and numeric strings)
                # Use existing levels if present; else compute.
                raw_levels = col_info.get("levels")
                levels_num: List[float] = []

                if isinstance(raw_levels, list) and len(raw_levels) > 0:
                    try:
                        levels_num = [float(v) for v in raw_levels]
                    except Exception:
                        levels_num = []
                else:
                    try:
                        # deterministic: use dropna uniques
                        vals = s.dropna().unique().tolist()
                        levels_num = [float(v) for v in vals]
                    except Exception:
                        levels_num = []

                # Keep only integer-like values (e.g., 1.0 is ok; 1.2 is not)
                int_like = []
                for x in levels_num:
                    if x is None:
                        continue
                    try:
                        if abs(float(x) - round(float(x))) < 1e-9:
                            int_like.append(int(round(float(x))))
                    except Exception:
                        continue

                int_like = sorted(set(int_like))

                # If most uniques are integer-like, we can reason about coding patterns.
                if len(int_like) == unique_count and unique_count >= 2:
                    lo, hi = int_like[0], int_like[-1]
                    span = hi - lo

                    # Binary code patterns commonly seen in surveys / datasets
                    if unique_count == 2 and int_like in ([0, 1], [1, 2], [-1, 1]):
                        coded_kind = "binary"
                        coded_levels = int_like
                        coded_conf_cap_numeric = 0.75

                    # Likert-ish / bounded scale patterns (3–11 levels, small contiguous-ish span)
                    # Examples: 1..5, 1..7, 0..10, 0..4
                    elif 3 <= unique_count <= 11 and 2 <= span <= 12:
                        # contiguous if unique_count == span+1; allow at most 1 missing level
                        missing_levels = (span + 1) - unique_count
                        if missing_levels in (0, 1):
                            coded_kind = "scale"
                            coded_levels = int_like
                            coded_conf_cap_numeric = 0.70

                    # Generic low-card integer coding (fallback): small set + small span
                    # Example: category codes 1..8 (not necessarily Likert, but categorical-coded is plausible)
                    elif unique_count <= 15 and span <= 20:
                        coded_kind = "coded_category"
                        coded_levels = int_like
                        coded_conf_cap_numeric = 0.78

            # Numeric candidates
            if strict_pct >= 80.0:

                # If a small suffix-multiplier minority exists (e.g. "1.6K"), record that explicitly.
                # This prevents downstream systems from assuming "strict-only".
                parse_mode = "strict"
                if suffix_multiplier_pct > 0.0:
                    parse_mode = "strict_with_suffix_minority"

                # Apply coded confidence cap so "coded" cannot dominate as 1.0 numeric.
                base_numeric_conf = strict_pct / 100.0
                if coded_kind is not None:
                    base_numeric_conf = min(base_numeric_conf, coded_conf_cap_numeric)

                add_candidate(base_numeric_conf, {
                    "type": "numeric",
                    "parse": parse_mode,
                    "evidence": {
                        "strict_pct": strict_pct,
                        "suffix_multiplier_pct": suffix_multiplier_pct,
                        "suffix_pct": suffix_pct,

                        # expose coded detection for auditability + downstream routing
                        "coded_kind": coded_kind,
                        "coded_levels": coded_levels,
                        "coded_numeric_conf_cap": coded_conf_cap_numeric,
                    }
                })

            if (currency_pct + suffix_pct) >= 10.0:
                add_candidate(min(0.95, (currency_pct + suffix_pct) / 100.0 + 0.1), {
                    "type": "numeric",
                    "parse": "currency+suffix_possible",
                    "evidence": {"currency_pct": currency_pct, "suffix_pct": suffix_pct}
                })

            # Numeric-with-unit candidate (e.g., Height/Weight)
            # Triggers when unit suffix detection is present or a unit_plan exists.
            if suffix_unit_pct >= 10.0 or isinstance(col_info.get("unit_plan"), dict):
                base = _clamp(suffix_unit_pct / 100.0, 0.0, 1.0)
                bump = 0.15 if isinstance(col_info.get("unit_plan"), dict) else 0.0
                add_candidate(_clamp(0.35 + 0.6 * base + bump, 0.0, 0.99), {
                    "type": "numeric_with_unit",
                    "parse": "unit_suffix",
                    "evidence": {
                        "suffix_unit_pct": suffix_unit_pct,
                        "unit_plan": col_info.get("unit_plan"),
                    }
                })

            if percent_pct >= 10.0:
                add_candidate(min(0.9, percent_pct / 100.0 + 0.1), {
                    "type": "numeric",
                    "parse": "percent_possible",
                    "evidence": {"percent_pct": percent_pct}
                })

            if thousands_pct >= 10.0:
                add_candidate(min(0.85, thousands_pct / 100.0 + 0.05), {
                    "type": "numeric",
                    "parse": "thousands_sep_possible",
                    "evidence": {"thousands_sep_pct": thousands_pct}
                })

            # Date candidates
            if date_parse_pct >= 60.0:
                add_candidate(min(0.95, date_parse_pct / 100.0), {
                    "type": "date",
                    "parse": "to_datetime",
                    "evidence": {"parse_success_pct": date_parse_pct, "iso_pct": iso_pct, "month_name_pct": month_pct}
                })
            elif (iso_pct + month_pct) >= 20.0:
                add_candidate(min(0.8, (iso_pct + month_pct) / 100.0 + 0.1), {
                    "type": "date",
                    "parse": "date_like_strings",
                    "evidence": {"iso_pct": iso_pct, "month_name_pct": month_pct, "parse_success_pct": date_parse_pct}
                })

         
            # Multi-value categorical candidates
            # Goal: avoid "rigging" to high confidence. Surface ambiguity:
            # - weak evidence -> low confidence multi (visible for review)
            # - decent evidence -> medium multi
            # - overwhelming evidence -> higher multi (still capped)
            token_vocab_size = int(multi_like.get("token_vocab_size") or 0)
            delimiter_presence_pct = float(multi_like.get("delimiter_presence_pct") or 0.0)
            token_shape_pct = float(multi_like.get("token_shape_pct") or 0.0)

            has_delim = (multi_like.get("delimiter_pattern") is not None)

            # Evidence that this is a controlled-vocab multi field (Positions-like)
            controlled_vocab_evidence = (
                has_delim and
                (delimiter_presence_pct >= 1.0) and
                (token_vocab_size > 0 and token_vocab_size <= 120) and
                (token_shape_pct >= 60.0)
            )

            # When evidence exists, always add a multi candidate, but keep confidence bounded.
            # - If multi_token_pct is tiny, candidate should exist but be low confidence.
            # - If multi_token_pct is strong, confidence can rise (still capped).
            if has_delim and (multi_token_pct >= 1.0 or controlled_vocab_evidence):
                base = _clamp(multi_token_pct / 100.0, 0.0, 1.0)

                # small bump for controlled vocab signal; not enough to guarantee dominance
                bump = 0.10 if controlled_vocab_evidence else 0.0

                # raw score (then tier-capped below)
                raw = _clamp(0.25 + 0.70 * base + bump, 0.0, 0.95)

                # Tier caps to preserve ambiguity and encourage review:
                # - <5% multi: keep multi confidence <= 0.60
                # - 5–20%: keep <= 0.80
                # - >=20%: keep <= 0.90
                if multi_token_pct < 5.0:
                    conf = min(raw, 0.60)
                elif multi_token_pct < 20.0:
                    conf = min(raw, 0.80)
                else:
                    conf = min(raw, 0.90)

                add_candidate(conf, {
                    "type": "categorical_multi",
                    "op": f"split_regex(pattern='{multi_like.get('delimiter_pattern')}')",
                    "evidence": {
                        "multi_token_pct": multi_token_pct,
                        "delimiter_presence_pct": delimiter_presence_pct,
                        "delimiter": multi_like.get("delimiter"),
                        "delimiter_pattern": multi_like.get("delimiter_pattern"),
                        "token_vocab_size": token_vocab_size,
                        "token_shape_pct": token_shape_pct,
                        "controlled_vocab_evidence": bool(controlled_vocab_evidence),
                    }
                })




            # Range candidates
            if (year_range_pct + num_range_pct) >= 10.0:
                add_candidate(min(0.8, (year_range_pct + num_range_pct) / 100.0 + 0.1), {
                    "type": "range_like",
                    "op": "split_range",
                    "evidence": {"year_range_pct": year_range_pct, "numeric_range_pct": num_range_pct}
                })

            # Single-value categorical candidate (low-cardinality, non-multi, non-range, non-date-dominant, non-numeric-dominant)
            # - fixes Preferred Foot (2 uniques)
            # - fixes things like W/F, SM, A/W, etc.
            is_low_card = (unique_count > 0 and unique_count <= max_cat)
            is_multiish = (multi_token_pct >= 20.0)
            is_rangeish = ((year_range_pct + num_range_pct) >= 10.0)
            is_dateish = (date_parse_pct >= 60.0 or (iso_pct + month_pct) >= 20.0)

            # numeric dominance estimate from pattern signals
            num_dom_est = max(strict_pct, currency_pct + suffix_pct, percent_pct, thousands_pct)  # pct scale 0..100
            is_numericish = (strict_pct >= 80.0) or ((currency_pct + suffix_pct) >= 10.0) or (percent_pct >= 10.0) or (thousands_pct >= 10.0)

            # Allow categorical even if numericish when we have strong coded evidence (binary / scale / coded_category).
            # (Decision) If coded_kind is set, treat as categorical-coded even though values are numeric.
            # Why it matters: fixes sex=1/2 and Likert 1..5 being blocked by is_numericish.
            if is_low_card and (not is_multiish) and (not is_rangeish) and (not is_dateish) and ((not is_numericish) or (coded_kind is not None)):

                # Confidence should be HIGH for "clear" categoricals (2–5 uniques) to avoid review spam.
                # (Decision) Make confidence scale inversely with unique_count much more steeply at the low end.
                # Why it matters: columns like W/F, SM, A/W should not be reviewed by default.
                if unique_count <= 1:
                    uniq_score = 0.10
                elif unique_count == 2:
                    uniq_score = 1.00
                elif unique_count == 3:
                    uniq_score = 0.98
                elif unique_count == 4:
                    uniq_score = 0.96
                elif unique_count == 5:
                    uniq_score = 0.94
                elif unique_count <= 8:
                    uniq_score = 0.88
                else:
                    # Smooth falloff toward 0.55 by max_cat (still allows categorical, but less “certain”)
                    uniq_score = 0.88 - (0.33 * ((unique_count - 8) / max(1, (max_cat - 8))))
                    uniq_score = _clamp(uniq_score, 0.55, 0.88)

                structural_signal = _clamp(1.0 - (multi_token_pct / 100.0), 0.0, 1.0)
                parse_signal = _clamp(1.0 - (num_dom_est / 100.0), 0.0, 1.0)

                # Missingness is coverage, not type evidence; do not penalize categorical confidence.
                cat_conf = _clamp(0.25 + 0.75 * uniq_score, 0.0, 0.99)

                add_candidate(cat_conf, {
                    "type": "categorical",
                    "parse": "levels",
                    "evidence": {
                        "unique_count": unique_count,
                        "max_categorical_cardinality": max_cat,
                        "multi_token_pct": multi_token_pct,
                        "missing_pct": missing_pct,
                        "uniq_score": round(float(uniq_score), 6),
                        "structural_signal": round(float(structural_signal), 6),
                        "parse_signal": round(float(parse_signal), 6),
                    }
                })

            # Mixed candidate (keeps text confidence moderate when patterns conflict)
            # If numeric signals exist but aren't dominant, emit "mixed" to prevent overconfident text.
            has_some_numeric = (num_dom_est >= 20.0)  # at least 20% numeric-like patterns
            not_dominant_numeric = (num_dom_est < 80.0)
            if has_some_numeric and not_dominant_numeric and (not is_dateish) and (not is_rangeish):
                mixed_conf = _clamp(0.35 + 0.5 * (num_dom_est / 100.0), 0.35, 0.85)
                add_candidate(mixed_conf, {
                    "type": "mixed",
                    "parse": "partial_numeric",
                    "evidence": {"numeric_signal_pct_est": num_dom_est}
                })

            # Text candidate with elimination-based + cardinality-scaled confidence + URL/ID boosts
            # A) elimination: how strongly it's NOT numeric/date/range/bool
            p_num = _clamp(num_dom_est / 100.0, 0.0, 1.0)
            p_date = _clamp(date_parse_pct / 100.0, 0.0, 1.0)
            p_range = _clamp((year_range_pct + num_range_pct) / 100.0, 0.0, 1.0)
            p_bool = _clamp(float(num_like.get("bool_like_pct", 0.0)) / 100.0, 0.0, 1.0)
            elim = _clamp(1.0 - max(p_num, p_date, p_range, p_bool), 0.0, 1.0)

            # B) cardinality: smoothly increases above max_cat
            u = float(unique_count)
            m = float(max_cat)
            k = 2.0  # by u = 3m, u_norm ~ 1
            u_norm = _clamp((u - m) / max(1.0, (k * m)), 0.0, 1.0)

            # base combine
            text_conf = _clamp(0.1 + 0.9 * elim * u_norm, 0.0, 0.99)

            # If the column shows evidence of being a multi-category controlled-vocab field (e.g. Positions),
            # do NOT allow text to dominate purely due to high uniqueness.
            # Design goal: when ambiguous, keep text low-medium so the system escalates to column detail.
            token_vocab_size = int(multi_like.get("token_vocab_size") or 0)
            delimiter_presence_pct = float(multi_like.get("delimiter_presence_pct") or 0.0)
            token_shape_pct = float(multi_like.get("token_shape_pct") or 0.0)
            has_delim = bool(multi_like.get("delimiter_pattern"))

            controlled_vocab_evidence = (
                has_delim and
                (delimiter_presence_pct >= 1.0) and
                (token_vocab_size > 0 and token_vocab_size <= 80) and
                (token_shape_pct >= 60.0)
            )

            # If there's any credible multi structure, cap text confidence lower.
            # - weak/marginal evidence: cap at 0.75
            # - controlled vocab evidence: cap at 0.65
            if has_delim and delimiter_presence_pct >= 1.0 and token_vocab_size > 0 and token_vocab_size <= 120:
                text_conf = min(text_conf, 0.75)

            if controlled_vocab_evidence:
                text_conf = min(text_conf, 0.65)


            # URL/identifier boosts (high-confidence text even at moderate cardinality)
            # apply boost if any evidence in scan_series
            url_like_pct = 0.0
            uuid_like_pct = 0.0
            email_like_pct = 0.0
            if scan_n > 0:
                url_like_pct = _pct(int(scan_series.str.contains(RE_URL, na=False).sum()), scan_n)
                uuid_like_pct = _pct(int(scan_series.str.contains(RE_UUID, na=False).sum()), scan_n)
                email_like_pct = _pct(int(scan_series.str.match(RE_EMAIL, na=False).sum()), scan_n)

            boost = 0.0
            if url_like_pct >= 10.0:
                boost = max(boost, 0.5)
            if uuid_like_pct >= 10.0:
                boost = max(boost, 0.5)
            if email_like_pct >= 10.0:
                boost = max(boost, 0.4)

            text_conf = _clamp(max(text_conf, 0.75 if boost > 0 else text_conf) + boost * 0.2, 0.0, 0.99)

            add_candidate(text_conf, {
                "type": "text",
                "evidence": {
                    "inferred_type": t,
                    "elim": round(elim, 6),
                    "u_norm": round(u_norm, 6),
                    "numeric_signal_pct_est": round(num_dom_est, 6),
                    "date_parse_pct": date_parse_pct,
                    "range_signal_pct": round(year_range_pct + num_range_pct, 6),
                    "bool_like_pct": float(num_like.get("bool_like_pct", 0.0)),
                    "url_like_pct": round(url_like_pct, 6),
                    "uuid_like_pct": round(uuid_like_pct, 6),
                    "email_like_pct": round(email_like_pct, 6),
                    "max_categorical_cardinality": max_cat,
                    "unique_count": unique_count,
                }
            })


            # Sort descending by confidence
            candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
            col_info["candidates"] = candidates

            # Review recommendation:
            # (Decision) If the top candidate requires *any* transformation (parse/split/normalize), always require review.
            # Why it matters: guarantees downstream does not silently apply the wrong normalization.
            top = candidates[0] if candidates else {}
            second = candidates[1] if len(candidates) > 1 else {}

            top_conf = float((top.get("confidence") if top else 0.0) or 0.0)
            second_conf = float((second.get("confidence") if second else 0.0) or 0.0)
            conf_gap = top_conf - second_conf

            top_type = str(top.get("type") or "")
            top_parse = top.get("parse")
            top_op = top.get("op")

            def _top_requires_transform(ttype: str, parse: Any, op: Any) -> Tuple[bool, str]:
                # Operations always imply transform (splitting, etc.)
                if op:
                    return True, "op_requires_transform"

                # Type-based transforms
                if ttype in ("numeric_with_unit", "categorical_multi", "range_like", "date"):
                    return True, f"type_requires_transform:{ttype}"

                # Parse-based transforms (numeric parsing modes beyond strict)
                # Include strict_with_suffix_minority because it signals mixed numeric conventions in the column.
                if parse in ("currency+suffix_possible", "percent_possible", "thousands_sep_possible",
                             "strict_with_suffix_minority"):
                    return True, f"parse_requires_transform:{parse}"

                return False, "none"

            requires_transform, transform_reason = _top_requires_transform(top_type, top_parse, top_op)

            # Default ambiguity-based review (your existing rule)
            # (Decision) If top is a strong low-card categorical, do not trigger review due to a small gap alone.
            # Why it matters: avoids review spam on stable enums where the second-best is often "text".
            clear_categorical_no_review = (
                    (top_type == "categorical") and
                    (unique_count > 0 and unique_count <= max_cat) and
                    (top_conf >= 0.90) and
                    (not top_op) and
                    (top_parse == "levels")
            )

            ambiguous_review = (top_conf < 0.90) or ((conf_gap < 0.15) and (not clear_categorical_no_review))

            # Force review if transformation is required, else fall back to ambiguity rule
            review_recommended = bool(requires_transform or ambiguous_review)

            # Prefer a specific transform reason when applicable
            reason = transform_reason
            if reason == "none":
                reason = (
                    "low_top_confidence" if top_conf < 0.90
                    else "small_gap_between_top_candidates" if conf_gap < 0.15
                    else "none"
                )

            col_info["review"] = {
                "recommended": bool(review_recommended),
                "top_confidence": round(top_conf, 6),
                "second_confidence": round(second_conf, 6),
                "confidence_gap": round(conf_gap, 6),
                "reason": reason,
            }

            # -----------------------------
            # Step 5 — Outlier visibility & parseability (numeric/date) (optional)
            # -----------------------------
            # Numeric/date deep profiling is expensive (quantiles, extremes, invalid examples).
            # Only compute it when explicitly requested (e.g., full profile / column detail).
            if include_deep_profiles and scan_n > 0:
                parsed_num, success_num = _try_parse_numeric_for_profile(scan_series)
                parseable_n = int(success_num.sum())
                parseable_pct = _pct(parseable_n, scan_n)

                numeric_profile: Dict[str, Any] = {
                    "parseable_count": parseable_n,
                    "parseable_pct": parseable_pct,
                    "scan_rows": scan_n,
                    "quantiles": None,
                    "extremes": {"bottom": [], "top": []},
                    "invalid_examples": [],
                }

                if parseable_n > 0:
                    qs = parsed_num[success_num].quantile([0.01, 0.05, 0.5, 0.95, 0.99])
                    numeric_profile["quantiles"] = {
                        "p1": float(qs.loc[0.01]),
                        "p5": float(qs.loc[0.05]),
                        "p50": float(qs.loc[0.5]),
                        "p95": float(qs.loc[0.95]),
                        "p99": float(qs.loc[0.99]),
                    }

                    # Extremes with row indices from scan_series (not full df)
                    parsed_ok = parsed_num[success_num]
                    # bottom/top indices within scan_series index
                    bottom_idx = parsed_ok.nsmallest(EXTREMES_N).index
                    top_idx = parsed_ok.nlargest(EXTREMES_N).index

                    numeric_profile["extremes"]["bottom"] = [
                        {
                            "row_index": int(i),
                            "raw_value": _stringify_value(scan_series.loc[i]),
                            "parsed_value": float(parsed_ok.loc[i])
                        }
                        for i in bottom_idx
                    ]
                    numeric_profile["extremes"]["top"] = [
                        {
                            "row_index": int(i),
                            "raw_value": _stringify_value(scan_series.loc[i]),
                            "parsed_value": float(parsed_ok.loc[i])
                        }
                        for i in top_idx
                    ]

                # Invalid examples: values that failed numeric parsing
                invalid_vals = scan_series[~success_num]
                numeric_profile["invalid_examples"] = _take_examples(invalid_vals, EXAMPLES_PER_PATTERN)

                col_info["numeric_profile"] = numeric_profile

                # Date parseability: reuse existing date parsing info but add parseable_count
                parsed_dt, success_dt, policy = _try_parse_date_for_profile(scan_series)
                dt_parseable_n = int(success_dt.sum())
                date_profile: Dict[str, Any] = {
                    "parseable_count": dt_parseable_n,
                    "parseable_pct": _pct(dt_parseable_n, scan_n),
                    "scan_rows": scan_n,
                    "parse_policy": policy,
                    "invalid_examples": _take_examples(scan_series[~success_dt], EXAMPLES_PER_PATTERN),
                }
                col_info["date_profile"] = date_profile


            out["columns"][col] = col_info

        except Exception as exc:
            stage_logger.exception(
                "stage=col_error request_id=%s col=%s error_type=%s error=%s",
                rid, col, type(exc).__name__, str(exc)
            )
            raise
        finally:
            stage_logger.info("stage=col_end request_id=%s col=%s", rid, col)

    return {"data_profile": out}


# -----------------------------
# Main endpoint (unchanged external behavior)
# -----------------------------
@app.post("/profile", include_in_schema=False, deprecated=True)
async def profile(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    max_categorical_cardinality: int = Form(20),
    _=Depends(require_token),
) -> Dict[str, Any]:
    return await _run_full_profile(
        request=request,
        file=file,
        dataset_id=dataset_id,
        max_categorical_cardinality=max_categorical_cardinality,
    )


# -----------------------------
# Dify-safe: shallow summary
# -----------------------------
@app.post("/profile_summary", include_in_schema=False, deprecated=True)
async def profile_summary(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    max_categorical_cardinality: int = Form(20),
    _=Depends(require_token),
) -> Dict[str, Any]:

    full = (await _run_full_profile(
        request=request,
        file=file,
        dataset_id=dataset_id,
        max_categorical_cardinality=max_categorical_cardinality,
        include_samples=False,
        include_pattern_examples=False,
        include_deep_profiles=False,
    ))["data_profile"]

    # Depth-safe: list of flat column cards (no deep nesting)
    columns = []
    for col, info in full["columns"].items():
        top = (info.get("candidates") or [{}])[0]
        columns.append({
            "column": col,
            "inferred_type": info.get("inferred_type"),
            "missing_pct": info.get("missing_pct"),
            "unique_count": info.get("unique_count"),

            "top_candidate": {
                "type": top.get("type"),
                "confidence": top.get("confidence"),
                "parse": top.get("parse"),
                "op": top.get("op"),
                # stringify deep evidence to avoid JSON depth errors in Dify
                "top_candidate_evidence_json": (
                    json.dumps(top.get("evidence"), ensure_ascii=False)
                    if isinstance(top.get("evidence"), dict)
                    else None
                ),
            },

            # keep review shallow (booleans + scalars only)
            "review": {
                "recommended": bool((info.get("review") or {}).get("recommended")),
                "reason": (info.get("review") or {}).get("reason"),
                "top_confidence": (info.get("review") or {}).get("top_confidence"),
                "second_confidence": (info.get("review") or {}).get("second_confidence"),
                "confidence_gap": (info.get("review") or {}).get("confidence_gap"),
            },

            # stringify unit_plan (deep object)
            "unit_plan_json": (
                json.dumps(info.get("unit_plan"), ensure_ascii=False)
                if isinstance(info.get("unit_plan"), dict)
                else None
            ),
        })

    return {
        "dataset_id": full["dataset_id"],
        "dataset_sha256": full["dataset_sha256"],
        "n_rows": full["n_rows"],
        "n_columns": full["n_columns"],

        # Surface profiler configuration for auditability
        # (Decision) Flatten as JSON string to avoid depth issues in Dify.
        # Why it matters: pattern confidence values depend on sampling policy.
        "profiler_config_json": json.dumps(full.get("profiler_config", {})),

        # stringify deep summary object
        "summary_json": json.dumps(full.get("summary"), ensure_ascii=False),

        "columns": columns,
    }


# -----------------------------
# Dify-safe: one column detail (returned as JSON string)
# -----------------------------
@app.post("/profile_column_detail", include_in_schema=False, deprecated=True)
async def profile_column_detail(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    column: str = Form(...),
    max_categorical_cardinality: int = Form(20),
    _=Depends(require_token),
) -> Dict[str, Any]:

    full = (await _run_full_profile(
        request=request,
        file=file,
        dataset_id=dataset_id,
        max_categorical_cardinality=max_categorical_cardinality,
    ))["data_profile"]

    if column not in full["columns"]:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    # Keep Dify under depth limit by returning the deep column object as a string.
    # The LLM can still consume this string as text.
    return {
        "dataset_id": full["dataset_id"],
        "dataset_sha256": full["dataset_sha256"],
        "column": column,
        "profile_json": json.dumps(full["columns"][column], ensure_ascii=False),
    }
# -----------------------------
# Dify-safe: association + dependency evidence
# -----------------------------
def _cap_df_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.iloc[:max_rows].copy()

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")

def _safe_json(obj: Any) -> str:
    # Keep response shallow for Dify by stringifying evidence dicts.
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"_stringify_failed": True, "repr": repr(obj)}, ensure_ascii=False)

def _to_numeric_series(series: pd.Series) -> Tuple[pd.Series, float]:
    """
    Returns (numeric_values, parse_success_pct over non-null).
    Uses your profiling numeric parser for string columns.

    NOTE: We must not build float64 arrays from pd.NA (causes TypeError).
    We use NaN for float64 buffers.
    """
    s = series
    non_null = s[~s.isna()]

    if len(non_null) == 0:
        nan = float("nan")
        return pd.Series([nan] * len(s), index=s.index, dtype="float64"), 0.0

    if pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce").astype("float64")
        ok = int(num.notna().sum())
        return num, round((ok / len(non_null)) * 100.0, 6)

    # String parse path (reuse your profiling parser)
    s_str = _as_str_series(non_null).str.strip()
    parsed, success = _try_parse_numeric_for_profile(s_str)

    # Ensure parsed is plain float64 with NaN (not pandas nullable Float64 with pd.NA)
    parsed_float = pd.to_numeric(parsed, errors="coerce").astype("float64")

    nan = float("nan")
    num_full = pd.Series([nan] * len(s), index=s.index, dtype="float64")
    num_full.loc[non_null.index] = parsed_float

    ok = int(pd.Series(success, index=s_str.index).astype(bool).sum())
    return num_full, round((ok / len(non_null)) * 100.0, 6)

def _to_datetime_series(series: pd.Series) -> Tuple[pd.Series, float]:
    non_null = series[~series.isna()]
    if len(non_null) == 0:
        return pd.Series([pd.NaT] * len(series), index=series.index), 0.0

    s_str = _as_str_series(non_null).str.strip()
    parsed_a = pd.to_datetime(s_str, errors="coerce", dayfirst=False)
    parsed_b = pd.to_datetime(s_str, errors="coerce", dayfirst=True)

    ok_a = int(parsed_a.notna().sum())
    ok_b = int(parsed_b.notna().sum())
    parsed = parsed_b if ok_b > ok_a else parsed_a
    ok = max(ok_a, ok_b)

    out = pd.Series([pd.NaT] * len(series), index=series.index)
    out.loc[non_null.index] = parsed
    return out, round((ok / len(non_null)) * 100.0, 6)

def _cramers_v_from_crosstab(ct: pd.DataFrame) -> Optional[float]:
    """
    Cramer's V without scipy/numpy explicit calls.
    Returns None if not computable.
    """
    try:
        n = float(ct.values.sum())
        if n <= 0:
            return None

        row_sums = ct.sum(axis=1).astype("float64")
        col_sums = ct.sum(axis=0).astype("float64")

        # Expected counts under independence
        expected = pd.DataFrame(index=ct.index, columns=ct.columns, dtype="float64")
        for r in ct.index:
            for c in ct.columns:
                expected.loc[r, c] = (row_sums.loc[r] * col_sums.loc[c]) / n

        observed = ct.astype("float64")
        with pd.option_context("mode.use_inf_as_na", True):
            chi2 = (((observed - expected) ** 2) / expected).fillna(0.0).values.sum()

        r, k = ct.shape
        if r <= 1 or k <= 1:
            return None

        phi2 = chi2 / n
        # Bias correction (Bergsma 2013 style); keep simple + safe
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1.0)) if n > 1 else 0.0
        rcorr = r - ((r - 1) ** 2) / (n - 1.0) if n > 1 else r
        kcorr = k - ((k - 1) ** 2) / (n - 1.0) if n > 1 else k
        denom = min((kcorr - 1.0), (rcorr - 1.0))
        if denom <= 0:
            return None
        return float((phi2corr / denom) ** 0.5)
    except Exception:
        return None

def _add_signal(out: List[Dict[str, Any]], *, kind: str, columns: List[str], score: Optional[float], metric: str, evidence: Dict[str, Any]) -> None:
    out.append({
        "kind": kind,
        "columns": columns,
        "score": (None if score is None else round(float(score), 6)),
        "metric": metric,
        "evidence_json": _safe_json(evidence),
    })

@app.post("/evidence_associations", include_in_schema=False, deprecated=True)
async def evidence_associations(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    max_categorical_cardinality: int = Form(20),
    _=Depends(require_token),
) -> Dict[str, Any]:
    """
    Deterministic association + dependency evidence:
    - numeric–numeric correlations
    - categorical–categorical association (Cramer's V)
    - numeric by group (eta^2-like variance explained)
    - dependency checks from name-based heuristics (subtotal+tax=total; start<=end)
    """
    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    dataset_sha256 = sha256_hex(raw_bytes)

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    df = _cap_df_rows(df, ASSOC_MAX_ROWS)
    n_rows_scanned = int(len(df))

    # --- Column selection (deterministic caps) ---
    # (Decision) Use observed non-null counts to prioritize columns for pairwise tests.
    # Why it matters: better signal density under strict caps.
    non_null_counts = df.notna().sum().sort_values(ascending=False)

    # Build numeric frame (native numeric + parseable numeric strings)
    numeric_cols: List[str] = []
    numeric_parse_pct: Dict[str, float] = {}

    # Prefer existing numeric dtype columns first
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    # Add parseable object columns next (by non-null density)
    obj_candidates = [c for c in non_null_counts.index.tolist() if c not in numeric_cols]
    for c in obj_candidates:
        if len(numeric_cols) >= ASSOC_MAX_NUMERIC_COLS:
            break
        ser = df[c]
        # Only attempt parse if it's not huge-cardinality text (quick gate)
        try:
            uniq = int(ser.nunique(dropna=True))
        except Exception:
            uniq = 10**9
        if uniq > max(ASSOC_MAX_CAT_CARD, 200):
            continue

        num, pct = _to_numeric_series(ser)
        if pct >= 60.0:  # parseable enough to be meaningfully numeric
            df[c + "__num__"] = num
            numeric_cols.append(c + "__num__")
            numeric_parse_pct[c] = pct

    # cap numeric cols deterministically (by non-null)
    numeric_cols = sorted(
        numeric_cols,
        key=lambda c: int(df[c].notna().sum()) if c in df.columns else 0,
        reverse=True
    )[:ASSOC_MAX_NUMERIC_COLS]

    # Build categorical candidates: low-card columns by unique_count (and not numeric dtypes)
    cat_cols: List[str] = []
    for c in non_null_counts.index.tolist():
        if len(cat_cols) >= ASSOC_MAX_CAT_COLS:
            break
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        try:
            uniq = int(df[c].nunique(dropna=True))
        except Exception:
            continue
        if 2 <= uniq <= min(int(max_categorical_cardinality), ASSOC_MAX_CAT_CARD):
            cat_cols.append(c)

    signals: List[Dict[str, Any]] = []

    # -----------------------------
    # 1) numeric–numeric correlations (Pearson; no scipy dependency)
    # -----------------------------
    # (Decision) Use Pearson instead of Spearman to avoid SciPy.
    # Why it matters: pandas' Spearman path imports scipy.stats, which is not available in this runtime.
    corr_pairs: List[Tuple[float, Dict[str, Any]]] = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            a = numeric_cols[i]
            b = numeric_cols[j]
            # Coerce to numeric to avoid pd.NA / mixed object issues during corr()
            sa_raw = df[a]
            sb_raw = df[b]

            sa = pd.to_numeric(sa_raw, errors="coerce")
            sb = pd.to_numeric(sb_raw, errors="coerce")

            mask = sa.notna() & sb.notna()
            overlap = int(mask.sum())
            if overlap < ASSOC_MIN_OVERLAP:
                continue

            # corr() will now see float + np.nan only (no NAType)
            r = sa[mask].corr(sb[mask], method="pearson")
            if pd.isna(r):
                continue

            r = float(r)
            corr_pairs.append((abs(r), {
                "a": a,
                "b": b,
                "r": r,
                "overlap": overlap,
                "non_null_a": int(sa.notna().sum()),
                "non_null_b": int(sb.notna().sum()),
                "raw_non_null_a": int(sa_raw.notna().sum()),
                "raw_non_null_b": int(sb_raw.notna().sum()),
            }))

    corr_pairs.sort(key=lambda x: x[0], reverse=True)
    for r_abs, meta in corr_pairs[:ASSOC_TOP_K_PAIRS]:
        _add_signal(
            signals,
            kind="numeric_numeric_corr",
            columns=[meta["a"], meta["b"]],
            score=r_abs,
            metric="abs_pearson_r",
            evidence=meta,
        )

    # -----------------------------
    # 2) categorical–categorical association (Cramer's V)
    # -----------------------------
    assoc_pairs: List[Tuple[float, Dict[str, Any]]] = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            a = cat_cols[i]
            b = cat_cols[j]
            sa = df[a]
            sb = df[b]
            overlap_mask = sa.notna() & sb.notna()
            overlap = int(overlap_mask.sum())
            if overlap < ASSOC_MIN_OVERLAP:
                continue

            ct = pd.crosstab(sa[overlap_mask], sb[overlap_mask])
            v = _cramers_v_from_crosstab(ct)
            if v is None:
                continue
            assoc_pairs.append((abs(float(v)), {
                "a": a, "b": b, "cramers_v": float(v), "overlap": overlap,
                "shape": [int(ct.shape[0]), int(ct.shape[1])]
            }))

    assoc_pairs.sort(key=lambda x: x[0], reverse=True)
    for v_abs, meta in assoc_pairs[:ASSOC_TOP_K_PAIRS]:
        _add_signal(
            signals,
            kind="categorical_categorical_assoc",
            columns=[meta["a"], meta["b"]],
            score=v_abs,
            metric="cramers_v",
            evidence=meta,
        )

    # -----------------------------
    # 3) numeric by group (eta^2-like) + explicit encoding_pair
    # -----------------------------
    ng_pairs: List[Tuple[float, Dict[str, Any]]] = []

    for num_c in numeric_cols[: min(len(numeric_cols), 20)]:
        s_num = df[num_c]
        if int(s_num.notna().sum()) < ASSOC_MIN_OVERLAP:
            continue

        for grp_c in cat_cols[: min(len(cat_cols), 20)]:
            s_grp = df[grp_c]
            mask = s_num.notna() & s_grp.notna()
            overlap = int(mask.sum())
            if overlap < ASSOC_MIN_OVERLAP:
                continue

            x = s_num[mask].astype("float64")
            g = s_grp[mask].astype("string")

            overall_mean = float(x.mean())
            ss_total = float(((x - overall_mean) ** 2).sum())
            if ss_total <= 0:
                continue

            ss_between = 0.0
            group_means: Dict[str, float] = {}
            group_nunique_num: Dict[str, int] = {}

            for level, idx in g.groupby(g).groups.items():
                xv = x.loc[idx]
                if len(xv) == 0:
                    continue
                mu = float(xv.mean())
                group_means[str(level)] = mu
                group_nunique_num[str(level)] = int(xv.nunique(dropna=True))
                ss_between += float(len(xv)) * (mu - overall_mean) ** 2

            eta2 = ss_between / ss_total if ss_total > 0 else 0.0
            n_groups = int(g.nunique(dropna=True))

            # (Decision) Treat eta2 ~ 1 as "this numeric column is an encoding of the categorical".
            # Why it matters: downstream cleaning must avoid double-counting (keep one, drop/derive the other).
            if eta2 >= 0.999 and n_groups >= 2:
                invertible = all(v == 1 for v in group_nunique_num.values()) if group_nunique_num else False
                _add_signal(
                    signals,
                    kind="encoding_pair",
                    columns=[grp_c, num_c],
                    score=float(min(1.0, eta2)),
                    metric="eta2_encoding",
                    evidence={
                        "overlap": overlap,
                        "n_groups": n_groups,
                        "invertible": bool(invertible),
                        "group_means": group_means,  # mapping hint
                        "group_nunique_num": group_nunique_num,  # sanity check
                    },
                )
                continue

            ng_pairs.append((eta2, {
                "numeric": num_c,
                "group": grp_c,
                "eta2": float(eta2),
                "overlap": overlap,
                "n_groups": n_groups,
            }))

    ng_pairs.sort(key=lambda x: x[0], reverse=True)
    for eta2, meta in ng_pairs[:ASSOC_TOP_K_PAIRS]:
        _add_signal(
            signals,
            kind="numeric_by_group",
            columns=[meta["numeric"], meta["group"]],
            score=eta2,
            metric="eta2",
            evidence=meta,
        )

    def _stem_for_items(col: str) -> Optional[str]:
        """
        Detect stems like HOP1.r -> HOP, auditc2.r -> auditc
        Returns None if it doesn't match the expected item pattern.
        """
        m = re.match(r"^(.+?)(\d+)\.r$", str(col))
        if not m:
            return None
        return m.group(1)

    def _find_total_col_for_stem(df_cols: List[str], stem: str) -> Optional[str]:
        """
        Prefer exact '<stem>.total', else '<stem>_total', else any column containing both stem and 'total'.
        """
        exact1 = f"{stem}.total"
        exact2 = f"{stem}_total"
        if exact1 in df_cols:
            return exact1
        if exact2 in df_cols:
            return exact2

        stem_l = stem.lower()
        for c in df_cols:
            cl = str(c).lower()
            if ("total" in cl) and (stem_l in cl):
                return c
        return None

    # -----------------------------
    # 3b) Key / grain evidence + time structure evidence (deterministic)
    # -----------------------------
    # (Decision) Emit “grain” candidates as evidence signals.
    # Why it matters: it prevents reshape steps that duplicate or collapse records incorrectly.
    #
    # (Decision) Detect time structure before reshape planning.
    # Why it matters: long format often needs (id, time, measure) not just (id, measure).

    def _is_idish_name(col: str) -> bool:
        cl = _norm_name(col)
        return any(k in cl for k in ["id", "uuid", "guid", "respondent", "subject", "participant", "record", "case"])

    def _col_uniqueness_stats(s: pd.Series) -> Dict[str, Any]:
        nn = int(s.notna().sum())
        nu = int(s.nunique(dropna=True))
        uniq_ratio = float(nu / nn) if nn > 0 else 0.0
        dup_rows = int(s.dropna().duplicated().sum()) if nn > 0 else 0
        return {
            "non_null": nn,
            "n_unique": nu,
            "uniq_ratio": uniq_ratio,
            "dup_rows": dup_rows,
            "n_rows": int(len(s)),
            "non_null_rate": float(nn / len(s)) if len(s) else 0.0,
        }

    # --- Candidate primary key (single col) ---
    # heuristic: high uniqueness + decent coverage; prefer id-ish names
    pk_candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for c in df.columns:
        s = df[c]
        st = _col_uniqueness_stats(s)
        if st["non_null"] < ASSOC_MIN_OVERLAP:
            continue
        # allow either: extremely unique, or reasonably unique with id-ish name
        score = st["uniq_ratio"]
        if score >= 0.98 or (score >= 0.95 and _is_idish_name(c)):
            # mild boost for id-ish names
            boosted = score + (0.02 if _is_idish_name(c) else 0.0)
            pk_candidates.append((boosted, c, st))

    pk_candidates.sort(key=lambda t: t[0], reverse=True)
    for boosted, c, st in pk_candidates[:5]:
        _add_signal(
            signals,
            kind="candidate_primary_key",
            columns=[c],
            score=float(min(1.0, st["uniq_ratio"])),
            metric="uniq_ratio",
            evidence={
                **st,
                "idish_name": bool(_is_idish_name(c)),
            },
        )
        _add_signal(
            signals,
            kind="duplicate_rows_rate_under_key",
            columns=[c],
            score=float(st["dup_rows"] / len(df)) if len(df) else 0.0,
            metric="dup_rows_rate",
            evidence={
                "key_cols": [c],
                "dup_rows": int(st["dup_rows"]),
                "dup_rows_rate": float(st["dup_rows"] / len(df)) if len(df) else 0.0,
                "n_rows": int(len(df)),
            },
        )

    # --- Candidate composite key (pairs), bounded search ---
    # Choose top 20 "id-ish or near-unique" columns, then test pairwise uniqueness.
    id_like_pool: List[Tuple[float, str]] = []
    for c in df.columns:
        st = _col_uniqueness_stats(df[c])
        if st["non_null"] < ASSOC_MIN_OVERLAP:
            continue
        score = st["uniq_ratio"]
        if score >= 0.90 or _is_idish_name(c):
            id_like_pool.append((score + (0.02 if _is_idish_name(c) else 0.0), c))

    id_like_pool.sort(key=lambda t: t[0], reverse=True)
    id_like_cols = [c for _, c in id_like_pool[:20]]

    comp_pairs: List[Tuple[float, Tuple[str, str], Dict[str, Any]]] = []
    for i in range(len(id_like_cols)):
        for j in range(i + 1, len(id_like_cols)):
            a = id_like_cols[i]
            b = id_like_cols[j]
            sa = df[a]
            sb = df[b]
            mask = sa.notna() & sb.notna()
            n = int(mask.sum())
            if n < ASSOC_MIN_OVERLAP:
                continue
            # composite uniqueness among rows with both present
            key_df = pd.DataFrame({"a": sa[mask].astype("string"), "b": sb[mask].astype("string")})
            n_unique = int(key_df.dropna().drop_duplicates().shape[0])
            uniq_ratio = float(n_unique / n) if n else 0.0
            dup_rows = int(n - n_unique)
            if uniq_ratio >= 0.98:
                comp_pairs.append((
                    uniq_ratio,
                    (a, b),
                    {
                        "key_cols": [a, b],
                        "n": n,
                        "n_unique": n_unique,
                        "uniq_ratio": uniq_ratio,
                        "dup_rows": dup_rows,
                        "dup_rows_rate": float(dup_rows / n) if n else 0.0,
                    }
                ))

    comp_pairs.sort(key=lambda t: t[0], reverse=True)
    for uniq_ratio, (a, b), ev in comp_pairs[:10]:
        _add_signal(
            signals,
            kind="candidate_composite_key",
            columns=[a, b],
            score=float(uniq_ratio),
            metric="uniq_ratio",
            evidence=ev,
        )
        _add_signal(
            signals,
            kind="duplicate_rows_rate_under_key",
            columns=[a, b],
            score=float(ev["dup_rows_rate"]),
            metric="dup_rows_rate",
            evidence=ev,
        )

    # --- Time column candidates ---
    time_candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for c in df.columns:
        sdt, pct = _to_datetime_series(df[c])
        if pct < 60.0:
            continue
        nn = int(sdt.notna().sum())
        if nn < ASSOC_MIN_OVERLAP:
            continue

        # monotonic-ish score on consecutive non-null pairs in original row order
        sdt_non = sdt.dropna()
        mono_rate = 0.0
        if len(sdt_non) >= 3:
            diffs = sdt_non.diff().dropna()
            if len(diffs) > 0:
                nonneg = float((diffs >= pd.Timedelta(0)).mean())
                nonpos = float((diffs <= pd.Timedelta(0)).mean())
                mono_rate = max(nonneg, nonpos)

        # combine parse success + monotonic-ish (bounded to [0,1])
        score = (pct / 100.0) * 0.7 + mono_rate * 0.3
        time_candidates.append((
            score,
            c,
            {
                "parse_success_pct": float(pct),
                "non_null": nn,
                "non_null_rate": float(nn / len(df)) if len(df) else 0.0,
                "monotonic_rate": float(mono_rate),
            }
        ))

    time_candidates.sort(key=lambda t: t[0], reverse=True)
    for score, c, ev in time_candidates[:5]:
        _add_signal(
            signals,
            kind="time_col_candidate",
            columns=[c],
            score=float(min(1.0, score)),
            metric="time_candidate_score",
            evidence=ev,
        )

    # -----------------------------
    # 4) Dependency checks (name-heuristic, deterministic)
    # -----------------------------
    name_map = {_norm_name(c): c for c in df.columns}

    def _find_any(keys: List[str]) -> List[str]:
        out = []
        for nk, orig in name_map.items():
            for k in keys:
                if k in nk:
                    out.append(orig)
                    break
        return out

    # 4a) subtotal + tax ≈ total
    subtotal_cols = _find_any(["subtotal", "sub_total", "subtot"])
    tax_cols = _find_any(["tax", "gst", "vat"])
    total_cols = _find_any(["total", "grand_total", "amount_due", "invoice_total"])

    # (Decision) Only run arithmetic identity checks when we can parse >=60% numeric for each.
    # Why it matters: avoids garbage results on mostly-text money fields.
    for sub in subtotal_cols[:3]:
        sub_num, sub_pct = _to_numeric_series(df[sub])
        if sub_pct < 60.0:
            continue
        for tax in tax_cols[:3]:
            tax_num, tax_pct = _to_numeric_series(df[tax])
            if tax_pct < 60.0:
                continue
            for tot in total_cols[:3]:
                tot_num, tot_pct = _to_numeric_series(df[tot])
                if tot_pct < 60.0:
                    continue

                mask = sub_num.notna() & tax_num.notna() & tot_num.notna()
                n = int(mask.sum())
                if n < ASSOC_MIN_OVERLAP:
                    continue

                lhs = (sub_num[mask].astype("float64") + tax_num[mask].astype("float64"))
                rhs = tot_num[mask].astype("float64")
                diff = (lhs - rhs).abs()

                rel = (diff / (rhs.abs() + 1e-9))
                ok = (diff <= ASSOC_TOL_ABS) | (rel <= ASSOC_TOL_REL)
                ok_rate = float(ok.mean()) if n else 0.0

                _add_signal(
                    signals,
                    kind="dependency_check",
                    columns=[sub, tax, tot],
                    score=ok_rate,
                    metric="pct_rows_satisfying_subtotal_plus_tax_eq_total",
                    evidence={
                        "n": n,
                        "ok_rate": ok_rate,
                        "tol_abs": ASSOC_TOL_ABS,
                        "tol_rel": ASSOC_TOL_REL,
                        "parse_success_pct": {"subtotal": sub_pct, "tax": tax_pct, "total": tot_pct},
                    },
                )

    # 4b) start/end pairing evidence (end >= start)
    start_cols = _find_any(["start_date", "start", "begin", "from_date"])
    end_cols = _find_any(["end_date", "end", "finish", "to_date"])

    for sc in start_cols[:5]:
        sdt, spct = _to_datetime_series(df[sc])
        if spct < 60.0:
            continue
        for ec in end_cols[:5]:
            edt, epct = _to_datetime_series(df[ec])
            if epct < 60.0:
                continue

            mask = sdt.notna() & edt.notna()
            n = int(mask.sum())
            if n < ASSOC_MIN_OVERLAP:
                continue

            ok = (edt[mask] >= sdt[mask])
            ok_rate = float(ok.mean()) if n else 0.0

            _add_signal(
                signals,
                kind="start_end_pair_check",
                columns=[sc, ec],
                score=ok_rate,
                metric="pct_end_ge_start",
                evidence={
                    "n": n,
                    "ok_rate": ok_rate,
                    "parse_success_pct": {"start": spct, "end": epct},
                },
            )

    # 4c) id_time_duplicates: multiple records per id per time (if we have both candidates)
    # choose the strongest pk candidate (if any) and strongest time candidate (if any)
    best_pk = pk_candidates[0][1] if pk_candidates else None
    best_time = time_candidates[0][1] if time_candidates else None
    if best_pk and best_time:
        s_id = df[best_pk].astype("string")
        tdt, tpct = _to_datetime_series(df[best_time])
        mask = s_id.notna() & tdt.notna()
        n = int(mask.sum())
        if n >= ASSOC_MIN_OVERLAP:
            tmp = pd.DataFrame({"id": s_id[mask], "t": tdt[mask]})
            # duplicates on (id,t)
            dup_mask = tmp.duplicated(subset=["id", "t"], keep=False)
            n_dup_rows = int(dup_mask.sum())
            rate = float(n_dup_rows / n) if n else 0.0
            _add_signal(
                signals,
                kind="id_time_duplicates",
                columns=[best_pk, best_time],
                score=rate,
                metric="dup_rows_rate_in_id_time",
                evidence={
                    "id_col": best_pk,
                    "time_col": best_time,
                    "n": n,
                    "dup_rows": n_dup_rows,
                    "dup_rows_rate": rate,
                    "time_parse_success_pct": float(tpct),
                },
            )

    # 4d) Survey scale totals: choose best-fitting aggregate rule (sum / mean / scaled_sum)
    # (Decision) Emit “best-fitting aggregate rule” rather than just sum pass/fail.
    # Why it matters: avoids throwing away real derived totals that aren’t sums.
    items_by_stem: Dict[str, List[str]] = {}
    for c in df.columns:
        stem = _stem_for_items(c)
        if stem:
            items_by_stem.setdefault(stem, []).append(c)

    for stem, item_cols in sorted(items_by_stem.items(), key=lambda kv: (kv[0].lower(), len(kv[1]))):
        if len(item_cols) < 3:
            continue

        total_col = _find_total_col_for_stem(list(df.columns), stem)
        if not total_col:
            continue

        def _item_key(cname: str) -> int:
            m = re.match(r"^.+?(\\d+)\\.r$", str(cname))
            return int(m.group(1)) if m else 10 ** 9

        item_cols_sorted = sorted(item_cols, key=_item_key)

        item_nums = []
        min_parse = 100.0
        for ic in item_cols_sorted:
            s_num, pct = _to_numeric_series(df[ic])
            item_nums.append(s_num.astype("float64"))
            min_parse = min(min_parse, pct)

        tot_num, tot_pct = _to_numeric_series(df[total_col])
        min_parse = min(min_parse, tot_pct)

        if min_parse < 60.0:
            continue

        # compute sum with overlap mask
        sum_items = item_nums[0].copy()
        mask = sum_items.notna()
        for s in item_nums[1:]:
            sum_items = sum_items + s
            mask = mask & s.notna()

        mask = mask & tot_num.notna()
        n = int(mask.sum())
        if n < ASSOC_MIN_OVERLAP:
            continue

        rhs = tot_num[mask].astype("float64")
        k = float(len(item_cols_sorted))

        # hypotheses
        lhs_sum = sum_items[mask]
        lhs_mean = lhs_sum / k if k > 0 else lhs_sum

        # scaled_sum: choose a robust scalar that best maps sum -> total (median ratio)
        scale = None
        lhs_for_scale = lhs_sum.copy()
        denom_mask = lhs_for_scale.notna() & rhs.notna() & (lhs_for_scale.abs() > 1e-12)
        if int(denom_mask.sum()) >= ASSOC_MIN_OVERLAP:
            ratios = (rhs[denom_mask] / lhs_for_scale[denom_mask]).astype("float64")
            try:
                scale = float(ratios.median())
            except Exception:
                scale = None
        lhs_scaled = (lhs_sum * scale) if (scale is not None) else None

        def _ok_rate(lhs: pd.Series) -> float:
            diff = (lhs - rhs).abs()
            rel = (diff / (rhs.abs() + 1e-9))
            ok = (diff <= ASSOC_TOL_ABS) | (rel <= ASSOC_TOL_REL)
            return float(ok.mean()) if n else 0.0

        ok_sum = _ok_rate(lhs_sum)
        ok_mean = _ok_rate(lhs_mean)
        ok_scaled = _ok_rate(lhs_scaled) if lhs_scaled is not None else -1.0

        candidates = [
            ("sum", ok_sum),
            ("mean", ok_mean),
        ]
        if lhs_scaled is not None:
            candidates.append(("scaled_sum", ok_scaled))

        best_rule, best_ok = max(candidates, key=lambda t: t[1])

        evidence = {
            "stem": stem,
            "n": n,
            "tol_abs": ASSOC_TOL_ABS,
            "tol_rel": ASSOC_TOL_REL,
            "parse_success_floor_pct": float(min_parse),
            "total_col": total_col,
            "item_cols": item_cols_sorted,
            "ok_rates": {"sum": float(ok_sum), "mean": float(ok_mean),
                         "scaled_sum": float(ok_scaled) if lhs_scaled is not None else None},
            "best_rule": best_rule,
            "best_ok_rate": float(best_ok),
            "scaled_sum_factor": float(scale) if scale is not None else None,
        }

        _add_signal(
            signals,
            kind="dependency_check",
            columns=item_cols_sorted + [total_col],
            score=float(best_ok),
            metric="best_fitting_aggregate_rule_ok_rate",
            evidence=evidence,
        )

    return {
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "n_rows_scanned": n_rows_scanned,
        "limits": {
            "ASSOC_MAX_ROWS": ASSOC_MAX_ROWS,
            "ASSOC_TOP_K_PAIRS": ASSOC_TOP_K_PAIRS,
            "ASSOC_MAX_NUMERIC_COLS": ASSOC_MAX_NUMERIC_COLS,
            "ASSOC_MAX_CAT_COLS": ASSOC_MAX_CAT_COLS,
            "ASSOC_MAX_CAT_CARD": ASSOC_MAX_CAT_CARD,
            "ASSOC_MIN_OVERLAP": ASSOC_MIN_OVERLAP,
            "ASSOC_TOL_ABS": ASSOC_TOL_ABS,
            "ASSOC_TOL_REL": ASSOC_TOL_REL,
        },
        "signals": signals,
    }


class XlsxExportRequest(BaseModel):
    rows_table_json: str
    filename: str | None = None


class LightContractXlsxRequest(BaseModel):
    run_id: str
    grain_worker_output: Dict[str, Any]
    filename: str | None = None


class LightContractFinalizeAcceptedRequest(BaseModel):
    run_id: str


class CanonicalColumnContractRequest(BaseModel):
    run_id: str
    light_contract_decisions: Any
    semantic_context_json: Any
    type_transform_worker_json: Any
    missingness_worker_json: Any
    family_worker_json: Any
    table_layout_worker_json: Any
    scale_mapping_json: Any = None
    debug: bool = False


class ScaleMappingResolverRequest(BaseModel):
    run_id: str
    light_contract_decisions: Any
    family_worker_json: Any
    scale_mapping_extractor_json: Any = None
    debug: bool = False


def _sanitize_export_filename(name: str, default_name: str) -> str:
    name = (name or "").strip().replace("\n", "").replace("\r", "")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".xlsx"):
        name += ".xlsx"
    return name or default_name


def _upload_xlsx_and_sign(xlsx_bytes: bytes, filename: str) -> Dict[str, Any]:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    ttl_minutes = int(os.getenv("EXPORT_SIGNED_URL_TTL_MINUTES", "30"))
    object_name = f"exports/{uuid4().hex}_{filename}"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        blob.upload_from_string(
            xlsx_bytes,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        signing_sa = os.getenv("SIGNING_SA_EMAIL")
        if not signing_sa:
            raise HTTPException(status_code=500, detail="Missing SIGNING_SA_EMAIL env var")

        source_creds, _ = google.auth.default()
        signing_creds = impersonated_credentials.Credentials(
            source_credentials=source_creds,
            target_principal=signing_sa,
            target_scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
            lifetime=3600,
        )

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=ttl_minutes),
            method="GET",
            response_disposition=f'attachment; filename="{filename}"',
            response_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            credentials=signing_creds,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS upload/sign failed: {e}") from e

    return {
        "status": "ok",
        "filename": filename,
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "bucket": bucket_name,
        "object_path": object_name,
        "signed_url": signed_url,
        "expires_in_minutes": ttl_minutes,
    }


def _upload_json_to_run_object(run_id: str, object_name: str, payload: Dict[str, Any]) -> str:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"runs/{run_id}/{object_name}")
        blob.upload_from_string(_json_bytes(payload), content_type="application/json")
        return blob.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist light contract context: {exc}") from exc


def _upload_bytes_to_run_object(run_id: str, object_name: str, payload: bytes, *, content_type: str) -> str:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"runs/{run_id}/{object_name}")
        blob.upload_from_string(payload, content_type=content_type)
        return blob.name
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist run object: {exc}") from exc


def _load_json_from_run_object(run_id: str, object_name: str) -> Dict[str, Any]:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"runs/{run_id}/{object_name}")
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail=f"{object_name} not found for run_id")
        return json.loads(blob.download_as_bytes().decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load light contract context: {exc}") from exc


def _load_bytes_from_run_object(run_id: str, object_name: str) -> bytes:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"runs/{run_id}/{object_name}")
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail=f"{object_name} not found for run_id")
        return blob.download_as_bytes()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load run object: {exc}") from exc


def _sign_run_object_download(
    run_id: str,
    object_name: str,
    *,
    filename: str,
    content_type: str,
) -> str:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    ttl_minutes = int(os.getenv("EXPORT_SIGNED_URL_TTL_MINUTES", "30"))
    object_path = f"runs/{run_id}/{object_name}"
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail=f"{object_name} not found for run_id")

        signing_sa = os.getenv("EXPORT_SIGNER_SERVICE_ACCOUNT")
        if not signing_sa:
            raise HTTPException(status_code=500, detail="Missing EXPORT_SIGNER_SERVICE_ACCOUNT env var")

        source_creds, _ = google.auth.default()
        signing_creds = impersonated_credentials.Credentials(
            source_credentials=source_creds,
            target_principal=signing_sa,
            target_scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
            lifetime=3600,
        )

        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=ttl_minutes),
            method="GET",
            response_disposition=f'attachment; filename="{filename}"',
            response_type=content_type,
            credentials=signing_creds,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to sign run object download: {exc}") from exc


def _persist_light_contract_decisions(run_id: str, handoff: Dict[str, Any], source: str) -> Tuple[Dict[str, Any], str]:
    decisions = dict(handoff)
    decisions["run_id"] = run_id
    decisions["source"] = source
    decisions["finalized_at"] = pd.Timestamp.utcnow().isoformat()
    decisions.setdefault("parse_validation", {})
    object_path = _upload_json_to_run_object(run_id, "light_contract_decisions.json", decisions)
    return decisions, object_path


def _load_optional_json_from_run_object(run_id: str, object_name: str) -> Optional[Dict[str, Any]]:
    try:
        return _load_json_from_run_object(run_id, object_name)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


def _extract_codebook_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    if PdfReader is None:
        raise HTTPException(
            status_code=500,
            detail="pypdf is not installed, so PDF codebook extraction is unavailable",
        )

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse codebook PDF: {exc}") from exc

    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        normalized = re.sub(r"\s+", " ", text).strip()
        pages.append(
            {
                "page_number": idx,
                "text": normalized,
                "char_count": len(normalized),
            }
        )
    return pages


def _preview_values_from_a2_row(a2_row: Dict[str, Any], *, limit: int = 8) -> List[str]:
    values: List[str] = []
    for value in (a2_row.get("top_levels") or []):
        text = str(value).strip()
        if text:
            values.append(text)
    sample_groups = a2_row.get("a2_samples") or {}
    if isinstance(sample_groups, dict):
        for bucket in ("head", "random", "tail"):
            for value in sample_groups.get(bucket) or []:
                text = str(value).strip()
                if text:
                    values.append(text)
    return _dedupe_preserve_order(values)[:max(0, limit)]


def _semantic_text_has_mapping_cues(text: Any) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    cues = [
        "likert",
        "familiarity scale",
        "agreement scale",
        "frequency scale",
        "rating scale",
        "ordered labels",
        "strongly agree",
        "strongly disagree",
        "very familiar",
        "never heard",
        "label mapping",
        "1=",
        "2=",
        "3=",
    ]
    return any(cue in lowered for cue in cues)


def _reverse_family_column_map(family_by_column: Dict[str, str]) -> Dict[str, List[str]]:
    reverse: Dict[str, List[str]] = {}
    for column, family_id in family_by_column.items():
        if not family_id:
            continue
        reverse.setdefault(family_id, []).append(column)
    for family_id in reverse:
        reverse[family_id] = sorted(reverse[family_id])
    return reverse


def _looks_scale_like_values(values: Iterable[Any]) -> bool:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return False
    lowered = [value.lower() for value in cleaned]
    return (
        any(any(token in value for token in LIKERT_TOKENS) for value in lowered)
        or any(re.search(r"\b(?:never heard|very familiar|strongly agree|strongly disagree|neutral|somewhat)\b", value) for value in lowered)
        or any(re.search(r"\b\d+\b", value) for value in cleaned)
    )


def _infer_scale_kind(values: Iterable[Any], semantic_text: str) -> str:
    blob = f"{semantic_text or ''} {' '.join(str(v) for v in values)}".lower()
    if "familiar" in blob or "never heard" in blob:
        return "familiarity_scale"
    if "agree" in blob or "disagree" in blob:
        return "agreement_scale"
    if "frequency" in blob or "often" in blob or "rarely" in blob or "never" in blob:
        return "frequency_scale"
    if "satisf" in blob or "rating" in blob:
        return "rating_scale"
    return "ordinal_scale"


def _extract_numeric_suffix(value: str) -> Optional[float]:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*$", str(value or "").strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _deterministic_scale_inference(
    *,
    target_kind: str,
    target_id: str,
    observed_values: List[str],
    semantic_text: str,
) -> Optional[Dict[str, Any]]:
    values = _dedupe_preserve_order(observed_values)
    if len(values) < 2 or not _looks_scale_like_values(values):
        return None

    numeric_pairs: List[Tuple[float, str]] = []
    for value in values:
        numeric = _extract_numeric_suffix(value)
        if numeric is None:
            numeric_pairs = []
            break
        numeric_pairs.append((numeric, value))

    ordered_labels = values
    label_to_numeric_score: Dict[str, float] = {}
    if numeric_pairs and len({pair[0] for pair in numeric_pairs}) == len(values):
        ordered_labels = [label for _, label in sorted(numeric_pairs, key=lambda item: item[0])]
        label_to_numeric_score = {label: score for score, label in sorted(numeric_pairs, key=lambda item: item[0])}

    return {
        "target_kind": target_kind,
        "target_id": target_id,
        "mapping_status": "deterministic_inferred",
        "response_scale_kind": _infer_scale_kind(ordered_labels, semantic_text),
        "ordered_labels": ordered_labels,
        "label_to_ordinal_position": {label: idx + 1 for idx, label in enumerate(ordered_labels)},
        "label_to_numeric_score": label_to_numeric_score,
        "numeric_score_semantics_confirmed": False,
        "source": "deterministic_safe_inference",
        "notes": "Deterministic inference from observed ordered labels; numeric scoring remains advisory until human or codebook confirmation.",
        "confidence": 0.72 if label_to_numeric_score else 0.64,
    }


def _normalize_scale_mapping_contract(
    payload: Any,
    *,
    known_columns: Set[str],
    known_family_ids: Set[str],
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"mappings": [], "by_target": {}, "extra_targets": set()}

    by_target: Dict[Tuple[str, str], Dict[str, Any]] = {}
    extra_targets: Set[str] = set()
    mappings: List[Dict[str, Any]] = []

    for mapping in _coerce_list_of_dicts(payload.get("mappings")):
        target_kind = str(mapping.get("target_kind") or "").strip()
        target_id = str(mapping.get("target_id") or "").strip()
        if target_kind not in {"family", "column"} or not target_id:
            continue
        if target_kind == "family" and target_id not in known_family_ids:
            extra_targets.add(f"family:{target_id}")
            continue
        if target_kind == "column" and target_id not in known_columns:
            extra_targets.add(f"column:{target_id}")
            continue

        ordered_labels = [
            str(label).strip()
            for label in (mapping.get("ordered_labels") or [])
            if str(label or "").strip()
        ]
        ordinal_map = mapping.get("label_to_ordinal_position") if isinstance(mapping.get("label_to_ordinal_position"), dict) else {}
        numeric_map = mapping.get("label_to_numeric_score") if isinstance(mapping.get("label_to_numeric_score"), dict) else {}

        normalized = {
            "target_kind": target_kind,
            "target_id": target_id,
            "mapping_status": str(mapping.get("mapping_status") or "unresolved").strip(),
            "response_scale_kind": str(mapping.get("response_scale_kind") or "").strip(),
            "ordered_labels": ordered_labels,
            "label_to_ordinal_position": {
                str(label).strip(): int(position)
                for label, position in ordinal_map.items()
                if str(label or "").strip() and isinstance(position, (int, float)) and not isinstance(position, bool)
            },
            "label_to_numeric_score": {
                str(label).strip(): float(score)
                for label, score in numeric_map.items()
                if str(label or "").strip() and isinstance(score, (int, float)) and not isinstance(score, bool)
            },
            "numeric_score_semantics_confirmed": bool(mapping.get("numeric_score_semantics_confirmed", False)),
            "source": str(mapping.get("source") or "").strip(),
            "notes": str(mapping.get("notes") or "").strip(),
            "confidence": round(max(0.0, min(1.0, _safe_float(mapping.get("confidence"), 0.0))), 6),
        }
        mappings.append(normalized)
        by_target[(target_kind, target_id)] = normalized

    return {
        "mappings": mappings,
        "by_target": by_target,
        "extra_targets": extra_targets,
    }


def _validate_scale_mapping_contract(
    payload: Any,
    *,
    known_columns: Set[str],
    known_family_ids: Set[str],
) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["scale mapping payload must be an object"]

    for key in ("worker", "summary", "mappings", "review_flags", "assumptions"):
        if key not in payload:
            errors.append(f"Missing required top-level key: {key}")
    if payload.get("worker") != "scale_mapping_extractor" and payload.get("worker") != "scale_mapping_resolver":
        errors.append("worker must be 'scale_mapping_extractor' or 'scale_mapping_resolver'")

    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        errors.append("mappings must be an array")
        return errors

    seen: Set[Tuple[str, str]] = set()
    for idx, mapping in enumerate(mappings):
        path = f"mappings[{idx}]"
        if not isinstance(mapping, dict):
            errors.append(f"{path} must be an object")
            continue
        for key in (
            "target_kind",
            "target_id",
            "mapping_status",
            "response_scale_kind",
            "ordered_labels",
            "label_to_ordinal_position",
            "label_to_numeric_score",
            "numeric_score_semantics_confirmed",
            "source",
            "notes",
            "confidence",
        ):
            if key not in mapping:
                errors.append(f"{path}.{key} is required")
        target_kind = str(mapping.get("target_kind") or "").strip()
        target_id = str(mapping.get("target_id") or "").strip()
        if target_kind not in {"family", "column"}:
            errors.append(f"{path}.target_kind must be family or column")
        if not target_id:
            errors.append(f"{path}.target_id must be non-empty")
        target_key = (target_kind, target_id)
        if target_key in seen:
            errors.append(f"Duplicate mapping target: {target_kind}:{target_id}")
        else:
            seen.add(target_key)
        if target_kind == "family" and target_id and target_id not in known_family_ids:
            errors.append(f"{path}.target_id does not resolve to a known family_id")
        if target_kind == "column" and target_id and target_id not in known_columns:
            errors.append(f"{path}.target_id does not resolve to a known source column")
        if str(mapping.get("mapping_status") or "").strip() not in {
            "human_confirmed",
            "codebook_confirmed",
            "deterministic_inferred",
            "unresolved",
        }:
            errors.append(f"{path}.mapping_status is invalid")
        ordered_labels = mapping.get("ordered_labels")
        if not isinstance(ordered_labels, list):
            errors.append(f"{path}.ordered_labels must be an array")
        elif ordered_labels:
            for label_idx, label in enumerate(ordered_labels):
                if not isinstance(label, str) or not label.strip():
                    errors.append(f"{path}.ordered_labels[{label_idx}] must be a non-empty string")
        ordinal_map = mapping.get("label_to_ordinal_position")
        if not isinstance(ordinal_map, dict):
            errors.append(f"{path}.label_to_ordinal_position must be an object")
        numeric_map = mapping.get("label_to_numeric_score")
        if not isinstance(numeric_map, dict):
            errors.append(f"{path}.label_to_numeric_score must be an object")
        if not isinstance(mapping.get("numeric_score_semantics_confirmed"), bool):
            errors.append(f"{path}.numeric_score_semantics_confirmed must be boolean")
        confidence = mapping.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            errors.append(f"{path}.confidence must be numeric")
        elif not (0.0 <= float(confidence) <= 1.0):
            errors.append(f"{path}.confidence must be between 0 and 1")
    return errors


def _build_scale_mapping_bundle(
    *,
    run_id: str,
    light_contract_decisions: Dict[str, Any],
    family_worker_json: Any,
) -> Dict[str, Any]:
    support = _load_canonical_support_artifacts(run_id)
    family_results_by_id = _extract_family_results(family_worker_json)
    light_contract_maps = _normalize_light_contract_maps(light_contract_decisions)
    family_by_column = dict(support["family_by_column"])
    reverse_family_map = _reverse_family_column_map(family_by_column)

    structured_human_mappings = _normalize_light_contract_scale_mapping_input(
        light_contract_decisions.get("scale_mapping_input") or light_contract_decisions.get("scale_mapping_rows") or []
    )
    semantic_context_input = light_contract_decisions.get("semantic_context_input") or {}
    raw_semantic_notes = {
        "dataset_context_and_collection_notes": str(semantic_context_input.get("dataset_context_and_collection_notes") or "").strip(),
        "semantic_codebook_and_important_variables": str(semantic_context_input.get("semantic_codebook_and_important_variables") or "").strip(),
    }

    accepted_family_ids = _dedupe_preserve_order(
        list(family_results_by_id.keys())
        + list((light_contract_maps.get("family_by_id") or {}).keys())
        + list(reverse_family_map.keys())
    )

    accepted_families: List[Dict[str, Any]] = []
    for family_id in accepted_family_ids:
        family_result = family_results_by_id.get(family_id) or {}
        member_columns = reverse_family_map.get(family_id, [])
        observed_values: List[str] = []
        for column in member_columns[:6]:
            a2_row = support["a2_by_col"].get(column) or {}
            observed_values.extend(_preview_values_from_a2_row(a2_row, limit=4))
        accepted_families.append(
            {
                "family_id": family_id,
                "recommended_family_role": str(family_result.get("recommended_family_role") or "").strip(),
                "member_semantics_notes": str(family_result.get("member_semantics_notes") or "").strip(),
                "member_columns_preview": member_columns[:8],
                "member_column_count": len(member_columns),
                "observed_value_preview": _dedupe_preserve_order(observed_values)[:12],
            }
        )

    candidate_standalone_columns: List[Dict[str, Any]] = []
    for a2_row in support["a2_rows"]:
        column = str(a2_row.get("column") or "").strip()
        if not column or column in family_by_column:
            continue
        preview_values = _preview_values_from_a2_row(a2_row, limit=8)
        unique_count = int(a2_row.get("unique_count") or 0)
        if unique_count > 12 or not _looks_scale_like_values(preview_values):
            continue
        candidate_standalone_columns.append(
            {
                "column": column,
                "observed_value_preview": preview_values,
                "unique_count": unique_count,
                "top_candidate_type": str((a2_row.get("top_candidate") or {}).get("type") or "").strip(),
            }
        )
        if len(candidate_standalone_columns) >= 24:
            break

    codebook_document = _load_optional_json_from_run_object(run_id, "codebook_document.json") or {}
    codebook_pages = _load_optional_json_from_run_object(run_id, "codebook_pages.json") or {}
    page_records = _coerce_list_of_dicts(codebook_pages.get("pages"))

    query_terms = _dedupe_preserve_order(
        [mapping.get("target_id") for mapping in structured_human_mappings]
        + [mapping.get("response_scale_kind") for mapping in structured_human_mappings]
        + [label for mapping in structured_human_mappings for label in (mapping.get("ordered_labels") or [])]
        + [family.get("family_id") for family in accepted_families]
        + [column.get("column") for column in candidate_standalone_columns]
        + _split_mapping_text_list(raw_semantic_notes.get("semantic_codebook_and_important_variables"), allow_comma_fallback=False)
    )

    relevant_page_snippets: List[Dict[str, Any]] = []
    if page_records and query_terms:
        scored_pages: List[Tuple[int, Dict[str, Any], List[str]]] = []
        for page in page_records:
            text = str(page.get("text") or "")
            lowered = text.lower()
            matched_terms = [term for term in query_terms if term and term.lower() in lowered]
            if matched_terms:
                scored_pages.append((len(matched_terms), page, matched_terms[:8]))
        for _, page, matched_terms in sorted(scored_pages, key=lambda item: (-item[0], int(item[1].get("page_number") or 0)))[:6]:
            text = str(page.get("text") or "")
            snippet = text[:1600]
            relevant_page_snippets.append(
                {
                    "page_number": int(page.get("page_number") or 0),
                    "matched_terms": matched_terms,
                    "snippet": snippet,
                }
            )

    codebook_present = bool(codebook_document.get("object_path"))
    has_mapping_evidence = bool(
        structured_human_mappings
        or codebook_present
        or _semantic_text_has_mapping_cues(raw_semantic_notes.get("semantic_codebook_and_important_variables"))
    )

    return {
        "bundle_version": "1",
        "has_mapping_evidence": has_mapping_evidence,
        "structured_human_mappings": structured_human_mappings,
        "raw_semantic_notes": raw_semantic_notes,
        "accepted_families": accepted_families,
        "candidate_standalone_columns": candidate_standalone_columns,
        "codebook_context": {
            "present": codebook_present,
            "document_filename": str(codebook_document.get("filename") or "").strip(),
            "document_object_path": str(codebook_document.get("object_path") or "").strip(),
            "document_signed_url": str(codebook_document.get("signed_url") or "").strip(),
            "page_count": int(codebook_document.get("page_count") or 0),
            "relevant_page_snippets": relevant_page_snippets,
        },
    }


def _first_non_empty(items: List[Any]) -> str:
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            return text
    return ""


def _coerce_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_key_triplet(keys: Any) -> List[str]:
    if not isinstance(keys, list):
        return ["", "", ""]
    cleaned = [str(k).strip() for k in keys if str(k).strip()]
    return (cleaned + ["", "", ""])[:3]


def _keys_to_label(keys: Any) -> str:
    parts = [str(k).strip() for k in (keys or []) if str(k).strip()]
    return " + ".join(parts)


def _family_id_from_table_name(table_name: Any) -> str:
    text = str(table_name or "").strip().lower()
    if not text:
        return ""
    for prefix in ("survey_", "family_", "dim_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for suffix in ("_rows", "_row", "_table", "_child", "_family"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return "_".join(_tokenize_col_name(text))


def _load_run_manifest(run_id: str) -> Dict[str, Any]:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"runs/{run_id}/manifest.json")
        if not blob.exists(client):
            raise HTTPException(status_code=404, detail="run_id not found")
        return json.loads(blob.download_as_bytes().decode("utf-8"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to load run manifest: {exc}") from exc


def _column_order_from_manifest_and_a2(run_id: str) -> List[str]:
    manifest = _load_run_manifest(run_id)
    evidence_fields = ((manifest.get("evidence_primitives") or {}).get("fields") or [])
    try:
        kind, a2_payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A2")
    except HTTPException:
        kind, a2_payload = "other", None
    columns: List[str] = []
    if kind == "jsonl" and isinstance(a2_payload, list):
        for row in a2_payload:
            if isinstance(row, dict):
                col = str(row.get("column") or "").strip()
                if col:
                    columns.append(col)
    if columns:
        return columns
    manifest_cols = (((manifest.get("profiling_limits") or {}).get("column_names")) or [])
    if isinstance(manifest_cols, list):
        return [str(col).strip() for col in manifest_cols if str(col).strip()]
    return [str(col).strip() for col in evidence_fields if str(col).strip()]


def _family_column_map(run_id: str) -> Dict[str, str]:
    try:
        kind, a8_payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A8")
    except HTTPException:
        return {}
    if kind != "json" or not isinstance(a8_payload, dict):
        return {}
    mapping: Dict[str, str] = {}
    for family in _coerce_list_of_dicts(a8_payload.get("families")):
        family_id = _fallback_family_id(
            family.get("family_id"),
            [str(col) for col in (family.get("columns") or [])],
            ((family.get("stem_evidence") or {}).get("raw_stem") if isinstance(family.get("stem_evidence"), dict) else None),
        )
        for col in family.get("columns", []) or []:
            col_name = str(col).strip()
            if col_name and col_name not in mapping:
                mapping[col_name] = family_id
    return mapping


def _is_numeric_like_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text))


def _load_structural_gate_rows(run_id: str) -> List[Dict[str, Any]]:
    try:
        kind, a16_payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A16")
    except HTTPException as exc:
        if exc.status_code == 404:
            return []
        raise
    if kind != "json" or not isinstance(a16_payload, dict):
        return []

    rows: List[Dict[str, Any]] = []
    master_switches = _coerce_list_of_dicts(a16_payload.get("master_switch_candidates"))
    for candidate in master_switches[:5]:
        top_values = [str(v) for v in (candidate.get("top_trigger_values") or [])[:3] if str(v).strip()]
        explained_cols = int(candidate.get("explained_column_count") or 0)
        family_ids = [str(v) for v in (candidate.get("affected_family_ids") or []) if str(v).strip()]
        if top_values and all(_is_numeric_like_text(v) for v in top_values):
            continue
        if explained_cols < 3 and not family_ids:
            continue
        rows.append({
            "trigger_column": str(candidate.get("trigger_column") or ""),
            "trigger_value": ", ".join(top_values),
            "affected_column_count": explained_cols,
            "affected_family_ids": family_ids,
            "missing_explained_pct": "",
            "directionality": "master_switch_candidate",
            "interpretation": str(candidate.get("interpretation") or "Likely structural gate controlling downstream question blocks."),
        })

    if rows:
        return rows

    for rule in _coerce_list_of_dicts(a16_payload.get("detected_skip_logic"))[:5]:
        trigger_value = str(rule.get("trigger_value") or "")
        family_ids = [str(v) for v in (rule.get("affected_family_ids") or []) if str(v).strip()]
        affected_count = int(rule.get("affected_column_count") or 0)
        if _is_numeric_like_text(trigger_value):
            continue
        if affected_count < 3 and not family_ids:
            continue
        rows.append({
            "trigger_column": str(rule.get("trigger_column") or ""),
            "trigger_value": trigger_value,
            "affected_column_count": affected_count,
            "affected_family_ids": family_ids,
            "missing_explained_pct": rule.get("missing_explained_pct", ""),
            "directionality": str(rule.get("directionality") or ""),
            "interpretation": str(rule.get("interpretation") or ""),
        })
    return rows


def _build_grain_summary_rows(grain_worker_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    primary = grain_worker_output.get("recommended_primary_grain") or {}
    if isinstance(primary, dict):
        rows.append({
            "topic": "Primary grain",
            "recommendation": _keys_to_label(primary.get("keys")) or str(primary.get("description") or ""),
            "why": str(primary.get("justification") or primary.get("description") or ""),
            "needs_review": "yes",
        })

    diagnostics = grain_worker_output.get("diagnostics") or {}
    if isinstance(diagnostics, dict):
        rows.append({
            "topic": "Encoding pattern",
            "recommendation": str(diagnostics.get("encoding_hint") or "unknown"),
            "why": str(diagnostics.get("encoding_justification") or diagnostics.get("recommended_next_action") or ""),
            "needs_review": "yes" if str(diagnostics.get("encoding_hint") or "") == "unknown" else "no",
        })
        rejected = _coerce_list_of_dicts(diagnostics.get("rejected_primary_candidates"))
        for candidate in rejected[:3]:
            rows.append({
                "topic": "Rejected primary candidate",
                "recommendation": _keys_to_label(candidate.get("keys")),
                "why": str(candidate.get("reason") or ""),
                "needs_review": "no",
            })

    for ref in _candidate_reference_tables(grain_worker_output)[:3]:
        supporting_attributes = _sorted_nonempty_strings(ref.get("supporting_attributes") or [])
        why_parts = [
            str(ref.get("justification") or "").strip(),
            str(ref.get("why_not_base_attribute") or "").strip(),
        ]
        if supporting_attributes:
            why_parts.append(f"Supporting attributes: {', '.join(supporting_attributes[:4])}")
        rows.append({
            "topic": "Candidate reference",
            "recommendation": _first_non_empty([ref.get("suggested_table_name"), _keys_to_label(ref.get("keys"))]),
            "why": " ".join(bit for bit in why_parts if bit).strip() or str(ref.get("entity_description") or ""),
            "needs_review": "yes",
        })

    family_candidates = _coerce_list_of_dicts(grain_worker_output.get("family_review_candidates"))
    if family_candidates:
        rows.append({
            "topic": "Repeat family overview",
            "recommendation": f"{len(family_candidates)} family candidates detected",
            "why": "Review the Repeat Families sheet to confirm child-table handling, parent keys, and repeat index naming.",
            "needs_review": "yes",
        })

    assumptions = _coerce_list_of_dicts(grain_worker_output.get("assumptions"))
    for assumption in assumptions[:2]:
        assumption_text = str(assumption.get("assumption") or "")
        lower = assumption_text.lower()
        target = "Overrides"
        if any(token in lower for token in ("grain", "key", "row", "respondent")):
            target = "Primary Grain"
        elif any(token in lower for token in ("family", "repeat", "matrix", "rown")):
            target = "Repeat Families"
        rows.append({
            "topic": "Open question",
            "recommendation": assumption_text,
            "why": f"{str(assumption.get('explanation') or '')} Respond on the {target} sheet.".strip(),
            "needs_review": "yes" if assumption.get("needs_user_validation") else "no",
        })
    return rows


def _build_primary_grain_rows(grain_worker_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    primary = grain_worker_output.get("recommended_primary_grain") or {}
    primary_keys = _as_key_triplet((primary.get("keys") or []) if isinstance(primary, dict) else [])
    return [{
        "item": "primary_grain",
        "recommended_key_1": primary_keys[0],
        "recommended_key_2": primary_keys[1],
        "recommended_key_3": primary_keys[2],
        "your_key_1": "",
        "your_key_2": "",
        "your_key_3": "",
        "status": "accept",
        "comments": str(primary.get("description") or ""),
    }]


def _candidate_reference_tables(grain_worker_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_reference_tables = _coerce_list_of_dicts(grain_worker_output.get("candidate_reference_tables"))
    if candidate_reference_tables:
        return candidate_reference_tables
    return _coerce_list_of_dicts(grain_worker_output.get("candidate_dimension_tables"))


def _build_reference_rows(grain_worker_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ref in _candidate_reference_tables(grain_worker_output):
        keys = _as_key_triplet(ref.get("keys") or [])
        supporting_attributes = _sorted_nonempty_strings(ref.get("supporting_attributes") or [])
        comments_bits = [
            str(ref.get("justification") or "").strip(),
            str(ref.get("why_not_base_attribute") or "").strip(),
        ]
        if supporting_attributes:
            comments_bits.append(f"Supporting attributes: {', '.join(supporting_attributes[:6])}")
        rows.append({
            "table_name": _first_non_empty([ref.get("suggested_table_name"), _keys_to_label(ref.get("keys"))]),
            "reference_kind": str(ref.get("reference_kind") or ""),
            "recommended_key_1": keys[0],
            "recommended_key_2": keys[1],
            "recommended_key_3": keys[2],
            "your_key_1": "",
            "your_key_2": "",
            "your_key_3": "",
            "relationship_to_primary": str(ref.get("relationship_to_primary") or ""),
            "supporting_attributes_preview": ", ".join(supporting_attributes[:6]),
            "status": "unsure" if "review" in str(ref.get("justification") or "").lower() else "accept",
            "comments": " ".join(bit for bit in comments_bits if bit).strip() or str(ref.get("entity_description") or ""),
        })
    return rows


def _light_contract_reference_rows(source_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _coerce_list_of_dicts(source_payload.get("reference_rows") or source_payload.get("dimension_rows"))


def _reference_decisions_from_rows(rows: List[Dict[str, Any]], key_mode: str) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "").strip().lower() or "unsure"
        if key_mode == "your_if_modify":
            keys = _keys_from_row(row, "your") if status == "modify" else _keys_from_row(row, "recommended")
        else:
            keys = _keys_from_row(row, "recommended")
        decisions.append({
            "table_name": str(row.get("table_name") or ""),
            "status": status if key_mode == "your_if_modify" else str(row.get("status") or "accept"),
            "keys": keys,
            "relationship_to_primary": str(row.get("relationship_to_primary") or ""),
            "reference_kind": str(row.get("reference_kind") or ""),
            "comments": str(row.get("comments") or ""),
        })
    return decisions


def _build_repeat_family_rows(grain_worker_output: Dict[str, Any], structural_gate_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    primary = grain_worker_output.get("recommended_primary_grain") or {}
    primary_label = _keys_to_label((primary.get("keys") or []) if isinstance(primary, dict) else [])
    gate_map: Dict[str, List[str]] = {}
    for gate in structural_gate_rows:
        trigger = _first_non_empty([gate.get("trigger_column"), gate.get("trigger_value")])
        for family_id in gate.get("affected_family_ids", []) or []:
            gate_map.setdefault(str(family_id), []).append(trigger)
    rows: List[Dict[str, Any]] = []
    for family in _coerce_list_of_dicts(grain_worker_output.get("family_review_candidates")):
        family_id = _fallback_family_id(
            family.get("family_id"),
            [str(col) for col in (family.get("source_columns") or []) if str(col).strip()],
            _family_id_from_table_name(family.get("suggested_table_name")),
        )
        gate_context = gate_map.get(family_id, [])
        comment = str(family.get("why_review") or "")
        if gate_context:
            gate_text = ", ".join(gate_context[:3])
            comment = f"{comment} Structural gate context: condition(s) involving {gate_text} may control whether this family appears.".strip()
        rows.append({
            "family_id": family_id,
            "recommended_table_name": str(family.get("suggested_table_name") or ""),
            "your_table_name": "",
            "recommended_repeat_index_name": str(family.get("repeat_index_name") or ""),
            "your_repeat_index_name": "",
            "recommended_parent_key": primary_label,
            "your_parent_key": "",
            "status": "accept" if str(family.get("status") or "") == "confirm_detected" else "unsure",
            "comments": comment,
        })
    return rows


def _build_override_rows(grain_worker_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_descriptors = grain_worker_output.get("user_inputs_requested") or {}
    rows: List[Dict[str, Any]] = []
    for default in DEFAULT_OVERRIDE_FIELDS:
        descriptor = input_descriptors.get(default["field"]) if isinstance(input_descriptors, dict) else None
        purpose = ""
        if isinstance(descriptor, dict):
            purpose = str(descriptor.get("purpose") or "").strip()
        desc = default["description"] + (f" Purpose: {purpose}." if purpose else "")
        rows.append({
            "field": default["field"],
            "description": desc,
            "user_input": "",
        })
    return rows


def _build_scale_mapping_rows(column_guide_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_family_ids: Set[str] = set()
    for row in column_guide_rows:
        family_id = str(row.get("family_id") or "").strip()
        if not family_id or family_id in seen_family_ids:
            continue
        seen_family_ids.add(family_id)
        rows.append(
            {
                "target_kind": "family",
                "target_id": family_id,
                "response_scale_kind": "",
                "ordered_labels_low_to_high": "",
                "numeric_scores_low_to_high": "",
                "notes": "",
            }
        )
    return rows


def _keys_from_row(row: Dict[str, Any], prefix: str) -> List[str]:
    return [
        str(row.get(f"{prefix}_key_{idx}") or "").strip()
        for idx in (1, 2, 3)
        if str(row.get(f"{prefix}_key_{idx}") or "").strip()
    ]


def _extract_semantic_context_input(override_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    override_lookup = {
        str(row.get("field") or "").strip(): str(row.get("user_input") or "")
        for row in (override_rows or [])
        if str(row.get("field") or "").strip()
    }
    return {
        "dataset_context_and_collection_notes": override_lookup.get("dataset_context_and_collection_notes", ""),
        "semantic_codebook_and_important_variables": override_lookup.get("semantic_codebook_and_important_variables", ""),
    }


def _split_mapping_text_list(raw_value: Any, *, allow_comma_fallback: bool = False) -> List[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    if "|" in text:
        parts = text.split("|")
    elif "\n" in text:
        parts = text.splitlines()
    elif ";" in text:
        parts = text.split(";")
    elif allow_comma_fallback and "," in text:
        parts = text.split(",")
    else:
        parts = [text]
    return [str(part).strip() for part in parts if str(part).strip()]


def _normalize_light_contract_scale_mapping_input(scale_mapping_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_targets: Set[Tuple[str, str]] = set()
    for row in scale_mapping_rows or []:
        if not isinstance(row, dict):
            continue

        target_kind = str(row.get("target_kind") or "").strip().lower()
        target_id = str(row.get("target_id") or "").strip()
        response_scale_kind = str(row.get("response_scale_kind") or "").strip()
        if isinstance(row.get("ordered_labels"), list):
            ordered_labels = [str(value).strip() for value in (row.get("ordered_labels") or []) if str(value or "").strip()]
        else:
            ordered_labels = _split_mapping_text_list(row.get("ordered_labels_low_to_high"))
        if isinstance(row.get("numeric_scores"), list):
            numeric_scores_raw = [str(value).strip() for value in (row.get("numeric_scores") or []) if str(value or "").strip()]
        else:
            numeric_scores_raw = _split_mapping_text_list(
                row.get("numeric_scores_low_to_high"),
                allow_comma_fallback=True,
            )
        notes = str(row.get("notes") or "").strip()

        if not target_id and not ordered_labels and not numeric_scores_raw and not response_scale_kind and not notes:
            continue
        if target_kind not in {"family", "column"} or not target_id:
            continue

        numeric_scores: List[float] = []
        numeric_scores_valid = True
        for score in numeric_scores_raw:
            try:
                numeric_scores.append(float(score))
            except (TypeError, ValueError):
                numeric_scores_valid = False
                break
        if not numeric_scores_valid:
            numeric_scores = []
        if numeric_scores and ordered_labels and len(numeric_scores) != len(ordered_labels):
            numeric_scores = []

        dedupe_key = (target_kind, target_id)
        if dedupe_key in seen_targets:
            continue
        seen_targets.add(dedupe_key)
        normalized.append(
            {
                "target_kind": target_kind,
                "target_id": target_id,
                "response_scale_kind": response_scale_kind,
                "ordered_labels": ordered_labels,
                "numeric_scores": numeric_scores,
                "notes": notes,
            }
        )
    return normalized


def _build_accepted_light_contract_handoff(contract_payload: Dict[str, Any]) -> Dict[str, Any]:
    primary_row = (contract_payload.get("primary_grain_rows") or [{}])[0] if contract_payload.get("primary_grain_rows") else {}
    override_rows = contract_payload.get("override_rows") or []
    scale_mapping_rows = contract_payload.get("scale_mapping_rows") or []
    override_notes = {
        str(row.get("field") or ""): str(row.get("user_input") or "")
        for row in override_rows
        if str(row.get("field") or "").strip()
    }
    reference_decisions = _reference_decisions_from_rows(_light_contract_reference_rows(contract_payload), key_mode="recommended")
    return {
        "run_id": str(contract_payload.get("run_id") or ""),
        "light_contract_status": "accepted",
        "primary_grain_decision": {
            "status": "accept",
            "keys": _keys_from_row(primary_row, "recommended"),
            "comments": str(primary_row.get("comments") or ""),
        },
        "reference_decisions": reference_decisions,
        "dimension_decisions": reference_decisions,
        "family_decisions": [
            {
                "family_id": str(row.get("family_id") or ""),
                "status": str(row.get("status") or "accept"),
                "table_name": str(row.get("recommended_table_name") or ""),
                "repeat_index_name": str(row.get("recommended_repeat_index_name") or ""),
                "parent_key": str(row.get("recommended_parent_key") or ""),
                "comments": str(row.get("comments") or ""),
            }
            for row in (contract_payload.get("repeat_family_rows") or [])
        ],
        "override_notes": override_notes,
        "semantic_context_input": _extract_semantic_context_input(override_rows),
        "scale_mapping_input": _normalize_light_contract_scale_mapping_input(scale_mapping_rows),
        "parse_validation": {
            "source": "stored_recommendation",
            "run_id_match": True,
        },
    }


def _build_parsed_light_contract_handoff(run_id: str, parsed_workbook: Dict[str, Any]) -> Dict[str, Any]:
    workbook_run_id = str((parsed_workbook.get("metadata") or {}).get("run_id") or "").strip()
    if workbook_run_id and workbook_run_id != run_id:
        raise HTTPException(status_code=422, detail=f"Workbook run_id '{workbook_run_id}' does not match submitted run_id '{run_id}'")

    primary_rows = parsed_workbook.get("primary_grain_rows") or []
    primary_row = primary_rows[0] if primary_rows else {}
    primary_status_raw = str(primary_row.get("status") or "").strip().lower() or "unsure"
    primary_status = "unsure" if primary_status_raw == "reject" else primary_status_raw
    if primary_status not in {"accept", "modify", "unsure"}:
        primary_status = "unsure"
    if primary_status == "modify":
        primary_keys = _keys_from_row(primary_row, "your")
        if not primary_keys:
            raise HTTPException(
                status_code=422,
                detail="Primary Grain is marked as modify, but no replacement key columns were provided in your_key_1..your_key_3",
            )
    else:
        primary_keys = _keys_from_row(primary_row, "recommended")

    reference_decisions = _reference_decisions_from_rows(
        _light_contract_reference_rows(parsed_workbook),
        key_mode="your_if_modify",
    )

    family_decisions = []
    for row in parsed_workbook.get("repeat_family_rows") or []:
        status = str(row.get("status") or "").strip().lower() or "unsure"
        table_name = str(row.get("your_table_name") or "").strip() if status == "modify" else str(row.get("recommended_table_name") or "")
        repeat_index_name = str(row.get("your_repeat_index_name") or "").strip() if status == "modify" else str(row.get("recommended_repeat_index_name") or "")
        parent_key = str(row.get("your_parent_key") or "").strip() if status == "modify" else str(row.get("recommended_parent_key") or "")
        family_decisions.append({
            "family_id": str(row.get("family_id") or ""),
            "status": status,
            "table_name": table_name,
            "repeat_index_name": repeat_index_name,
            "parent_key": parent_key,
            "comments": str(row.get("comments") or ""),
        })

    override_rows = parsed_workbook.get("override_rows") or []
    scale_mapping_rows = parsed_workbook.get("scale_mapping_rows") or []
    override_notes = {
        str(row.get("field") or ""): str(row.get("user_input") or "")
        for row in override_rows
        if str(row.get("field") or "").strip()
    }

    return {
        "run_id": run_id,
        "light_contract_status": "modified",
        "primary_grain_decision": {
            "status": primary_status,
            "keys": primary_keys,
            "comments": str(primary_row.get("comments") or ""),
        },
        "reference_decisions": reference_decisions,
        "dimension_decisions": reference_decisions,
        "family_decisions": family_decisions,
        "override_notes": override_notes,
        "semantic_context_input": _extract_semantic_context_input(override_rows),
        "scale_mapping_input": _normalize_light_contract_scale_mapping_input(scale_mapping_rows),
        "parse_validation": {
            "source": "uploaded_workbook",
            "workbook_run_id": workbook_run_id,
            "run_id_match": (workbook_run_id == run_id) if workbook_run_id else True,
            "primary_grain_reject_coerced_to_unsure": primary_status_raw == "reject",
        },
    }


def _normalize_light_contract_payload(run_id: str, grain_worker_output: Dict[str, Any]) -> Dict[str, Any]:
    column_order = _column_order_from_manifest_and_a2(run_id)
    family_map = _family_column_map(run_id)
    structural_gate_rows = _load_structural_gate_rows(run_id)
    column_guide_rows = [
        {
            "column_index": idx + 1,
            "column_name": col,
            "family_id": family_map.get(col, ""),
            "notes": "",
        }
        for idx, col in enumerate(column_order)
    ]
    return {
        "run_id": run_id,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_endpoint": "/light-contracts/xlsx",
        "readme_rows": [],
        "column_guide_rows": column_guide_rows,
        "grain_summary_rows": _build_grain_summary_rows(grain_worker_output),
        "primary_grain_rows": _build_primary_grain_rows(grain_worker_output),
        "reference_rows": _build_reference_rows(grain_worker_output),
        "repeat_family_rows": _build_repeat_family_rows(grain_worker_output, structural_gate_rows),
        "structural_gate_rows": structural_gate_rows,
        "scale_mapping_rows": _build_scale_mapping_rows(column_guide_rows),
        "override_rows": _build_override_rows(grain_worker_output),
    }


@app.post("/export/light-contract-xlsx", include_in_schema=False, deprecated=True)
def export_light_contract_xlsx(
    req: XlsxExportRequest,
    _: None = Depends(require_token),
):
    # (Decision) Use GCS + signed URL instead of returning bytes.
    # Why it matters: Dify HTTP nodes handle JSON cleanly; a signed URL gives a reliable “download” UX.

    try:
        xlsx_bytes = build_xlsx_bytes(req.rows_table_json, sheet_name="Light Contract")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"XLSX build failed: {e}")

    filename = _sanitize_export_filename(req.filename or "light_contract_template.xlsx", "light_contract_template.xlsx")
    result = _upload_xlsx_and_sign(xlsx_bytes=xlsx_bytes, filename=filename)
    return {
        "status": "success",
        "filename": result["filename"],
        "object": result["object_path"],
        "signed_url": result["signed_url"],
        "expires_minutes": result["expires_in_minutes"],
    }


@app.post("/light-contracts/xlsx")
def export_structured_light_contract_xlsx(
    req: LightContractXlsxRequest,
    _: None = Depends(require_token),
):
    if not isinstance(req.grain_worker_output, dict):
        raise HTTPException(status_code=422, detail="grain_worker_output must be a JSON object")

    try:
        normalized_payload = _normalize_light_contract_payload(
            run_id=req.run_id,
            grain_worker_output=req.grain_worker_output,
        )
        _upload_json_to_run_object(req.run_id, "light_contract_context.json", normalized_payload)
        xlsx_bytes = build_light_contract_xlsx_bytes(normalized_payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Light contract XLSX build failed: {exc}") from exc

    filename = _sanitize_export_filename(req.filename or f"{req.run_id}_light_contract.xlsx", "light_contract.xlsx")
    result = _upload_xlsx_and_sign(xlsx_bytes=xlsx_bytes, filename=filename)
    result["run_id"] = req.run_id
    return result


@app.get("/light-contracts/context")
def get_light_contract_context(
    run_id: str,
    _: None = Depends(require_token),
):
    context = _load_json_from_run_object(run_id, "light_contract_context.json")
    handoff = _build_accepted_light_contract_handoff(context)
    return {
        "run_id": run_id,
        "light_contract_context": context,
        "downstream_handoff": handoff,
    }


@app.post("/light-contracts/finalize-accepted")
def finalize_accepted_light_contract(
    req: LightContractFinalizeAcceptedRequest,
    _: None = Depends(require_token),
):
    context = _load_json_from_run_object(req.run_id, "light_contract_context.json")
    handoff = _build_accepted_light_contract_handoff(context)
    decisions, object_path = _persist_light_contract_decisions(
        run_id=req.run_id,
        handoff=handoff,
        source="accepted_context",
    )
    return {
        "run_id": req.run_id,
        "light_contract_status": decisions.get("light_contract_status"),
        "light_contract_decisions": decisions,
        "object_path": object_path,
    }


@app.post("/light-contracts/parse")
async def parse_light_contract_xlsx(
    run_id: str = Form(...),
    file: UploadFile = File(...),
    _: None = Depends(require_token),
):
    filename = str(file.filename or "").lower()
    if not (filename.endswith(".xlsx") or filename.endswith(".xlsm") or filename.endswith(".xltx") or filename.endswith(".xltm")):
        raise HTTPException(status_code=415, detail="Expected an Excel workbook upload")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded workbook is empty")

    try:
        parsed = parse_light_contract_xlsx_bytes(payload)
        downstream_handoff = _build_parsed_light_contract_handoff(run_id=run_id, parsed_workbook=parsed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse light contract workbook: {exc}") from exc

    return {
        "run_id": run_id,
        "parsed_workbook": parsed,
        "downstream_handoff": downstream_handoff,
    }


@app.post("/light-contracts/finalize-modified")
async def finalize_modified_light_contract(
    run_id: str = Form(...),
    file: UploadFile = File(...),
    _: None = Depends(require_token),
):
    filename = str(file.filename or "").lower()
    if not (filename.endswith(".xlsx") or filename.endswith(".xlsm") or filename.endswith(".xltx") or filename.endswith(".xltm")):
        raise HTTPException(status_code=415, detail="Expected an Excel workbook upload")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded workbook is empty")

    try:
        parsed = parse_light_contract_xlsx_bytes(payload)
        handoff = _build_parsed_light_contract_handoff(run_id=run_id, parsed_workbook=parsed)
        decisions, object_path = _persist_light_contract_decisions(
            run_id=run_id,
            handoff=handoff,
            source="uploaded_workbook",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to finalize modified light contract workbook: {exc}") from exc

    return {
        "run_id": run_id,
        "light_contract_status": decisions.get("light_contract_status"),
        "light_contract_decisions": decisions,
        "object_path": object_path,
        "parsed_workbook": parsed,
    }


@app.post("/codebooks/upload")
async def upload_codebook(
    run_id: str = Form(...),
    file: UploadFile = File(...),
    _: None = Depends(require_token),
):
    filename = str(file.filename or "").strip() or "codebook.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Expected a PDF codebook upload")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded codebook is empty")

    object_name = "codebook.pdf"
    object_path = _upload_bytes_to_run_object(
        run_id,
        object_name,
        payload,
        content_type="application/pdf",
    )
    pages = _extract_codebook_pages(payload)
    pages_payload = {
        "run_id": run_id,
        "filename": filename,
        "page_count": len(pages),
        "pages": pages,
    }
    pages_object_path = _upload_json_to_run_object(run_id, "codebook_pages.json", pages_payload)
    signed_url = _sign_run_object_download(
        run_id,
        object_name,
        filename=filename,
        content_type="application/pdf",
    )
    document_payload = {
        "run_id": run_id,
        "filename": filename,
        "object_name": object_name,
        "object_path": object_path,
        "page_count": len(pages),
        "pages_object_path": pages_object_path,
        "signed_url": signed_url,
        "uploaded_at": pd.Timestamp.utcnow().isoformat(),
    }
    _upload_json_to_run_object(run_id, "codebook_document.json", document_payload)

    return {
        "run_id": run_id,
        "filename": filename,
        "object_path": object_path,
        "pages_object_path": pages_object_path,
        "page_count": len(pages),
        "signed_url": signed_url,
    }


@app.get("/codebooks/context")
def get_codebook_context(
    run_id: str,
    _: None = Depends(require_token),
):
    document = _load_json_from_run_object(run_id, "codebook_document.json")
    pages = _load_optional_json_from_run_object(run_id, "codebook_pages.json") or {"pages": []}
    return {
        "run_id": run_id,
        "codebook_document": document,
        "codebook_pages": {
            "page_count": int(document.get("page_count") or 0),
            "pages_preview": _coerce_list_of_dicts(pages.get("pages"))[:3],
        },
    }


@app.get("/light-contracts/decisions")
def get_light_contract_decisions(
    run_id: str,
    _: None = Depends(require_token),
):
    try:
        decisions = _load_json_from_run_object(run_id, "light_contract_decisions.json")
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="light_contract_decisions.json not found for run_id. Finalize the light contract before continuing.",
            ) from exc
        raise
    return decisions

@app.post("/export/manifest-txt", include_in_schema=False, deprecated=True)
def export_manifest_txt(body: dict):
    run_id = body.get("run_id")
    manifest_text = body.get("manifest_text")

    if not run_id:
        raise HTTPException(status_code=400, detail="run_id required")
    if manifest_text is None:
        raise HTTPException(status_code=400, detail="manifest_text required")

    export_bucket = os.environ.get("EXPORT_BUCKET")
    if not export_bucket:
        raise HTTPException(status_code=500, detail="EXPORT_BUCKET env var not set")

    object_name = f"manifests/{run_id}_manifest.txt"

    signed_url = upload_and_sign_text(
        bucket_name=export_bucket,
        object_name=object_name,
        text_content=str(manifest_text),
        expiration_minutes=60,  # (inference) 60min is nicer for humans than 30min
    )

    return {
        "filename": f"{run_id}_manifest.txt",
        "signed_url": signed_url,
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

#testing

ARTIFACT_SPECS: Dict[str, Dict[str, str]] = {
    "A1": {"filename": "run_manifest.json", "content_type": "application/json"},
    "A2": {"filename": "column_dictionary.jsonl", "content_type": "application/jsonl"},
    "A3-T": {"filename": "transform_review_queue.json", "content_type": "application/json"},
    "A3-V": {"filename": "variable_type_review_queue.json", "content_type": "application/json"},
    "A4": {"filename": "missingness_catalog.json", "content_type": "application/json"},
    "A5": {"filename": "key_candidates_and_integrity.json", "content_type": "application/json"},
    "A6": {"filename": "grain_tests.json", "content_type": "application/json"},
    "A7": {"filename": "duplicates_report.json", "content_type": "application/json"},
    "A8": {"filename": "repeat_dimension_candidates.json", "content_type": "application/json"},
    "A9": {"filename": "role_scores.json", "content_type": "application/json"},
    "A10": {"filename": "relationships_and_derivations.json", "content_type": "application/json"},
    "A11": {"filename": "glimpses.json", "content_type": "application/json"},
    "A12": {"filename": "table_layout_candidates.json", "content_type": "application/json"},
    "A13": {"filename": "semantic_anchors.json", "content_type": "application/json"},
    "A14": {"filename": "quality_heatmap.json", "content_type": "application/json"},
    "A16": {"filename": "conditional_missingness.json", "content_type": "application/json"},
    "A17": {"filename": "baseline_column_resolution.json", "content_type": "application/json"},
    "B1": {"filename": "family_packets.jsonl", "content_type": "application/jsonl"},
}


def _require_api_key_from_request(request: Request) -> None:
    expected = os.getenv("PROFILER_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Missing PROFILER_API_KEY env var")

    auth = request.headers.get("authorization", "")
    api_key = request.headers.get("x-api-key")
    bearer_token = None
    if auth.lower().startswith("bearer "):
        bearer_token = auth[7:].strip()

    if api_key == expected or bearer_token == expected:
        return
    raise HTTPException(status_code=401, detail="Invalid token")


def _read_dataframe(raw_bytes: bytes, filename: Optional[str]) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw_bytes))
    return pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)


def _json_default(o: Any) -> Any:
    """
    Make JSON serialization robust to numpy/pandas scalars and common non-JSON types.

    (Decision) Convert scalar-like objects via .item() when available.
    Why it matters: pandas/numpy frequently produce np.int64/np.float64 that break json.dumps.

    (Decision) Serialize pandas timestamps/timedeltas to ISO strings.
    Why it matters: timestamps are common in profiling outputs; failing here crashes /full-bundle.
    """
    # numpy scalar (np.int64, np.float64, etc.)
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return o.item()
        except Exception:
            pass

    # pandas Timestamp / Timedelta
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.Timedelta):
        return str(o)

    # sets, tuples -> lists
    if isinstance(o, (set, tuple)):
        return list(o)

    # bytes -> utf-8 string (best-effort)
    if isinstance(o, (bytes, bytearray)):
        try:
            return o.decode("utf-8")
        except Exception:
            return str(o)

    # fallback
    return str(o)


import math

SKIP_LOGIC_TARGET_MISSINGNESS_PCT = float(os.getenv("SKIP_LOGIC_TARGET_MISSINGNESS_PCT", "20"))
SKIP_LOGIC_TRIGGER_MAX_CARD = int(os.getenv("SKIP_LOGIC_TRIGGER_MAX_CARD", "12"))
SKIP_LOGIC_MIN_SUPPORT_ROWS = int(os.getenv("SKIP_LOGIC_MIN_SUPPORT_ROWS", "5"))
SKIP_LOGIC_NAME_HINT_RE = re.compile(
    r"screen|screener|eligible|eligibility|consent|used|ever|skip|branch|gate|filter|routing|logic",
    re.IGNORECASE,
)

def _json_sanitize(x: Any) -> Any:
    """
    Convert values that are not JSON-compliant (NaN/Inf) into JSON-safe values.

    (Decision) Map NaN/Inf -> None.
    Why it matters: JSON does not support NaN/Inf; leaving them crashes /full-bundle under allow_nan=False.
    """
    # Handle numpy/pandas scalars early (so we can catch np.nan as a float)
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            x = x.item()
        except Exception:
            pass

    # None / bool / int / str are already JSON-safe
    if x is None or isinstance(x, (bool, int, str)):
        return x

    # float: replace NaN/Inf
    if isinstance(x, float):
        return x if math.isfinite(x) else None

    # pandas Timestamp / Timedelta
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, pd.Timedelta):
        return str(x)

    # dict
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}

    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return [_json_sanitize(v) for v in x]

    # bytes -> string best effort
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)

    # fallback: stringify
    return str(x)


def _json_bytes(payload: Any) -> bytes:
    safe = _json_sanitize(payload)
    return json.dumps(
        safe,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,   # now safe because we've removed NaN/Inf
    ).encode("utf-8")


def _jsonl_bytes(rows: List[Dict[str, Any]]) -> bytes:
    safe_rows = _json_sanitize(rows)
    return (
        "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in safe_rows)
        + "\n"
    ).encode("utf-8")

def _series_samples(s: pd.Series, dataset_sha256: str, col: str) -> Dict[str, List[Optional[str]]]:
    s_non_na = _non_null_series(s)
    s_non_na_str = _as_str_series(s_non_na)
    head_vals = [_stringify_value(v) for v in s_non_na_str.head(3).tolist()]
    tail_vals = [_stringify_value(v) for v in s_non_na_str.tail(2).tolist()]
    if len(s_non_na_str) > 0:
        rnd_n = min(3, len(s_non_na_str))
        rnd = s_non_na_str.sample(n=rnd_n, random_state=_stable_int_seed(dataset_sha256, col))
        random_vals = [_stringify_value(v) for v in rnd.tolist()]
    else:
        random_vals = []
    return {"head": head_vals, "tail": tail_vals, "random": random_vals}


def _tokenize_col_name(col: str) -> List[str]:
    # Split snake/kebab/space + camelCase + numeric boundaries
    base = re.sub(r"[\s\-]+", "_", str(col))
    base = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base)
    base = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", base)
    base = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", base)
    return [t for t in re.split(r"[_]+", base.lower()) if t]


def _fallback_family_id(candidate_id: Any, columns: List[str], raw_stem: Any = None) -> str:
    for value in (candidate_id, raw_stem):
        text = "_".join(_tokenize_col_name(str(value))) if value is not None else ""
        if text:
            return text

    clean_cols = [str(col).strip() for col in columns if str(col).strip()]
    if clean_cols:
        alpha_prefixes = []
        for col in clean_cols:
            match = re.match(r"^([A-Za-z]+)\d+$", col)
            if not match:
                alpha_prefixes = []
                break
            alpha_prefixes.append(match.group(1).lower())
        if alpha_prefixes and len(set(alpha_prefixes)) == 1:
            return alpha_prefixes[0]

        first_col = "_".join(_tokenize_col_name(clean_cols[0]))
        if first_col:
            return f"family_{first_col}"

    return "family_unknown"


def _build_families(columns: List[str]) -> Dict[str, Any]:
    family_map: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    label_index_tokens = {
        "baseline", "base", "screen", "pre", "post", "followup", "fu", "endline", "midline", "t0", "t1", "t2", "t3"
    }
    repeat_keywords = {"row", "col", "item", "wave", "visit", "session", "trial", "tp", "timepoint", "question", "q"}

    def _norm_stem(stem: str) -> str:
        toks = _tokenize_col_name(stem)
        return "_".join(toks)

    def _add_member(family_id: str, col: str, index_token: str, pattern: str, *, raw_stem: Optional[str] = None, keyword: Optional[str] = None, camelcase_split_used: bool = False) -> None:
        fam = family_map.setdefault(family_id, {
            "columns": [],
            "index_tokens": [],
            "patterns": set(),
            "index_by_column": {},
            "stem_evidence": {
                "raw_stem": raw_stem or family_id,
                "normalized_stem": _norm_stem(raw_stem or family_id),
                "camelcase_split_used": bool(camelcase_split_used),
            },
            "keyword": keyword,
        })
        fam["columns"].append(col)
        fam["index_tokens"].append(index_token)
        fam["patterns"].add(pattern)
        fam["index_by_column"][col] = [str(index_token)]
        if keyword and not fam.get("keyword"):
            fam["keyword"] = keyword

    for col in columns:
        clean = str(col)
        toks = _tokenize_col_name(clean)
        camel_split = clean != re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", clean)

        # Stage A: suffix keyword + numeric, e.g. Q6Main_cell_groupRow12
        m_kw_num = re.match(r"^(.*?)(row|col|item|wave|visit|session|trial|tp|timepoint|question|q)[_\-]?(\d+)$", clean, re.IGNORECASE)
        if m_kw_num:
            raw_stem = m_kw_num.group(1)
            keyword = m_kw_num.group(2).lower()
            idx = m_kw_num.group(3)
            fid = _norm_stem(raw_stem)
            _add_member(fid, clean, idx, "suffix_keyword_numeric", raw_stem=raw_stem, keyword=keyword, camelcase_split_used=camel_split)

        # Generic suffix numeric with separator
        m_suffix_num = re.match(r"^(.*?)[_\-](\d+)$", clean, re.IGNORECASE)
        if m_suffix_num:
            fid = _norm_stem(m_suffix_num.group(1))
            _add_member(fid, clean, m_suffix_num.group(2), "suffix_numeric_index", raw_stem=m_suffix_num.group(1), camelcase_split_used=camel_split)

        # Generic suffix numeric without separator (DASS1)
        m_suffix_num_no_sep = re.match(r"^(.*?)(\d+)$", clean, re.IGNORECASE)
        if m_suffix_num_no_sep and not m_kw_num:
            stem = m_suffix_num_no_sep.group(1)
            idx = m_suffix_num_no_sep.group(2)
            # require meaningful stem length to reduce false positives
            if len(_tokenize_col_name(stem)) >= 1 and len(stem) >= 2:
                fid = _norm_stem(stem)
                _add_member(fid, clean, idx, "suffix_numeric_nosep", raw_stem=stem, camelcase_split_used=camel_split)

        # Prefix label indices from tokenized names
        if len(toks) >= 2:
            if toks[-1] in label_index_tokens:
                _add_member("_".join(toks[:-1]), clean, toks[-1], "suffix_label_index", raw_stem="_".join(toks[:-1]), camelcase_split_used=camel_split)
            if toks[0] in label_index_tokens:
                _add_member("_".join(toks[1:]), clean, toks[0], "prefix_label_index", raw_stem="_".join(toks[1:]), camelcase_split_used=camel_split)

    families: List[Dict[str, Any]] = []
    for stem, info in sorted(family_map.items()):
        cols = sorted(set(info["columns"]))
        idx_tokens = [str(x) for x in info["index_tokens"]]
        idx_num = sorted({int(t) for t in idx_tokens if re.fullmatch(r"\d+", t)})
        dense_ok = False
        missing_idx: List[int] = []
        if len(idx_num) >= 3:
            low, high = min(idx_num), max(idx_num)
            expected = set(range(low, high + 1))
            missing_idx = sorted(expected - set(idx_num))
            dense_ok = len(missing_idx) <= max(1, int(0.2 * len(expected)))

        keyword = info.get("keyword")
        has_keyword_signal = bool(keyword in repeat_keywords)
        index_type = "ordinal" if all(re.fullmatch(r"\d+", t) for t in idx_tokens) else "label"

        reject_reason = None
        if len(cols) < 3:
            reject_reason = "insufficient_family_size"
        elif index_type == "ordinal" and not dense_ok and len(idx_num) >= 3:
            reject_reason = "non_dense_indices"
        elif (not has_keyword_signal) and ("suffix_keyword_numeric" not in info["patterns"]) and len(cols) < 4:
            reject_reason = "no_repeat_keyword_and_insufficient_family_evidence"

        if reject_reason:
            rejected.append({
                "candidate_stem": stem,
                "columns_count": len(cols),
                "reason": reject_reason,
                "patterns": sorted(info["patterns"]),
            })
            continue

        conf = 0.45
        conf += min(0.25, 0.03 * len(cols))
        conf += 0.2 if has_keyword_signal else 0.0
        conf += 0.1 if dense_ok else 0.0
        conf = min(0.99, conf)

        recommended_repeat = keyword if keyword in repeat_keywords else ("index" if index_type == "ordinal" else "label")
        family_id = _fallback_family_id(stem, cols, (info.get("stem_evidence") or {}).get("raw_stem"))
        families.append({
            "family_id": family_id,
            "detection_confidence": round(conf, 6),
            "columns_count": len(cols),
            "columns": cols,
            "columns_preview": cols[:8],
            "patterns": sorted(info["patterns"]),
            "index_pattern": {
                "kind": "suffix_keyword_numeric" if "suffix_keyword_numeric" in info["patterns"] else sorted(info["patterns"])[0],
                "keyword": keyword,
                "index_type_candidate": index_type,
                "extracted_index_set_preview": [str(x) for x in sorted(set(idx_tokens))[:10]],
                "is_dense_sequence": bool(dense_ok) if index_type == "ordinal" else None,
                "missing_indices": [str(x) for x in missing_idx[:20]],
            },
            "extracted_index_set": sorted(set(idx_tokens)),
            "index_by_column": info.get("index_by_column", {}),
            "stem_evidence": info.get("stem_evidence", {}),
            "recommended_repeat_dimension_name": recommended_repeat,
            "downstream_hints": {
                "likely_member_role": "measure",
                "test_composite_keys_with": ["id_candidate", recommended_repeat],
            },
            "flags": {
                "variable_specific_indices": False,
                "suspected_timepoint": recommended_repeat in {"wave", "visit", "session", "timepoint", "tp"},
                "suspected_item": recommended_repeat in {"item", "question", "q", "row", "col"},
                "suspected_event": recommended_repeat in {"session", "visit", "trial"},
            },
        })

    families_sorted = sorted(
        families,
        key=lambda f: (-float(f.get("detection_confidence", 0.0)), -int(f.get("columns_count", 0)), str(f.get("family_id", ""))),
    )
    covered_cols = sorted({c for fam in families_sorted for c in fam.get("columns", [])})
    families_index = [
        {
            "family_id": fam.get("family_id"),
            "columns_count": int(fam.get("columns_count", 0) or 0),
            "recommended_repeat_dimension_name": fam.get("recommended_repeat_dimension_name"),
            "repeat_dim": fam.get("recommended_repeat_dimension_name"),
            "patterns": fam.get("patterns", []),
            "detection_confidence": float(fam.get("detection_confidence", 0.0) or 0.0),
            "confidence": float(fam.get("detection_confidence", 0.0) or 0.0),
        }
        for fam in families_sorted
    ]

    near_misses = [
        r for r in sorted(rejected, key=lambda x: (-int(x.get("columns_count", 0) or 0), str(x.get("candidate_stem", ""))))
        if int(r.get("columns_count", 0) or 0) in {2, 3}
    ][:20]

    return {
        "artifact": "A8",
        "purpose": "repeat_family_detection",
        "families_index": families_index,
        "coverage": {
            "covered_columns_count": len(covered_cols),
            "covered_columns_pct": round((len(covered_cols) / max(1, len(columns))) * 100.0, 6),
        },
        "near_misses": near_misses,
        "families": families_sorted,
        "rejected_candidates": sorted(rejected, key=lambda r: (-int(r.get("columns_count", 0)), str(r.get("candidate_stem", ""))))[:100],
        "debug_trace": {
            "n_columns_scanned": len(columns),
            "n_candidate_stems": len(family_map),
            "n_accepted_families": len(families),
            "camelcase_split_enabled": True,
        },
    }


def _upload_artifact(bucket: storage.Bucket, object_path: str, payload: bytes, content_type: str) -> Dict[str, Any]:
    blob = bucket.blob(object_path)
    blob.upload_from_string(payload, content_type=content_type)
    return {
        "object_path": object_path,
        "content_type": content_type,
        "sha256": sha256_hex(payload),
        "size_bytes": len(payload),
    }


def _build_artifact_url(base_url: str, artifact_id: str, run_id: str, mode: str) -> str:
    return f"{base_url}/artifacts/{artifact_id}/{mode}?run_id={run_id}"


def _build_artifact_view_get_url(
    base_url: str,
    artifact_id: str,
    run_id: str,
    mode: Optional[str] = None,
    keep: Optional[Iterable[str]] = None,
    drop: Optional[Iterable[str]] = None,
    limits: Optional[Dict[str, int]] = None,
) -> str:
    query: Dict[str, Any] = {"run_id": run_id}
    if mode:
        query["mode"] = mode
    if keep:
        query["keep"] = ",".join(str(k) for k in keep)
    if drop:
        query["drop"] = ",".join(str(k) for k in drop)
    if limits:
        query["limits"] = ",".join(f"{k}:{v}" for k, v in limits.items())
    return f"{base_url}/artifacts/{artifact_id}?{urlencode(query)}"


def _is_skip_logic_trigger_candidate(col: str, s: pd.Series, unique_count: int) -> bool:
    if unique_count < 2:
        return False
    if unique_count <= SKIP_LOGIC_TRIGGER_MAX_CARD:
        return True
    return bool(SKIP_LOGIC_NAME_HINT_RE.search(col or ""))


def _build_conditional_missingness_artifact(
    df: pd.DataFrame,
    cols_profile: Dict[str, Any],
    column_signal_map: Dict[str, Dict[str, Any]],
    artifact_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    n_rows = int(len(df))
    min_support_rows = max(SKIP_LOGIC_MIN_SUPPORT_ROWS, int(round(n_rows * 0.01))) if n_rows else SKIP_LOGIC_MIN_SUPPORT_ROWS
    repeat_candidates = _build_families(list(df.columns))
    family_by_col: Dict[str, str] = {}
    for fam in repeat_candidates.get("families", []) or []:
        fam_id = str(fam.get("family_id") or "")
        for col in fam.get("columns", []) or []:
            family_by_col[str(col)] = fam_id

    target_columns: List[str] = []
    trigger_columns: List[str] = []
    for col in df.columns:
        sig = column_signal_map.get(col, {})
        missing_pct = float(sig.get("missing_pct", cols_profile.get(col, {}).get("missing_pct", 0.0)) or 0.0)
        unique_count = int(sig.get("unique_count", cols_profile.get(col, {}).get("unique_count", int(df[col].nunique(dropna=True))) or 0))

        if missing_pct > SKIP_LOGIC_TARGET_MISSINGNESS_PCT:
            target_columns.append(col)
        if col not in family_by_col and _is_skip_logic_trigger_candidate(col, df[col], unique_count):
            trigger_columns.append(col)

    grouped_rules: Dict[Tuple[str, str], Dict[str, Any]] = {}
    candidate_rule_count = 0

    for target_col in target_columns:
        target_missing = df[target_col].isna()
        if not bool(target_missing.any()):
            continue

        for trigger_col in trigger_columns:
            if trigger_col == target_col:
                continue

            trigger_series = df[trigger_col]
            trigger_non_null = trigger_series.dropna()
            if trigger_non_null.empty:
                continue

            trigger_missing_rate = float(trigger_series.isna().mean()) if len(trigger_series) else 0.0
            if trigger_missing_rate >= 0.8:
                continue

            value_counts = trigger_non_null.astype("string").value_counts(dropna=True)
            for trigger_value, support in value_counts.items():
                support_n = int(support)
                if support_n < min_support_rows:
                    continue

                candidate_rule_count += 1
                match_mask = trigger_series.astype("string") == str(trigger_value)
                if not bool(match_mask.any()):
                    continue

                if bool(target_missing.loc[match_mask].all()):
                    rule_key = (trigger_col, str(trigger_value))
                    if rule_key not in grouped_rules:
                        grouped_rules[rule_key] = {
                            "trigger_column": trigger_col,
                            "trigger_value": str(trigger_value),
                            "affected_columns": [],
                            "rule_strength": 1.0,
                            "interpretation": "Valid structural missingness (Skip Logic).",
                        }
                    grouped_rules[rule_key]["affected_columns"].append(target_col)

    detected_skip_logic: List[Dict[str, Any]] = []
    for rule in grouped_rules.values():
        affected = sorted(set(rule["affected_columns"]))
        if not affected:
            continue
        affected_family_ids = sorted({family_by_col[c] for c in affected if c in family_by_col})
        explained_missing_pcts: List[float] = []
        directionality = "bidirectional"
        for col in affected:
            target_missing = df[col].isna()
            total_missing = int(target_missing.sum())
            match_mask = df[rule["trigger_column"]].astype("string") == str(rule["trigger_value"])
            explained_missing = int((target_missing & match_mask).sum())
            explained_pct = float((explained_missing / total_missing) * 100.0) if total_missing else 0.0
            explained_missing_pcts.append(explained_pct)
            if explained_missing != total_missing:
                directionality = "sufficient_only"
        detected_skip_logic.append({
            "trigger_column": rule["trigger_column"],
            "trigger_value": rule["trigger_value"],
            "affected_column_count": int(len(affected)),
            "affected_family_ids": affected_family_ids,
            "affected_family_count": int(len(affected_family_ids)),
            "sample_affected_columns": affected[:5],
            "rule_strength": float(rule["rule_strength"]),
            "directionality": directionality,
            "missing_explained_pct": round(min(explained_missing_pcts) if explained_missing_pcts else 0.0, 6),
            "interpretation": rule["interpretation"],
        })

    detected_skip_logic.sort(
        key=lambda item: (-int(item["affected_column_count"]), item["trigger_column"], item["trigger_value"])
    )

    master_switch_candidates: List[Dict[str, Any]] = []
    trigger_aggregate: Dict[str, Dict[str, Any]] = {}
    for rule in grouped_rules.values():
        trigger_col = str(rule["trigger_column"])
        agg = trigger_aggregate.setdefault(trigger_col, {
            "values": [],
            "columns": set(),
            "family_ids": set(),
        })
        affected = sorted(set(rule["affected_columns"]))
        agg["values"].append({
            "trigger_value": str(rule["trigger_value"]),
            "affected_column_count": len(affected),
            "sample_affected_columns": affected[:5],
        })
        agg["columns"].update(affected)
        agg["family_ids"].update({family_by_col[c] for c in affected if c in family_by_col})

    for trigger_col, agg in trigger_aggregate.items():
        explained_columns = sorted(agg["columns"])
        explained_family_ids = sorted(agg["family_ids"])
        score = 0.35 + min(0.5, 0.03 * len(explained_columns)) + (0.1 if SKIP_LOGIC_NAME_HINT_RE.search(trigger_col) else 0.0)
        master_switch_candidates.append({
            "trigger_column": trigger_col,
            "explained_column_count": int(len(explained_columns)),
            "explained_family_count": int(len(explained_family_ids)),
            "affected_family_ids": explained_family_ids,
            "top_trigger_values": [
                item["trigger_value"]
                for item in sorted(agg["values"], key=lambda x: (-int(x["affected_column_count"]), x["trigger_value"]))[:3]
            ],
            "sample_affected_columns": explained_columns[:5],
            "confidence": round(float(min(0.99, score)), 6),
            "interpretation": "Likely master screening/gating field for structural missingness.",
        })

    master_switch_candidates.sort(
        key=lambda item: (-int(item["explained_column_count"]), -int(item["explained_family_count"]), item["trigger_column"])
    )

    return {
        "artifact": "A16",
        "purpose": "conditional_missingness_skip_logic",
        "inputs": artifact_inputs["A16"],
        "detected_skip_logic": detected_skip_logic,
        "master_switch_candidates": master_switch_candidates,
        "audit_assumptions": {
            "family_member_triggers_excluded_from_master_switch_candidates": True,
        },
        "audit_trace": {
            "target_missingness_threshold_pct": float(SKIP_LOGIC_TARGET_MISSINGNESS_PCT),
            "trigger_max_cardinality": int(SKIP_LOGIC_TRIGGER_MAX_CARD),
            "min_support_rows": int(min_support_rows),
            "rows_evaluated": n_rows,
            "target_columns_scanned": sorted(target_columns),
            "trigger_columns_scanned": sorted(trigger_columns),
            "candidate_rule_count_considered": int(candidate_rule_count),
            "family_member_triggers_excluded": True,
        },
    }


def _build_artifact_view_post_url(base_url: str, artifact_id: str) -> str:
    return f"{base_url}/artifacts/{artifact_id}/view"


def _enrich_artifact_entry_with_urls(base_url: str, run_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    aid = item["artifact_id"]
    return {
        **item,
        "download_url": _build_artifact_url(base_url, aid, run_id, "download"),
        "meta_url": _build_artifact_url(base_url, aid, run_id, "meta"),
        "view_get_url": _build_artifact_view_get_url(base_url, aid, run_id, mode="llm_baseline"),
        "view_post_url": _build_artifact_view_post_url(base_url, aid),
    }


def _candidate_evidence_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    ev = candidate.get("evidence") or {}
    out: Dict[str, Any] = {}
    keep_scalar_keys = [
        "suffix_unit_pct",
        "strict_numeric_pct",
        "date_parse_pct",
        "multi_token_pct",
        "numeric_range_pct",
        "year_range_pct",
        "level_pct",
        "strict_pct",
        "currency_pct",
        "percent_pct",
        "thousands_sep_pct",
        "inferred_type",
        "uniq_score",
        "structural_signal",
        "parse_signal",
    ]
    for k in keep_scalar_keys:
        if k in ev and isinstance(ev.get(k), (int, float, str, bool)):
            out[k] = ev[k]
    if "unit_plan" in ev:
        out["unit_plan_detected"] = bool(ev.get("unit_plan"))
    if "delimiter" in ev and isinstance(ev.get("delimiter"), str):
        out["delimiter"] = ev.get("delimiter")
    return out


def _is_one_hot_like_column(col: str, p: Dict[str, Any], s: pd.Series) -> bool:
    candidates = p.get("candidates") or []
    top = candidates[0] if candidates else {}
    top_type = str(top.get("type") or "")
    unique_count = int(p.get("unique_count", 0) or 0)
    if unique_count == 0:
        return False
    pattern_hint = bool(re.search(r"(_|\b)(yes|no|true|false|male|female|other|0|1)$", col, re.IGNORECASE))
    if top_type == "categorical" and unique_count <= 2 and pattern_hint:
        return True
    if top_type in {"categorical", "mixed", "text"} and unique_count <= 2:
        non_na = s.dropna().astype("string").str.strip().str.lower()
        if len(non_na) == 0:
            return False
        values = {v for v in non_na.unique().tolist() if v is not None}
        binary_lexicons = [
            {"yes", "no"},
            {"true", "false"},
            {"y", "n"},
            {"t", "f"},
            {"0", "1"},
            {"male", "female"},
            {"present", "absent"},
        ]
        return any(values.issubset(lex) for lex in binary_lexicons)
    return top_type in {"integer", "float", "numeric"} and unique_count <= 2 and pattern_hint


def _candidate_family(candidate_type: str) -> str:
    if candidate_type in {"numeric", "numeric_with_unit", "percent", "numeric_range"}:
        return "numeric"
    if candidate_type in {"date", "datetime"}:
        return "temporal"
    if candidate_type in {"categorical", "categorical_multi"}:
        return "categorical"
    if candidate_type in {"range_like"}:
        return "range"
    return "textual"


def _parse_repeat_structure_name(col: str) -> Optional[Dict[str, Any]]:
    patterns = [
        (r"^(.*?)(row|r)(\d+)$", "row"),
        (r"^(.*?)(col|c)(\d+)$", "col"),
        (r"^(.*?)(item|q|question)(\d+)$", "item"),
        (r"^(.*?)(trial|block|tp|wave|visit|session)(\d+)$", "index"),
    ]
    clean = re.sub(r"[\s\-]+", "_", str(col)).strip("_")
    for pat, axis_default in patterns:
        m = re.match(pat, clean, re.IGNORECASE)
        if m:
            base, keyword, idx = m.group(1), m.group(2), m.group(3)
            return {
                "base_name": base,
                "index_keyword": keyword.lower(),
                "index_value": int(idx),
                "repeat_axis_type": axis_default if axis_default != "index" else keyword.lower(),
                "family_candidate": re.sub(r"[_\-]+$", "", base),
            }
    return None


def _build_transform_review_queue(df: pd.DataFrame, cols_profile: Dict[str, Any], max_cat: int) -> Dict[str, Any]:
    policy = {
        "review_rules_version": "v3_transform",
        "ordered_by": "transform_risk",
    }
    review_rows: List[Dict[str, Any]] = []
    for col in df.columns:
        p = cols_profile.get(col, {})
        cands = p.get("candidates") or []
        top = cands[0] if cands else {}
        top_type = str(top.get("type") or "")
        top_parse = top.get("parse")
        top_op = top.get("op")
        delimiter_stats = (p.get("patterns", {}).get("multi_value") or {})

        transform_required = bool(top_op)
        transform_reason = "op_requires_transform" if top_op else "none"
        if not transform_required and top_type in ("numeric_with_unit", "categorical_multi", "range_like", "date"):
            transform_required = True
            transform_reason = f"type_requires_transform:{top_type}"
        if not transform_required and top_parse in ("currency+suffix_possible", "percent_possible", "thousands_sep_possible", "strict_with_suffix_minority"):
            transform_required = True
            transform_reason = f"parse_requires_transform:{top_parse}"

        if transform_required:
            transform_candidates = [
                {
                    "rank": i + 1,
                    "type": c.get("type"),
                    "confidence": round(float(c.get("confidence", 0.0) or 0.0), 6),
                    "parse": c.get("parse"),
                    "op": c.get("op"),
                }
                for i, c in enumerate(cands[:4])
                if c.get("op")
                or c.get("type") in ("numeric_with_unit", "categorical_multi", "range_like", "date")
                or c.get("parse") in ("currency+suffix_possible", "percent_possible", "thousands_sep_possible", "strict_with_suffix_minority")
            ]

            risk_score = 0.45
            if top_op:
                risk_score += 0.25
            if top_type in {"categorical_multi", "numeric_with_unit", "range_like"}:
                risk_score += 0.20
            if top_type == "date":
                risk_score += 0.15
            if delimiter_stats.get("multi_token_pct", 0.0) >= 40.0:
                risk_score += 0.05
            risk_score = min(0.99, risk_score)

            review_rows.append({
                "column": col,
                "top_candidate": {
                    "type": top.get("type"),
                    "confidence": round(float(top.get("confidence", 0.0) or 0.0), 6),
                    "parse": top_parse,
                    "op": top_op,
                },
                "requires_transform": True,
                "transform_reason": transform_reason,
                "transform_candidates": transform_candidates,
                "risk_level": "high" if risk_score >= 0.75 else ("medium" if risk_score >= 0.55 else "low"),
                "risk_score": round(risk_score, 6),
                "evidence_snippets": {
                    "parse_failure_examples": (p.get("patterns", {}).get("date_like", {}).get("parse_failure_examples", []) or [])[:3],
                    "multi_value_examples": (p.get("patterns", {}).get("multi_value_examples", {}) or {}).get("examples", [])[:3],
                },
            })
    review_rows.sort(key=lambda r: (-r["risk_score"], r["column"]))
    return {"policy": policy, "items": review_rows, "count": len(review_rows)}


def _build_variable_type_review_queue(df: pd.DataFrame, cols_profile: Dict[str, Any], max_cat: int) -> Dict[str, Any]:
    policy = {
        "review_rules_version": "v3_variable_type",
        "low_confidence_threshold": 0.9,
        "close_gap_threshold": 0.15,
        "minimum_second_confidence": 0.5,
    }
    review_rows: List[Dict[str, Any]] = []
    for col in df.columns:
        p = cols_profile.get(col, {})
        cands = p.get("candidates") or []
        if len(cands) < 2:
            continue
        top = cands[0]
        second = cands[1]
        top_conf = float(top.get("confidence", 0.0) or 0.0)
        second_conf = float(second.get("confidence", 0.0) or 0.0)
        conf_gap = float(top_conf - second_conf)
        top_type = str(top.get("type") or "")
        second_type = str(second.get("type") or "")
        fam1 = _candidate_family(top_type)
        fam2 = _candidate_family(second_type)

        clear_categorical_enum = (
            top_type == "categorical"
            and int(p.get("unique_count", 0) or 0) > 0
            and int(p.get("unique_count", 0) or 0) <= max_cat
            and not top.get("op")
            and top.get("parse") == "levels"
            and top_conf >= policy["low_confidence_threshold"]
        )
        if clear_categorical_enum:
            continue
        if fam1 == fam2:
            continue
        if top_conf >= policy["low_confidence_threshold"] and conf_gap >= policy["close_gap_threshold"]:
            continue
        if second_conf < policy["minimum_second_confidence"]:
            continue

        ambiguity_reason = "low_top_confidence" if top_conf < policy["low_confidence_threshold"] else "small_gap_between_semantic_alternatives"
        review_rows.append({
            "column": col,
            "top_candidate": {
                "type": top.get("type"),
                "confidence": round(top_conf, 6),
                "parse": top.get("parse"),
                "op": top.get("op"),
            },
            "second_candidate": {
                "type": second.get("type"),
                "confidence": round(second_conf, 6),
                "parse": second.get("parse"),
                "op": second.get("op"),
            },
            "confidence_gap": round(conf_gap, 6),
            "ambiguity_reason": ambiguity_reason,
            "evidence": {
                "candidate_families": {"top": fam1, "second": fam2},
                "unique_count": int(p.get("unique_count", 0) or 0),
                "missing_pct": round(float(p.get("missing_pct", 0.0) or 0.0), 6),
                "top_evidence_summary": _candidate_evidence_summary(top),
                "second_evidence_summary": _candidate_evidence_summary(second),
            },
        })
    review_rows.sort(key=lambda r: (r["top_candidate"]["confidence"], r["confidence_gap"], r["column"]))
    return {"policy": policy, "items": review_rows, "count": len(review_rows)}


def _build_table_layout_candidates(
    key_integrity: Dict[str, Any],
    grain_tests: List[Dict[str, Any]],
    repeat_candidates: Dict[str, Any],
    role_scores: Dict[str, Any],
    relationships: Dict[str, Any],
    all_columns: List[str],
) -> Dict[str, Any]:
    def _role_ranking_for_col(col: str) -> List[Dict[str, Any]]:
        row = next((r for r in role_scores.get("columns", []) if r.get("column") == col), None)
        if not row:
            return []
        rc = row.get("role_candidates")
        if isinstance(rc, list) and rc:
            return sorted(rc, key=lambda x: (-float(x.get("score", 0.0) or 0.0), str(x.get("role") or "")))
        rs = row.get("role_scores", {})
        return sorted(
            [{"role": k, "score": float(v or 0.0), "evidence": {}} for k, v in rs.items()],
            key=lambda x: (-x["score"], x["role"]),
        )

    def _rationale_unique_grain(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if int(g.get("dup_row_count", 10**9) or 10**9) == 0 and float(g.get("collision_severity_score", 1.0) or 1.0) <= 1e-9:
            key = ",".join(g.get("keys_tested", []))
            return {
                "code": "grain_unique_single_key",
                "text": f"{key} uniquely identifies rows ({int(g.get('dup_row_count', 0))} duplicate rows under tested key).",
                "evidence": {"A6": {"dup_row_count": int(g.get("dup_row_count", 0)), "collision_severity_score": float(g.get("collision_severity_score", 0.0) or 0.0)}},
            }
        return {
            "code": "grain_candidate_ambiguous",
            "text": f"{','.join(g.get('keys_tested', [])) or 'candidate key'} is a candidate row grain, but duplicate/collision evidence remains ambiguous.",
            "evidence": {"A6": {"dup_row_count": int(g.get("dup_row_count", 0) or 0), "collision_severity_score": float(g.get("collision_severity_score", 1.0) or 1.0)}},
        }

    best_grains = sorted(grain_tests, key=lambda g: (g.get("collision_severity_score", 1.0), g.get("dup_row_count", 10**9)))[:3]
    families = repeat_candidates.get("families", [])
    key_seeds = key_integrity.get("single_column_key_seed_candidates", [])
    stress_by_col = {x.get("candidate_column"): x for x in key_integrity.get("candidate_key_stress_tests", [])}
    relationship_support = relationships.get("dependency_candidates", [])

    layouts = []
    for i, g in enumerate(best_grains, start=1):
        pk = list(g.get("keys_tested", []) or [])
        role_assignments: Dict[str, Dict[str, Any]] = {}
        for c in all_columns:
            ranked = _role_ranking_for_col(c)
            primary = ranked[0] if ranked else {"role": "unknown", "score": 0.0}
            alts = [r for r in ranked[1:3] if float(r.get("score", 0.0) or 0.0) >= 0.35]
            role_assignments[c] = {
                "primary_role": primary.get("role", "unknown"),
                "primary_score": round(float(primary.get("score", 0.0) or 0.0), 6),
                "alternate_roles": [{"role": a.get("role"), "score": round(float(a.get("score", 0.0) or 0.0), 6)} for a in alts],
            }

        repeat_dims = [
            c for c, ra in role_assignments.items()
            if ra["primary_role"] in {"time_index", "repeat_index"}
            and c not in pk and ra["primary_score"] >= 0.5
        ][:3]
        entity_attrs = [c for c, ra in role_assignments.items() if ra["primary_role"] in {"invariant_attr", "coded_categorical"} and c not in pk][:30]
        event_measures = [c for c, ra in role_assignments.items() if ra["primary_role"] in {"measure", "measure_numeric", "measure_item"} and c not in pk and c not in repeat_dims][:40]

        tables = []
        entity_table_name = "entity_" + (pk[0].lower() if pk else f"candidate_{i}")
        tables.append({
            "table_id": "T1",
            "table_name": entity_table_name,
            "kind": "entity",
            "grain": pk,
            "columns_preview": {
                "keys": pk,
                "attributes": entity_attrs[:12],
                "sample_measures_if_any": [],
            },
            "column_counts": {
                "keys": len(pk),
                "attributes": len(entity_attrs),
                "measures": 0,
            },
            "foreign_keys": [],
            "primary_role_assignments_preview": {c: role_assignments[c] for c in (pk + entity_attrs[:8])},
        })

        if repeat_dims or event_measures:
            event_keys = list(dict.fromkeys(pk + repeat_dims))
            tables.append({
                "table_id": "T2",
                "table_name": "event_" + (pk[0].lower() if pk else f"candidate_{i}"),
                "kind": "event_repeat",
                "grain": event_keys,
                "repeat_dimension_candidates": repeat_dims,
                "columns_preview": {
                    "keys": event_keys,
                    "attributes": [],
                    "measures": event_measures[:15],
                },
                "column_counts": {
                    "keys": len(event_keys),
                    "attributes": 0,
                    "measures": len(event_measures),
                },
                "foreign_keys": [{"from_cols": pk, "to_table": entity_table_name, "to_cols": pk}] if pk else [],
                "primary_role_assignments_preview": {c: role_assignments[c] for c in (event_keys + event_measures[:8])},
            })

        fam_tables = []
        for idx, fam in enumerate(families[:3], start=1):
            fam_cols = [c for c in fam.get("columns", []) if c in all_columns]
            fam_measures = [c for c in fam_cols if role_assignments.get(c, {}).get("primary_role") in {"measure", "measure_numeric", "measure_item", "coded_categorical"}]
            fam_keys = list(dict.fromkeys(pk + repeat_dims[:1]))
            fam_tables.append({
                "table_id": f"TF{idx}",
                "table_name": f"family_{str(fam.get('family_id') or idx).lower()}",
                "kind": "family_repeat",
                "grain": fam_keys,
                "repeat_dimension_candidates": repeat_dims[:1],
                "columns_preview": {
                    "keys": fam_keys,
                    "attributes": [],
                    "measures": fam_measures[:15],
                },
                "column_counts": {
                    "keys": len(fam_keys),
                    "attributes": 0,
                    "measures": len(fam_measures),
                },
                "foreign_keys": [{"from_cols": pk, "to_table": entity_table_name, "to_cols": pk}] if pk else [],
                "family_reference": fam,
            })
        tables.extend(fam_tables)

        covered_by_layout = set()
        for t in tables:
            covered_by_layout.update(t.get("columns_preview", {}).get("keys", []))
            covered_by_layout.update(t.get("columns_preview", {}).get("attributes", []))
            covered_by_layout.update(t.get("columns_preview", {}).get("measures", []))
        role_scored = {c for c, ra in role_assignments.items() if ra.get("primary_score", 0.0) >= 0.5}
        covered_by_role_only = sorted([c for c in role_scored if c not in covered_by_layout])
        ambiguous = sorted([c for c, ra in role_assignments.items() if len(ra.get("alternate_roles", [])) > 0 and c not in covered_by_role_only])
        unmapped = sorted([c for c in all_columns if c not in covered_by_layout and c not in role_scored])

        dep_support = [d for d in relationship_support if (d.get("determinant") in pk or d.get("dependent") in pk)]
        stress = stress_by_col.get(pk[0], {}) if len(pk) == 1 else {}
        drift_quality = 1.0 - min(1.0, float((stress.get("n_drifted_attributes", 0) or 0) / max(1, stress.get("attributes_checked_count", 1) or 1)))
        family_quality = 1.0 if families else 0.4
        coverage_quality = len(covered_by_layout) / max(1, len(all_columns))
        simplicity_penalty = max(0.0, (len(tables) - 2) * 0.05)
        collision_quality = 1.0 - min(1.0, float(g.get("collision_severity_score", 1.0) or 1.0))
        dep_quality = min(1.0, len(dep_support) / 5.0)
        score = (
            0.30 * collision_quality +
            0.18 * drift_quality +
            0.15 * family_quality +
            0.12 * dep_quality +
            0.15 * coverage_quality +
            0.10 * (1.0 - simplicity_penalty)
        )

        rationale = [_rationale_unique_grain(g)]
        if drift_quality >= 0.7 and len(entity_attrs) >= 3:
            rationale.append({
                "code": "high_invariance_entity_attributes",
                "text": f"{len(entity_attrs)} columns are likely invariant under {','.join(pk) or 'candidate key'} (drift threshold 0.3).",
                "evidence": {"A5": {"drift_quality": round(drift_quality, 6), "candidate": pk[0] if pk else None}},
            })
        if families and repeat_dims:
            rationale.append({
                "code": "family_repeat_structure_detected",
                "text": f"A repeated family structure was detected with candidate repeat dimension {repeat_dims[0]}.",
                "evidence": {"A9": {"repeat_dimension": repeat_dims[0]}, "A8": {"family_count": len(families)}},
            })

        layouts.append({
            "layout_id": f"L{i}",
            "score": round(score, 6),
            "summary": {
                "current_row_grain": pk,
                "proposed_model": "entity + family tables" if families else ("entity + event table" if repeat_dims else "single-entity-centric"),
                "rationale": [r["text"] for r in rationale if r],
            },
            "grain": {
                "keys_tested": pk,
                "dup_row_count": int(g.get("dup_row_count", 0) or 0),
                "collision_severity_score": round(float(g.get("collision_severity_score", 1.0) or 1.0), 6),
                "evidence_refs": ["A6", "A9", "A10"],
            },
            "tables": tables,
            "coverage_summary": {
                "covered_by_layout_count": len(covered_by_layout),
                "covered_by_role_only_count": len(covered_by_role_only),
                "ambiguous_count": len(ambiguous),
                "unmapped_count": len(unmapped),
            },
            "column_placement": {
                "covered_by_layout": sorted(covered_by_layout),
                "covered_by_role_only": covered_by_role_only,
                "ambiguous": ambiguous,
                "unmapped": unmapped,
            },
            "ambiguities": [
                {
                    "column": c,
                    "primary_role": role_assignments[c]["primary_role"],
                    "alternate_roles": role_assignments[c]["alternate_roles"],
                    "why": "multiple plausible role assignments above threshold",
                }
                for c in ambiguous[:25]
            ],
            "debug_trace": {
                "role_thresholds": {"primary_min": 0.5, "alternate_min": 0.35},
                "families_used": [f.get("family_id") for f in families],
                "evidence_links": {"A5": "key seed + stress tests", "A6": "grain tests", "A9": "role candidates", "A10": "relationship support"},
                "rationale_items": rationale,
                "scoring_components": {
                    "collision_quality": round(collision_quality, 6),
                    "drift_quality": round(drift_quality, 6),
                    "family_quality": round(family_quality, 6),
                    "dependency_support_quality": round(dep_quality, 6),
                    "coverage_quality": round(coverage_quality, 6),
                    "simplicity_penalty": round(simplicity_penalty, 6),
                },
            },
        })

    layouts = sorted(layouts, key=lambda x: (-float(x.get("score", 0.0) or 0.0), x.get("layout_id", "")))
    return {
        "version": "2",
        "layout_candidates": layouts,
        "debug": {
            "candidate_layout_count": len(layouts),
            "all_columns_count": len(all_columns),
        },
    }


@app.post("/full-bundle")
async def full_bundle(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    max_categorical_cardinality: int = Form(20),
    _=Depends(require_token),
) -> Dict[str, Any]:
    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    dataset_sha256 = sha256_hex(raw_bytes)
    df = _read_dataframe(raw_bytes, file.filename)
    n_rows, n_cols = df.shape

    profile_upload = UploadFile(filename=file.filename, file=io.BytesIO(raw_bytes), headers=file.headers)
    profile_result = await _run_full_profile(
        request=request,
        file=profile_upload,
        dataset_id=dataset_id,
        max_categorical_cardinality=max_categorical_cardinality,
        include_samples=True,
        include_pattern_examples=True,
        include_deep_profiles=True,
    )
    cols_profile = profile_result["data_profile"]["columns"]

    # Canonical per-column evidence primitives used across downstream artifacts.
    # This prevents diverging field calculations between A2/A5/A9 and dependents.
    column_signal_map: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        p = cols_profile.get(col, {})
        candidates = p.get("candidates") or []
        top = candidates[0] if candidates else {}
        column_signal_map[col] = {
            "unique_count": int(p.get("unique_count", int(df[col].nunique(dropna=True)) or 0)),
            "unique_ratio": float((int(p.get("unique_count", int(df[col].nunique(dropna=True)) or 0)) / n_rows) if n_rows else 0.0),
            "missing_count": int(p.get("missing_count", int(df[col].isna().sum()) or 0)),
            "missing_pct": float(p.get("missing_pct", round((int(p.get("missing_count", int(df[col].isna().sum()) or 0)) / n_rows * 100.0) if n_rows else 0.0, 6))),
            "parsed_as_numeric_pct": float((p.get("numeric_profile") or {}).get("parseable_pct", 0.0) or 0.0),
            "parsed_as_datetime_pct": float((p.get("date_profile") or {}).get("parseable_pct", 0.0) or 0.0),
            "top_candidate_type": str(top.get("type") or ""),
            "top_candidate_confidence": float(top.get("confidence", 0.0) or 0.0),
            "top_candidate_parse": top.get("parse"),
            "top_candidate_op": top.get("op"),
        }

    artifact_inputs: Dict[str, Any] = {
        "dataset": {
            "source": "uploaded_csv",
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha256,
            "n_rows": int(n_rows),
            "n_columns": int(n_cols),
        },
        "A1": {"uses": ["dataset", "profile_result", "association_result", "all_artifact_outputs"]},
        "A2": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A3-T": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A3-V": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A4": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A5": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A6": {"uses": ["A5", "A8", "A9", "dataset", "column_signal_map"]},
        "A7": {"uses": ["dataset", "A6"]},
        "A8": {"uses": ["dataset_columns", "tokenized_column_names"]},
        "A9": {"uses": ["A2", "A5", "A8", "A13", "column_signal_map", "cols_profile"]},
        "A10": {"uses": ["association_result", "dataset"]},
        "A11": {"uses": ["A5", "A8", "A9", "A10", "dataset"]},
        "A12": {"uses": ["A5", "A6", "A8", "A9", "A10", "dataset"]},
        "A13": {"uses": ["A2", "dataset", "cols_profile"]},
        "A14": {"uses": ["dataset", "A2", "cols_profile"]},
        "A16": {"uses": ["dataset", "A4", "cols_profile", "column_signal_map"]},
        "A17": {"uses": ["A2", "A3-T", "A3-V", "A4", "A9", "A13", "A14", "A16"]},
        "B1": {"uses": ["A8", "A2", "A6", "A11", "A13", "A14"]},
    }

    association_upload = UploadFile(filename="bundle_assoc.csv", file=io.BytesIO(df.to_csv(index=False).encode("utf-8")), headers=file.headers)
    association_result = await evidence_associations(
        request=request,
        file=association_upload,
        dataset_id=dataset_id,
        max_categorical_cardinality=max_categorical_cardinality,
        _=None,
    )

    missing_global: Dict[str, int] = defaultdict(int)
    missing_by_col: Dict[str, Dict[str, int]] = {}
    token_norm_global: Dict[str, int] = defaultdict(int)
    token_norm_meta: Dict[str, Dict[str, Any]] = {}
    true_na_total = 0
    token_missing_like_total = 0
    per_column_missingness: List[Dict[str, Any]] = []
    column_dictionary_rows: List[Dict[str, Any]] = []

    for col in df.columns:
        p = cols_profile.get(col, {})
        sig = column_signal_map.get(col, {})
        missing_tokens = p.get("missing_like_tokens", {})
        missing_by_col[col] = missing_tokens
        for tkn, cnt in missing_tokens.items():
            missing_global[tkn] += int(cnt)

        candidates = p.get("candidates") or []
        top_candidate = candidates[0] if candidates else {}
        second_candidate = candidates[1] if len(candidates) > 1 else {}
        type_conf = float(sig.get("top_candidate_confidence", top_candidate.get("confidence", 0.0) or 0.0) or 0.0)
        unique_ratio = float(sig.get("unique_ratio", (p.get("unique_count", 0) / n_rows) if n_rows else 0.0) or 0.0)
        one_hot_like = _is_one_hot_like_column(col, p, df[col])
        top_frequent = p.get("samples", {}).get("top_frequent", [])[:8]

        missing_count = int(sig.get("missing_count", p.get("missing_count", int(df[col].isna().sum()))))
        missing_pct = float(sig.get("missing_pct", p.get("missing_pct", round((missing_count / n_rows * 100.0) if n_rows else 0.0, 6))))
        token_count = int(sum(int(v) for v in missing_tokens.values()))
        true_na_count = int(max(0, missing_count - token_count))
        true_na_total += true_na_count
        token_missing_like_total += token_count

        for raw_token, cnt in missing_tokens.items():
            normalized = raw_token.strip().lower() if isinstance(raw_token, str) else str(raw_token)
            token_norm_global[normalized] += int(cnt)
            if normalized not in token_norm_meta:
                token_norm_meta[normalized] = {"raw_examples": set(), "columns": set()}
            token_norm_meta[normalized]["raw_examples"].add(raw_token)
            token_norm_meta[normalized]["columns"].add(col)

        per_column_missingness.append({
            "column": col,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 6),
            "true_na_count": true_na_count,
            "token_missing_like_count": token_count,
            "token_breakdown": missing_tokens,
            "missingness_mode": "mixed" if (true_na_count > 0 and token_count > 0) else ("token_only" if token_count > 0 else ("true_na_only" if true_na_count > 0 else "none")),
        })

        row = {
            "column": col,
            "top_candidate": {
                "type": top_candidate.get("type"),
                "confidence": round(type_conf, 6),
                "parse": top_candidate.get("parse"),
                "op": top_candidate.get("op"),
                "evidence_summary": _candidate_evidence_summary(top_candidate),
            },
            "candidate_diagnostics": {
                "candidate_count": len(candidates),
                "second_candidate_type": second_candidate.get("type"),
                "second_candidate_confidence": round(float(second_candidate.get("confidence", 0.0) or 0.0), 6),
                "confidence_gap_1_2": round(type_conf - float(second_candidate.get("confidence", 0.0) or 0.0), 6),
            },
            "missing_tokens_observed": missing_tokens,
            "confidence": round(type_conf, 6),
            "confidence_gap": round(type_conf - float(second_candidate.get("confidence", 0.0) or 0.0), 6),
            "missing_pct": missing_pct,
            "unique_count": p.get("unique_count", 0),
            "unique_ratio": round(unique_ratio, 6),
            "type_confidence_components": {
                "uniq_score": round(float((top_candidate.get("evidence") or {}).get("uniq_score", 0.0) or 0.0), 6),
                "structural_signal": round(float((top_candidate.get("evidence") or {}).get("structural_signal", 0.0) or 0.0), 6),
                "parse_signal": round(float((top_candidate.get("evidence") or {}).get("parse_signal", 0.0) or 0.0), 6),
            },
            "high_missingness": bool(missing_pct >= 50.0),
            "top_levels": [x.get("value") for x in top_frequent],
            "profiler_samples": top_frequent,
            "a2_samples": _series_samples(df[col], dataset_sha256, col),
            "is_one_hot_like": one_hot_like,
            "numeric_profile": p.get("numeric_profile"),
            "datetime_profile": p.get("date_profile"),
            "high_uniqueness_candidate": bool(unique_ratio >= 0.98),
            "artifact_inputs": artifact_inputs["A2"]["uses"],
        }
        column_dictionary_rows.append(row)

    transform_review_queue = _build_transform_review_queue(df, cols_profile, max_categorical_cardinality)
    transform_review_queue["artifact"] = "A3-T"
    transform_review_queue["purpose"] = "transform_review_queue"
    transform_review_queue["inputs"] = artifact_inputs["A3-T"]

    variable_type_review_queue = _build_variable_type_review_queue(df, cols_profile, max_categorical_cardinality)
    variable_type_review_queue["artifact"] = "A3-V"
    variable_type_review_queue["purpose"] = "variable_type_review_queue"
    variable_type_review_queue["inputs"] = artifact_inputs["A3-V"]

    semantic_anchor_rows: List[Dict[str, Any]] = []
    semantic_anchor_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in column_dictionary_rows:
        col = row["column"]
        s_non_na = df[col].dropna()
        s_str = s_non_na.astype("string").str.strip()
        non_empty = s_str[s_str != ""]
        n = int(len(non_empty))
        anchors: List[Dict[str, Any]] = []

        def _rate(mask: pd.Series) -> float:
            return float(mask.mean()) if len(mask) else 0.0

        if n:
            zip_rate = _rate(non_empty.str.match(r"^\d{5}(?:-\d{4})?$", na=False))
            if zip_rate >= 0.9:
                anchors.append({"anchor": "US_ZIP_CODE", "confidence": round(zip_rate, 6), "implication": "FORCE_STRING_CAST", "cleaning_hint": "zfill(5)"})
                anchors.append({"anchor": "PII_LOCATION", "confidence": round(min(1.0, 0.8 + 0.2 * zip_rate), 6), "implication": "FLAG_FOR_ANONYMIZATION"})

            email_rate = _rate(non_empty.str.contains(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", regex=True, na=False))
            if email_rate >= 0.9:
                anchors.append({"anchor": "EMAIL", "confidence": round(email_rate, 6), "implication": "PII_SENSITIVE", "cleaning_hint": "lower().strip()"})

            url_rate = _rate(non_empty.str.contains(r"^https?://", regex=True, na=False))
            if url_rate >= 0.9:
                anchors.append({"anchor": "URL", "confidence": round(url_rate, 6), "implication": "PII_POSSIBLE", "cleaning_hint": "strip_url_whitespace"})

            phone_rate = _rate(non_empty.str.contains(r"^(?:\+?\d[\d\-() ]{7,}\d)$", regex=True, na=False))
            if phone_rate >= 0.9:
                anchors.append({"anchor": "PHONE", "confidence": round(phone_rate, 6), "implication": "PII_SENSITIVE", "cleaning_hint": "normalize_e164"})

            country_rate = _rate(non_empty.str.lower().isin({"us", "usa", "united states", "uk", "united kingdom", "ca", "canada", "au", "australia"}))
            if country_rate >= 0.7:
                anchors.append({"anchor": "ISO_COUNTRY", "confidence": round(country_rate, 6), "implication": "STANDARDIZE_TO_ISO3166"})

            state_rate = _rate(non_empty.str.upper().str.match(r"^[A-Z]{2}$", na=False))
            if state_rate >= 0.8 and any(t in col.lower() for t in ["state", "province", "region"]):
                anchors.append({"anchor": "STATE_CODE", "confidence": round(state_rate, 6), "implication": "STANDARDIZE_SUBNATIONAL_CODE"})

            currency_sym_rate = _rate(non_empty.str.contains(r"[$€£¥]", regex=True, na=False))
            if currency_sym_rate >= 0.7:
                anchors.append({"anchor": "CURRENCY", "confidence": round(currency_sym_rate, 6), "implication": "SPLIT_VALUE_AND_CURRENCY", "cleaning_hint": "extract_currency_symbol"})

            icd_rate = _rate(non_empty.str.upper().str.match(r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?$", na=False))
            if icd_rate >= 0.7:
                anchors.append({"anchor": "ICD10_CODE", "confidence": round(icd_rate, 6), "implication": "MEDICAL_CODE_DICTIONARY_LOOKUP"})

            fiscal_rate = _rate(non_empty.str.upper().str.match(r"^(Q[1-4][-_/]?(?:19|20)?\d{2}|(?:19|20)\d{2}[-_/]?Q[1-4])$", na=False))
            if fiscal_rate >= 0.7:
                anchors.append({"anchor": "FISCAL_QUARTER", "confidence": round(fiscal_rate, 6), "implication": "TEMPORAL_AXIS_QUARTERLY"})

        semantic_anchor_map[col] = anchors
        semantic_anchor_rows.append({
            "column": col,
            "technical_type": str((row.get("top_candidate") or {}).get("type") or "unknown"),
            "detected_anchors": anchors,
        })

    semantic_anchors = {
        "artifact": "A13",
        "purpose": "semantic_anchor_overlay",
        "inputs": artifact_inputs["A13"],
        "columns": semantic_anchor_rows,
    }

    segment_count = 5
    quality_rows: List[Dict[str, Any]] = []
    for col in df.columns:
        segs: List[Dict[str, Any]] = []
        size = max(1, int((n_rows + segment_count - 1) / segment_count))
        null_pcts = []
        date_success = []
        for i in range(segment_count):
            start_i = i * size
            if start_i >= n_rows:
                break
            end_i = min(n_rows, (i + 1) * size)
            chunk = df[col].iloc[start_i:end_i]
            null_pct = float(chunk.isna().mean() * 100.0) if len(chunk) else 0.0
            null_pcts.append(null_pct)
            non_na = chunk.dropna().astype("string").str.strip()
            top_format = "unparsed"
            stability = max(0.0, 1.0 - (null_pct / 100.0))
            if len(non_na):
                iso_rate = float(non_na.str.match(r"^\d{4}-\d{2}-\d{2}", na=False).mean())
                uk_rate = float(non_na.str.match(r"^\d{1,2}/\d{1,2}/\d{4}", na=False).mean())
                dt = pd.to_datetime(non_na, errors="coerce")
                dt_rate = float(dt.notna().mean())
                date_success.append(dt_rate)
                if iso_rate >= 0.7:
                    top_format = "ISO8601"
                elif uk_rate >= 0.7:
                    top_format = "UK_DATE"
                elif dt_rate >= 0.7:
                    top_format = "DATETIME_GENERIC"
                elif pd.to_numeric(non_na, errors="coerce").notna().mean() >= 0.7:
                    top_format = "NUMERIC"
                else:
                    top_format = "STRING"
                stability = max(0.0, min(1.0, 0.6 * (1.0 - null_pct / 100.0) + 0.4 * dt_rate))
            segs.append({
                "row_range": [int(start_i), int(end_i - 1)],
                "null_pct": round(null_pct, 6),
                "top_format": top_format,
                "stability": round(stability, 6),
            })

        drift_score = 0.0
        if segs:
            drift_score += (max(null_pcts) - min(null_pcts)) / 100.0 if null_pcts else 0.0
            if date_success:
                drift_score += (max(date_success) - min(date_success))
            formats = [x["top_format"] for x in segs]
            drift_score += 0.2 if len(set(formats)) > 1 else 0.0
        drift_score = float(min(1.0, max(0.0, drift_score)))
        global_quality = float(max(0.0, 1.0 - drift_score))
        interpretation = "Stable"
        if drift_score >= 0.5:
            interpretation = f"High drift detected around row {segs[1]['row_range'][0] if len(segs) > 1 else 0}. Likely format/source change."
        elif drift_score >= 0.25:
            interpretation = "Moderate drift detected across segments."

        quality_rows.append({
            "column": col,
            "global_quality_score": round(global_quality, 6),
            "drift_detected": bool(drift_score >= 0.25),
            "segments": segs,
            "interpretation": interpretation,
        })

    quality_heatmap = {
        "artifact": "A14",
        "purpose": "quality_heatmap_structural_entropy_and_drift",
        "inputs": artifact_inputs["A14"],
        "segment_count": segment_count,
        "columns": quality_rows,
    }

    token_classification = []
    for tok_norm, cnt in sorted(token_norm_global.items(), key=lambda kv: (-kv[1], kv[0])):
        raw_examples = sorted(token_norm_meta.get(tok_norm, {}).get("raw_examples", set()))
        cls = "sentinel_numeric" if re.fullmatch(r"-?\d+(\.\d+)?", tok_norm or "") else (
            "refused" if "prefer not" in tok_norm else (
                "not_applicable" if tok_norm in {"n/a", "na", "not applicable"} else (
                    "dont_know" if tok_norm in {"dk", "don't know", "dont know"} else (
                        "blank_string" if tok_norm in {"", "<whitespace_only>"} else "missing"
                    )
                )
            )
        )
        token_classification.append({
            "raw_token_examples": raw_examples,
            "normalized_token": tok_norm,
            "class_candidate": cls,
            "evidence": {
                "columns": sorted(token_norm_meta.get(tok_norm, {}).get("columns", set())),
                "global_count": int(cnt),
            },
        })

    columns_with_any_missing = int(sum(1 for r in per_column_missingness if r["missing_count"] > 0))
    notes: List[str] = []
    if token_missing_like_total == 0:
        notes.append("No explicit token-based missingness detected in parsed values.")
    if true_na_total > 0:
        notes.append(f"Missingness is present as parser-level NA values in {columns_with_any_missing} columns.")

    missing_catalog = {
        "artifact": "A4",
        "purpose": "missingness_catalog",
        "inputs": artifact_inputs["A4"],
        "summary": {
            "columns_with_any_missing": columns_with_any_missing,
            "total_missing_cells": int(sum(r["missing_count"] for r in per_column_missingness)),
            "explicit_missing_tokens_detected": bool(token_missing_like_total > 0),
            "notes": notes,
        },
        "global_missingness": {
            "true_na_total": int(true_na_total),
            "token_missing_like_total": int(token_missing_like_total),
            "whitespace_only_total": int(missing_global.get("<WHITESPACE_ONLY>", 0)),
        },
        "global_tokens": dict(sorted(missing_global.items(), key=lambda kv: (-kv[1], kv[0]))),
        "token_classification": token_classification,
        "per_column": per_column_missingness,
        "audit_trace": {
            "token_detection_scope": "parsed_non_na_string_values",
            "normalization": ["strip_whitespace", "casefold"],
            "classification_rules_version": "v2",
        },
    }

    conditional_missingness = _build_conditional_missingness_artifact(
        df=df,
        cols_profile=cols_profile,
        column_signal_map=column_signal_map,
        artifact_inputs=artifact_inputs,
    )

    key_seed_candidates = []
    excluded_seed_columns = []
    for col in df.columns:
        s = df[col]
        sig = column_signal_map.get(col, {})
        s_str = s.astype("string")
        uniq_pct = float(sig.get("unique_ratio", (s.nunique(dropna=True) / n_rows) if n_rows else 0.0) or 0.0)
        miss_pct = float((sig.get("missing_count", int(s.isna().sum())) / n_rows) if n_rows else 0.0)
        entropy_signal = 1.0 - min(1.0, float((s_str.value_counts(normalize=True, dropna=True).head(1).sum())) if n_rows else 0.0)
        id_shape = bool(s_str.str.match(r"^[A-Za-z0-9_-]{4,}$", na=False).mean() > 0.6)
        is_url_like = bool(s_str.str.contains(r"^https?://", na=False, regex=True).mean() > 0.3)
        is_file_path_like = bool(s_str.str.contains(r"[A-Za-z]:\\|/|\\", na=False, regex=True).mean() > 0.5)
        name_semantic_risk = bool(re.search(r"url|uri|link|image|img|photo|avatar|file|path", col, re.IGNORECASE))
        likely_external_resource_id = bool(is_url_like or (name_semantic_risk and "id" not in col.lower()))
        likely_measurement_value = bool(pd.api.types.is_numeric_dtype(s) and uniq_pct < 0.95 and any(k in col.lower() for k in ["score", "total", "mean", "avg", "pct", "percent", "amount", "value"]))

        semantic_risk_flags = []
        if is_url_like:
            semantic_risk_flags.append("url_like")
        if is_file_path_like:
            semantic_risk_flags.append("file_path_like")
        if likely_external_resource_id:
            semantic_risk_flags.append("external_resource_identifier")
        if likely_measurement_value:
            semantic_risk_flags.append("likely_measurement_value")
        if name_semantic_risk:
            semantic_risk_flags.append("name_semantic_risk")

        stability = max(0.0, min(1.0, uniq_pct * (1.0 - miss_pct) * (0.7 + 0.3 * entropy_signal)))
        key_seed_candidates.append({
            "column": col,
            "seed_score": round(stability, 6),
            "evidence": {
                "unique_ratio": round(uniq_pct, 6),
                "null_pct": round(miss_pct * 100, 6),
                "id_like_name": bool(re.search(r"(^|_)(id|key|uuid|guid)($|_)", col, re.IGNORECASE)),
                "regex_id_shape": id_shape,
                "entropy_like": round(entropy_signal, 6),
                "value_pattern": "url_like" if is_url_like else ("numeric_id_like" if pd.api.types.is_numeric_dtype(s) and uniq_pct >= 0.95 else "string_id_like" if id_shape else "unknown"),
            },
            "semantic_risk_flags": semantic_risk_flags,
            "use_as": "row_locator_only" if semantic_risk_flags else "composite_key_seed_or_primary_id_candidate",
            "not_final_key": True,
        })
        if semantic_risk_flags:
            excluded_seed_columns.append({"column": col, "reason": "+".join(semantic_risk_flags)})

    key_seed_candidates.sort(key=lambda x: (-x["seed_score"], x["column"]))
    key_seed_candidates = key_seed_candidates[: min(12, len(key_seed_candidates))]

    key_candidate_cols = [k["column"] for k in key_seed_candidates]

    invariants_ranked = []
    for col in df.columns:
        if col in key_candidate_cols:
            continue
        s = df[col]
        uniq = int(s.nunique(dropna=True))
        miss = float(s.isna().mean()) if n_rows else 0.0
        if uniq <= max(20, int(n_rows * 0.02)):
            invariants_ranked.append((miss, uniq, col))
    invariants_ranked.sort(key=lambda t: (t[0], t[1], t[2]))
    attrs = [c for _, _, c in invariants_ranked[:12]]

    candidate_key_stress_tests = []
    for k in key_seed_candidates[:8]:
        key_col = k["column"]
        grouped = df[[key_col] + attrs].dropna(subset=[key_col]).groupby(key_col, dropna=True)
        drifted = []
        for a in attrs:
            v = grouped[a].nunique(dropna=True)
            conflict_groups = int((v > 1).sum())
            if conflict_groups > 0:
                bad_ids = v[v > 1].index.tolist()[:3]
                eg = []
                for bid in bad_ids[:2]:
                    rows = df[df[key_col] == bid][[key_col, a]].drop_duplicates().head(2).to_dict(orient="records")
                    eg.append({"id": str(bid), "rows": rows})
                drifted.append({
                    "attribute": a,
                    "drift_groups": conflict_groups,
                    "drift_rate": round(float(conflict_groups / max(1, grouped.ngroups)), 6),
                    "example_conflicts": eg,
                })
        drifted = sorted(drifted, key=lambda d: (-d["drift_groups"], d["attribute"]))
        candidate_key_stress_tests.append({
            "candidate_column": key_col,
            "attributes_checked_count": len(attrs),
            "n_drifted_attributes": len(drifted),
            "top_drifted_attributes": drifted[:8],
            "interpretation": "Likely row identifier but not entity-invariant across records" if drifted else "Likely stable single-column identifier",
        })

    recommended_seed_columns = [k["column"] for k in key_seed_candidates if not k.get("semantic_risk_flags")][:8]
    key_integrity = {
        "artifact": "A5",
        "purpose": "key_seed_candidates_and_stress_tests",
        "inputs": artifact_inputs["A5"],
        "single_column_key_seed_candidates": key_seed_candidates,
        "candidate_key_stress_tests": candidate_key_stress_tests,
        "handoff_to_A6": {
            "recommended_seed_columns": recommended_seed_columns,
            "excluded_seed_columns": excluded_seed_columns,
            "recommended_a6_seed_columns": recommended_seed_columns,
            "excluded_from_seeding_reasons": excluded_seed_columns,
        },
        "invariant_candidates_checked": attrs,
    }

    repeat_candidates = _build_families(list(df.columns))
    repeat_candidates["inputs"] = artifact_inputs["A8"]
    families = repeat_candidates.get("families", [])

    repeat_index_cols = sorted({c for fam in families for c in fam.get("columns", [])})
    family_repeat_dims = [f.get("recommended_repeat_dimension_name") for f in families if f.get("recommended_repeat_dimension_name")]
    role_time_candidates = [
        c for c in df.columns
        if (float((cols_profile.get(c, {}).get("date_profile") or {}).get("parseable_pct", 0.0) or 0.0) >= 60.0)
        or any(t in c.lower() for t in ["time", "date", "visit", "wave", "month", "day", "week", "year", "session", "trial", "tp"])
    ]
    repeat_name_candidates = [
        c for c in df.columns
        if _parse_repeat_structure_name(c) is not None
    ]

    # (Decision) Never re-add excluded seeds (row index, timestamp-only, etc.) via key_candidate_cols.
    # Why it matters: avoids trivial 100%-unique seeds that make all composites meaningless.
    recommended = (key_integrity.get("handoff_to_A6", {}).get("recommended_seed_columns") or [])
    excluded_meta = (key_integrity.get("handoff_to_A6", {}).get("excluded_seed_columns") or [])
    excluded: set = set()
    for item in excluded_meta:
        if isinstance(item, dict):
            col = item.get("column")
            if col:
                excluded.add(str(col))
        elif item:
            excluded.add(str(item))

    row_index_like = re.compile(r"(^unnamed:?\s*\d+$|\bindex\b|\brow_?id\b|\brownum\b)", re.IGNORECASE)
    id_candidates = [
        c for c in list(dict.fromkeys(recommended))
        if c in df.columns
        and c not in excluded
        and not row_index_like.search(str(c))
        and not any(t in str(c).lower() for t in ["date", "time", "timestamp"])
    ][:10]

    grouping_candidates = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ["site", "arm", "condition", "group", "cohort", "cluster", "school", "center", "centre", "treat"])
    ][:10]

    repeat_dimension_candidates = list(dict.fromkeys(
        repeat_index_cols
        + repeat_name_candidates
        + [c for c in df.columns if any(k in c.lower() for k in ["wave", "visit", "session", "trial", "row", "item", "tp", "timepoint"])][:10]
    ))[:16]
    time_like = list(dict.fromkeys(role_time_candidates))[:10]

    family_hint_by_col: Dict[str, Dict[str, Any]] = {}
    for fam in families:
        repeat_dim = fam.get("recommended_repeat_dimension_name")
        pattern = f"id+{repeat_dim}" if repeat_dim else "id+family"
        for c in fam.get("columns", []):
            family_hint_by_col[c] = {
                "pattern": pattern,
                "source": "A8.downstream_hints",
                "family_id": fam.get("family_id"),
            }

    qualifier_meta: Dict[str, Dict[str, str]] = {}
    for c in repeat_dimension_candidates:
        qualifier_meta.setdefault(c, {"bucket": "repeat_dimension_candidates", "pattern": family_hint_by_col.get(c, {}).get("pattern", "id+row")})
    for c in time_like:
        qualifier_meta.setdefault(c, {"bucket": "time_candidates", "pattern": "id+time"})
    for c in grouping_candidates:
        qualifier_meta.setdefault(c, {"bucket": "grouping_candidates", "pattern": "id+group"})

    DELTA = 0.05
    MAX_INFORMATIVE_TESTS = 60
    uniqueness_cache: Dict[Tuple[str, ...], float] = {}

    def _u(cols: List[str]) -> float:
        k = tuple(cols)
        if k not in uniqueness_cache:
            if not cols:
                uniqueness_cache[k] = 0.0
            else:
                uniq_rows = int((~df.duplicated(subset=cols, keep=False)).sum())
                uniqueness_cache[k] = float(uniq_rows / max(1, n_rows))
        return uniqueness_cache[k]

    candidate_sets: List[Dict[str, Any]] = []
    for base in id_candidates:
        base_u = _u([base])
        candidate_sets.append({"keys": [base], "base_key": base, "added_qualifiers": [], "base_u": base_u, "with_u": base_u, "gain": 0.0, "composite_validity_hint": {"pattern": "id", "source": "A5.recommended_seed_columns"}})
        if base_u >= 0.99:
            continue

        # Prefer semantically valid composites first (ID + repeat/time/grouping).
        ordered_qualifiers = [
            q for q in list(dict.fromkeys(repeat_dimension_candidates + time_like + grouping_candidates))
            if q != base and q in df.columns
        ]
        for q in ordered_qualifiers:
            uq = _u([q])
            with_u = _u([base, q])
            gain = with_u - base_u
            if uq >= 0.99:
                continue
            if gain < DELTA:
                continue
            hint = family_hint_by_col.get(q, {"pattern": qualifier_meta.get(q, {}).get("pattern", "id+qualifier"), "source": "A8.downstream_hints" if q in repeat_dimension_candidates else "A6.heuristics"})
            candidate_sets.append({
                "keys": [base, q],
                "base_key": base,
                "added_qualifiers": [q],
                "base_u": base_u,
                "with_u": with_u,
                "gain": gain,
                "qualifier_u": uq,
                "qualifier_bucket": qualifier_meta.get(q, {}).get("bucket", "unknown"),
                "composite_validity_hint": hint,
            })

    # deterministic de-dupe + informative cap
    seen = set()
    dedup_candidates: List[Dict[str, Any]] = []
    for item in candidate_sets:
        t = tuple(item.get("keys", []))
        if t in seen:
            continue
        seen.add(t)
        dedup_candidates.append(item)

    informative = [x for x in dedup_candidates if x.get("added_qualifiers")]
    informative = sorted(informative, key=lambda x: (-float(x.get("gain", 0.0)), float(x.get("base_u", 0.0)), str(x.get("base_key", "")), ",".join(x.get("added_qualifiers", []))))[:MAX_INFORMATIVE_TESTS]
    base_only = sorted([x for x in dedup_candidates if not x.get("added_qualifiers")], key=lambda x: (float(x.get("base_u", 1.0)), str(x.get("base_key", ""))))[: min(10, len(id_candidates))]
    eval_sets = base_only + informative

    grain_tests = []
    for idx, test_item in enumerate(eval_sets, start=1):
        keys = test_item.get("keys", [])
        dup_mask = df.duplicated(subset=keys, keep=False)
        dup_rows = int(dup_mask.sum())
        dup_groups = int(df[df.duplicated(subset=keys, keep=False)].groupby(keys, dropna=False).ngroups) if dup_rows else 0
        uniq_rows = int((~df.duplicated(subset=keys, keep=False)).sum())
        uniqueness_rate = float(uniq_rows / max(1, n_rows))
        non_keys = [c for c in df.columns if c not in keys]
        identical_pct = 0.0
        conflict_summaries = []
        non_key_conflicting_groups = 0
        non_key_identical_groups = 0

        if dup_rows and non_keys:
            g = df[df.duplicated(subset=keys, keep=False)].groupby(keys, dropna=False)
            total_groups = 0
            for group_key, gdf in g:
                total_groups += 1
                dedup_non_key = gdf[non_keys].drop_duplicates()
                if len(dedup_non_key) == 1:
                    non_key_identical_groups += 1
                    continue
                non_key_conflicting_groups += 1
                if len(conflict_summaries) < 5:
                    row_ids = list(gdf.index[:2])
                    if len(row_ids) >= 2:
                        a = gdf.loc[row_ids[0], non_keys]
                        b = gdf.loc[row_ids[1], non_keys]
                        differing = [c for c in non_keys if (a[c] != b[c]) and not (pd.isna(a[c]) and pd.isna(b[c]))]
                    else:
                        differing = []
                    conflict_summaries.append({
                        "key": list(group_key) if isinstance(group_key, tuple) else [group_key],
                        "differing_columns_count": len(differing),
                        "differing_columns_preview": differing[:5],
                        "row_refs": [f"r{int(i)}" for i in row_ids],
                    })
            identical_pct = float((non_key_identical_groups / max(1, total_groups)) * 100.0)
        conflict_group_pct = float(non_key_conflicting_groups / max(1, dup_groups)) if dup_groups else 0.0

        fam_hits = [f.get("family_id") for f in families if any(c in f.get("columns", []) for c in keys)]
        repeat_dimension_present = any(k in repeat_dimension_candidates for k in keys)
        temporal_dimension_present = any(k in time_like for k in keys)
        pivot_safety = "safe" if dup_rows == 0 else ("conditional" if conflict_group_pct <= 0.2 else "unsafe")
        decision = "best_available_composite_key" if len(keys) > 1 else ("candidate_single_key" if dup_rows == 0 else "failed_single_key")

        qualifier_u = float(test_item.get("qualifier_u", 0.0) or 0.0)
        triviality_flags: List[str] = []
        if test_item.get("base_u", 0.0) >= 0.99:
            triviality_flags.append("base_near_unique")
        if qualifier_u >= 0.99:
            triviality_flags.append("qualifier_near_unique_alone")

        grain_tests.append({
            "test_id": f"GT{idx}",
            "keys_tested": keys,
            "base_key": test_item.get("base_key"),
            "added_qualifiers": test_item.get("added_qualifiers", []),
            "uniqueness_base": round(float(test_item.get("base_u", uniqueness_rate)), 6),
            "uniqueness_with_qualifiers": round(float(test_item.get("with_u", uniqueness_rate)), 6),
            "uniqueness_gain": round(float(test_item.get("gain", 0.0) or 0.0), 6),
            "triviality_flags": triviality_flags,
            "composite_validity_hint": test_item.get("composite_validity_hint", {"pattern": "id+qualifier", "source": "A6.heuristics"}),
            "uniqueness_rate": round(uniqueness_rate, 6),
            "dup_row_count": dup_rows,
            "dup_groups_count": dup_groups,
            "identical_duplicate_pct": round(identical_pct, 6),
            "non_key_conflict_group_pct": round(conflict_group_pct, 6),
            "non_key_conflict_summary": conflict_summaries,
            "family_evidence": {
                "families_touched": fam_hits,
                "repeat_dimension_present": repeat_dimension_present,
                "temporal_dimension_present": temporal_dimension_present,
            },
            "pivot_safety": pivot_safety,
            "decision": decision,
            "collision_severity_score": round(min(1.0, (0.6 * (dup_rows / max(1, n_rows)) + 0.4 * conflict_group_pct)), 6),
            "same_key_identical_nonkeys_groups": int(non_key_identical_groups),
            "same_key_conflicting_nonkeys_groups": int(non_key_conflicting_groups),
        })
    grain_tests = sorted(grain_tests, key=lambda g: (-float(g.get("uniqueness_gain", 0.0) or 0.0), g.get("collision_severity_score", 1.0), -g.get("uniqueness_rate", 0.0), g.get("dup_row_count", 10**9), len(g.get("keys_tested", []))))
    best_test = grain_tests[0] if grain_tests else None
    runner_ups = [
        {
            "keys": gt.get("keys_tested", []),
            "dup_row_count": gt.get("dup_row_count", 0),
            "reason_failed": "expected repeats / likely longitudinal rows" if (gt.get("dup_row_count", 0) > 0 and len(gt.get("keys_tested", [])) == 1) else "lower_ranked_alternative",
        }
        for gt in grain_tests[1:4]
    ]

    row_grain_status = "failed"
    if best_test:
        if best_test.get("dup_row_count", 0) == 0:
            row_grain_status = "validated"
        elif best_test.get("pivot_safety") == "conditional":
            row_grain_status = "ambiguous"

    grain_validation = {
        "artifact": "A6",
        "purpose": "grain_validation",
        "inputs": artifact_inputs["A6"],
        "row_grain_assessment": {
            "status": row_grain_status,
            "best_candidate": {
                "keys": best_test.get("keys_tested", []) if best_test else [],
                "uniqueness_rate": best_test.get("uniqueness_rate", 0.0) if best_test else 0.0,
                "dup_row_count": best_test.get("dup_row_count", 0) if best_test else 0,
                "dup_groups_count": best_test.get("dup_groups_count", 0) if best_test else 0,
                "non_key_conflict_group_pct": best_test.get("non_key_conflict_group_pct", 0.0) if best_test else 0.0,
                "pivot_safety": best_test.get("pivot_safety", "unsafe") if best_test else "unsafe",
                "decision": best_test.get("decision", "none") if best_test else "none",
            },
            "runner_ups": runner_ups,
        },
        "tests": grain_tests,
        "evidence_links": {"repeat_families": "A8", "role_scores": "A9", "seed_candidates": "A5"},
        "candidate_generation": {
            "id_candidates": id_candidates,
            "repeat_dimension_candidates": repeat_dimension_candidates,
            "time_candidates": time_like,
            "grouping_candidates": grouping_candidates,
            "family_repeat_dimensions": family_repeat_dims,
            "delta_uniqueness_gain_threshold": DELTA,
            "max_informative_tests": MAX_INFORMATIVE_TESTS,
        },
    }

    exact_dup_mask = df.duplicated(keep=False)
    exact_dup_rows = df[exact_dup_mask].copy()
    cluster_map: Dict[str, List[int]] = defaultdict(list)
    if len(exact_dup_rows):
        for idx_row, row in exact_dup_rows.iterrows():
            h = sha256_hex(json.dumps(row.to_dict(), sort_keys=True, default=str).encode("utf-8"))
            cluster_map[h].append(int(idx_row))

    exact_clusters = []
    for i, (h, idxs) in enumerate(sorted(cluster_map.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:50], start=1):
        exact_clusters.append({
            "cluster_id": f"ED{i}",
            "row_count": len(idxs),
            "row_refs": [f"r{x}" for x in idxs[:20]],
            "row_hash": h,
            "all_columns_identical": True,
        })

    best_near = best_test or {}
    duplicates_report = {
        "artifact": "A7",
        "purpose": "duplicate_report",
        "inputs": artifact_inputs["A7"],
        "summary": {
            "n_rows": int(n_rows),
            "rows_in_exact_duplicate_clusters": int(exact_dup_mask.sum()),
            "exact_duplicate_cluster_count": len(cluster_map),
            "exact_duplicate_prevalence_pct": round(float(exact_dup_mask.mean() * 100.0) if n_rows else 0.0, 6),
            "safe_drop_exact_duplicates": "recommended_after_review" if len(cluster_map) > 0 else "not_applicable",
        },
        "exact_duplicate_clusters": exact_clusters,
        "near_duplicate_checks": {
            "grain_ref": "A6",
            "best_grain": best_near.get("keys_tested", []),
            "same_key_identical_nonkeys_groups": int(best_near.get("same_key_identical_nonkeys_groups", 0) or 0),
            "same_key_conflicting_nonkeys_groups": int(best_near.get("same_key_conflicting_nonkeys_groups", 0) or 0),
        },
        "actions": [
            {
                "action": "drop_exact_duplicates",
                "status": "recommended_after_review" if len(cluster_map) > 0 else "no_action",
                "why": "all sampled clusters are fully identical" if len(cluster_map) > 0 else "no exact duplicate clusters detected",
            }
        ],
        "grain_context_ref": "A6",
    }

    role_scores = {
        "artifact": "A9",
        "purpose": "role_type_detection",
        "inputs": artifact_inputs["A9"],
        "evidence_primitives": {
            "source": "column_signal_map",
            "fields": [
                "unique_count", "unique_ratio", "missing_pct",
                "parsed_as_numeric_pct", "parsed_as_datetime_pct",
                "top_candidate_type", "top_candidate_confidence", "top_candidate_parse", "top_candidate_op",
            ],
        },
        "columns": [],
    }
    family_member_cols = {c for fam in families for c in fam.get("columns", [])}
    family_repeat_names = {f.get("recommended_repeat_dimension_name") for f in families if f.get("recommended_repeat_dimension_name")}
    likert_tokens = {"strongly", "agree", "disagree", "neutral", "sometimes", "often", "never", "always", "rarely", "satisfied", "unsatisfied"}

    for row in column_dictionary_rows:
        uniq = float(row["unique_ratio"])
        top_candidate = row.get("top_candidate") or {}
        top_type = str(top_candidate.get("type") or "")
        top_conf = float(top_candidate.get("confidence", 0.0) or 0.0)
        parsed_as_numeric_pct = float((row.get("numeric_profile") or {}).get("parseable_pct", 0.0) or 0.0)
        parsed_as_datetime_pct = float((row.get("datetime_profile") or {}).get("parseable_pct", 0.0) or 0.0)
        unique_count = int(row.get("unique_count", 0) or 0)
        sample_values = ((row.get("a2_samples") or {}).get("random") or [])[:5]
        repeat_parse = _parse_repeat_structure_name(row["column"])

        measure_numeric_signal = max(parsed_as_numeric_pct / 100.0, 1.0 if top_type in {"numeric", "numeric_with_unit", "percent", "numeric_range"} else 0.0)
        family_member_hint = row["column"] in family_member_cols
        coded_cat_hint = bool(unique_count <= max_categorical_cardinality and top_type in {"categorical", "categorical_multi", "text", "mixed", "numeric"})
        identifier_numeric_hint = bool(measure_numeric_signal >= 0.8 and uniq >= 0.98 and unique_count > max_categorical_cardinality)

        top_levels = [str(x).strip().lower() for x in row.get("top_levels", []) if x is not None]
        likert_pattern_detected = bool(any(any(tok in lvl for tok in likert_tokens) for lvl in top_levels))
        likert_level_count_hint = bool(2 <= unique_count <= 7)

        measure_score = 0.15
        measure_score += 0.55 * measure_numeric_signal
        if family_member_hint:
            measure_score += 0.25
        if likert_pattern_detected:
            measure_score += 0.25
        if likert_level_count_hint and coded_cat_hint:
            measure_score += 0.20
        if uniq > 0.98:
            measure_score -= 0.30

        anchors = semantic_anchor_map.get(row["column"], [])
        anchor_names = {a.get("anchor") for a in anchors}
        semantic_non_measure = bool(anchor_names & {"US_ZIP_CODE", "ISO_COUNTRY", "STATE_CODE", "EMAIL", "URL", "PHONE", "ICD10_CODE"})
        if semantic_non_measure:
            measure_score *= 0.15
        measure_score = round(float(_clamp(measure_score, 0.0, 1.0)), 6)

        if row.get("is_one_hot_like"):
            encoding_type = "one_hot"
        elif likert_pattern_detected or (likert_level_count_hint and top_type in {"categorical", "text", "mixed"}):
            encoding_type = "ordinal"
        elif top_type in {"numeric", "numeric_with_unit", "percent", "numeric_range"}:
            encoding_type = "numeric"
        elif top_type in {"categorical", "categorical_multi", "text", "mixed"}:
            encoding_type = "nominal"
        else:
            encoding_type = "unknown"

        role_candidates = [
            {
                "role": "id_key",
                "score": round(min(1.0, uniq * (0.7 + 0.3 * top_conf)), 6),
                "evidence": {"unique_ratio": round(uniq, 6), "top_candidate_confidence": round(top_conf, 6)},
            },
            {
                "role": "time_index",
                "score": round(min(1.0, max(parsed_as_datetime_pct / 100.0, 0.8 if any(x in row["column"].lower() for x in ["date", "time", "timestamp", "year", "month", "day"]) else 0.0)), 6),
                "evidence": {"parsed_as_datetime_pct": round(parsed_as_datetime_pct, 6), "name_hint": any(x in row["column"].lower() for x in ["date", "time", "timestamp", "year", "month", "day"])},
            },
            {
                "role": "repeat_index",
                "score": round(0.82 if repeat_parse else 0.1, 6),
                "evidence": {"repeat_name_parse": repeat_parse},
            },
            {
                "role": "measure",
                "score": measure_score,
                "evidence": {
                    "parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6),
                    "is_family_member": family_member_hint,
                    "likert_pattern_detected": likert_pattern_detected,
                    "likert_level_count_hint": likert_level_count_hint,
                    "encoding_type": encoding_type,
                },
            },
            {
                "role": "measure_numeric",
                "score": round(min(1.0, measure_numeric_signal * (0.8 if not identifier_numeric_hint else 0.4)), 6),
                "evidence": {"parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6), "top_candidate_type": top_type, "identifier_numeric_hint": identifier_numeric_hint},
            },
            {
                "role": "measure_item",
                "score": round(0.9 if ((likert_pattern_detected or likert_level_count_hint) and family_member_hint) else (0.8 if (likert_pattern_detected or likert_level_count_hint) else (0.65 if family_member_hint else 0.2)), 6),
                "evidence": {"parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6), "unique_count": unique_count, "likert_pattern_detected": likert_pattern_detected, "family_member_hint": family_member_hint},
            },
            {
                "role": "coded_categorical",
                "score": round(0.8 if coded_cat_hint else 0.2, 6),
                "evidence": {"parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6), "unique_count": unique_count, "max_categorical_cardinality": int(max_categorical_cardinality), "encoding_type": encoding_type},
            },
            {
                "role": "invariant_attr",
                "score": round(min(1.0, max(0.0, (1.0 - uniq) + (0.6 if semantic_non_measure else 0.0))), 6),
                "evidence": {"unique_ratio": round(uniq, 6), "semantic_anchor_boost": semantic_non_measure},
            },
            {
                "role": "derived",
                "score": round(0.7 if any(x in row["column"].lower() for x in ["total", "sum", "score", "avg", "mean"]) else 0.1, 6),
                "evidence": {"name_hint": any(x in row["column"].lower() for x in ["total", "sum", "score", "avg", "mean"])},
            },
            {
                "role": "multivalue_cell",
                "score": round(0.8 if (cols_profile.get(row["column"], {}).get("patterns", {}).get("multi_value", {}).get("multi_token_pct", 0) > 20 and not re.search(r"url|link", row["column"], re.IGNORECASE)) else 0.1, 6),
                "evidence": {"multi_token_pct": cols_profile.get(row["column"], {}).get("patterns", {}).get("multi_value", {}).get("multi_token_pct", 0.0)},
            },
            {
                "role": "one_hot_member",
                "score": 0.8 if row["is_one_hot_like"] else 0.1,
                "evidence": {"is_one_hot_like": bool(row["is_one_hot_like"])},
            },
        ]
        role_candidates = sorted(role_candidates, key=lambda r: (-r["score"], r["role"]))
        role_decision = role_candidates[0]["role"] if role_candidates else "ignore"
        primary_role = "measure" if measure_score >= 0.6 else role_decision
        review_reasons = []
        if top_conf < 0.75:
            review_reasons.append("low_top_candidate_confidence")
        if parsed_as_numeric_pct >= 80.0 and top_type in {"text", "mixed"}:
            review_reasons.append("numeric_parse_vs_candidate_mismatch")
        if parsed_as_datetime_pct >= 80.0 and top_type not in {"date", "datetime"}:
            review_reasons.append("datetime_parse_vs_candidate_mismatch")

        scores = {c["role"]: c["score"] for c in role_candidates}
        role_scores["columns"].append({
            "column": row["column"],
            "primary_role": primary_role,
            "encoding_type": encoding_type,
            "role_scores": scores,
            "role_candidates": role_candidates,
            "role_decision": role_decision,
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "audit": {
                "top_candidate_type": top_type,
                "top_candidate_confidence": round(top_conf, 6),
                "parsed_as_datetime_pct": round(parsed_as_datetime_pct, 6),
                "parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6),
                "sample_values": sample_values,
            },
            "evidence_used": {
                "unique_ratio": uniq,
                "unique_count": unique_count,
                "top_candidate": top_candidate,
                "repeat_name_parse": repeat_parse,
                "family_member_hint": family_member_hint,
                "likert_pattern_detected": likert_pattern_detected,
                "likert_level_count_hint": likert_level_count_hint,
                "semantic_anchors": anchors,
                "family_repeat_dimensions": sorted([x for x in family_repeat_names if x]),
                "missing_pct": row.get("missing_pct", 0.0),
                "numeric_profile_parseable_pct": parsed_as_numeric_pct,
                "datetime_profile_parseable_pct": parsed_as_datetime_pct,
            },
            "evidence_not_used": [],
            "features": {"unique_ratio": uniq, "top_candidate_type": top_type, "encoding_type": encoding_type},
        })
    # Reuse existing deterministic association/dependency engine from /evidence_associations
    raw_signals = association_result.get("signals", [])
    parsed_signals = []
    for s in raw_signals:
        ev = s.get("evidence_json")
        if isinstance(ev, str):
            try:
                ev_obj = json.loads(ev)
            except Exception:
                ev_obj = {"raw": ev}
        else:
            ev_obj = ev if isinstance(ev, dict) else {}
        parsed_signals.append({
            "kind": s.get("kind"),
            "columns": s.get("columns", []),
            "score": s.get("score"),
            "metric": s.get("metric"),
            "evidence": ev_obj,
        })

    near_duplicate_candidates = []
    dependency_candidates = []
    derived_total_candidates = []
    time_column_candidates = []
    family_screening_correlations = []
    included_audit_signals = []
    suppressed_count = 0

    for s in parsed_signals:
        kind = str(s.get("kind") or "")
        score = float(s.get("score", 0.0) or 0.0)
        cols = s.get("columns") or []
        ev = s.get("evidence") or {}

        if kind == "numeric_numeric_corr":
            abs_r = abs(score)
            overlap = int(ev.get("n_overlap", 0) or 0)
            if abs_r >= 0.98 and overlap >= ASSOC_MIN_OVERLAP:
                near_duplicate_candidates.append({
                    "a": cols[0] if len(cols) > 0 else None,
                    "b": cols[1] if len(cols) > 1 else None,
                    "confidence": round(abs_r, 6),
                    "evidence": {"abs_pearson_r": round(abs_r, 6), "overlap": overlap},
                    "action": "review_for_duplicate_or_alias",
                })
                included_audit_signals.append(s)
            elif abs_r >= 0.85:
                family_screening_correlations.append({
                    "a": cols[0] if len(cols) > 0 else None,
                    "b": cols[1] if len(cols) > 1 else None,
                    "correlation": round(score, 6),
                    "evidence": {"abs_pearson_r": round(abs_r, 6), "overlap": overlap},
                    "purpose": "family_screening",
                })
                included_audit_signals.append(s)
            else:
                suppressed_count += 1
            continue

        if kind == "dependency_check":
            if s.get("metric") == "best_fitting_aggregate_rule_ok_rate":
                derived_total_candidates.append({
                    "target": ev.get("total_col"),
                    "components": ev.get("item_cols", []),
                    "identity_type": ev.get("best_rule"),
                    "confidence": round(float(ev.get("best_ok_rate", score) or 0.0), 6),
                    "evidence": {
                        "rows_checked": int(ev.get("n", 0) or 0),
                        "rows_matching": int(round(float(ev.get("best_ok_rate", 0.0) or 0.0) * float(ev.get("n", 0) or 0))),
                        "tolerance_abs": ev.get("tol_abs"),
                        "tolerance_rel": ev.get("tol_rel"),
                    },
                })
            else:
                dependency_candidates.append({
                    "determinant": cols[0] if len(cols) > 0 else None,
                    "dependent": cols[1] if len(cols) > 1 else None,
                    "coverage": round(score, 6),
                    "violations": int(ev.get("violations", 0) or 0),
                    "action": "candidate_invariant_attribute",
                    "evidence": ev,
                })
            included_audit_signals.append(s)
            continue

        if kind == "time_col_candidate":
            time_column_candidates.append({
                "column": cols[0] if cols else None,
                "confidence": round(score, 6),
                "evidence": {
                    "datetime_parse_rate": float(ev.get("parse_success_pct", 0.0) or 0.0),
                    "monotonic_fraction_within_id": float(ev.get("monotonic_increasing_pct", 0.0) or 0.0),
                },
            })
            included_audit_signals.append(s)
            continue

        if kind in {"id_time_duplicates"}:
            dependency_candidates.append({
                "determinant": ev.get("id_col"),
                "dependent": ev.get("time_col"),
                "coverage": round(1.0 - score, 6),
                "violations": int(ev.get("dup_rows", 0) or 0),
                "action": "check_time_grain_duplicates",
                "evidence": ev,
            })
            included_audit_signals.append(s)
            continue

        suppressed_count += 1

    near_duplicate_candidates = sorted(near_duplicate_candidates, key=lambda x: (-x["confidence"], x["a"] or "", x["b"] or ""))[:20]
    family_screening_correlations = sorted(family_screening_correlations, key=lambda x: (-abs(x["correlation"]), x["a"] or "", x["b"] or ""))[:20]
    derived_total_candidates = sorted(derived_total_candidates, key=lambda x: (-x["confidence"], str(x["target"])))[:20]
    dependency_candidates = sorted(dependency_candidates, key=lambda x: (-x["coverage"], str(x["determinant"]), str(x["dependent"])))[:30]
    time_column_candidates = sorted(time_column_candidates, key=lambda x: (-x["confidence"], str(x["column"])))[:20]

    rel = {
        "artifact": "A10",
        "purpose": "structural_relationships_evidence_for_table_grain_and_derived_feature_detection",
        "inputs": artifact_inputs["A10"],
        "limits": association_result.get("limits", {}),
        "n_rows_scanned": association_result.get("n_rows_scanned"),
        "derived_total_candidates": derived_total_candidates,
        "near_duplicate_candidates": near_duplicate_candidates,
        "dependency_candidates": dependency_candidates,
        "time_column_candidates": time_column_candidates,
        "family_screening_correlations": family_screening_correlations,
        "one_hot_blocks": [],
        "coded_categorical_flags": [],
        "audit_trail_signals": {
            "included_count": len(included_audit_signals),
            "suppressed_count": int(max(0, suppressed_count)),
            "suppression_rules": ["generic_pairwise_corr_not_actionable"],
            "included_signals": included_audit_signals,
        },
    }

    one_hot_groups: Dict[str, List[str]] = defaultdict(list)
    for c in df.columns:
        m = re.match(r"^(.*?)[_\-](yes|no|true|false|male|female|other|1|0)$", c, re.IGNORECASE)
        if m:
            one_hot_groups[m.group(1)].append(c)
    for stem, cols in one_hot_groups.items():
        if len(cols) >= 2:
            sub = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            row_sum = sub.sum(axis=1)
            rel["one_hot_blocks"].append({
                "group": stem,
                "columns": cols,
                "exclusive_pct": float((row_sum <= 1).mean() * 100.0),
                "exactly_one_pct": float((row_sum == 1).mean() * 100.0),
            })

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        if df[c].nunique(dropna=True) <= max_categorical_cardinality:
            rel["coded_categorical_flags"].append({"column": c, "reason": "numeric_low_cardinality"})

    rel["deterministic_constraints"] = {
        "arithmetic": [],
        "chronological": [],
        "exclusivity": [],
        "inclusion": [],
    }

    numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:18]
    for target in numeric_candidates:
        for a, b in combinations([x for x in numeric_candidates if x != target][:10], 2):
            lhs_a = pd.to_numeric(df[a], errors="coerce")
            lhs_b = pd.to_numeric(df[b], errors="coerce")
            rhs = pd.to_numeric(df[target], errors="coerce")
            mask = lhs_a.notna() & lhs_b.notna() & rhs.notna()
            if int(mask.sum()) < max(30, int(0.1 * n_rows)):
                continue
            checks = {
                "sum": lhs_a + lhs_b,
                "product": lhs_a * lhs_b,
                "difference": lhs_a - lhs_b,
            }
            if (lhs_b.abs() > ASSOC_TOL_ABS).mean() >= 0.7:
                checks["ratio"] = lhs_a / lhs_b.replace(0, pd.NA)
            for op, calc in checks.items():
                delta = (calc[mask] - rhs[mask]).abs()
                tol = ASSOC_TOL_ABS + ASSOC_TOL_REL * rhs[mask].abs()
                satisfied = int((delta <= tol).sum())
                confidence = float(satisfied / max(1, int(mask.sum())))
                if confidence >= 0.995:
                    rel["deterministic_constraints"]["arithmetic"].append({
                        "formula": f"{target} = {a} { {'sum':'+','product':'*','difference':'-','ratio':'/'}[op] } {b}",
                        "target": target,
                        "inputs": [a, b],
                        "operation": op,
                        "confidence": round(confidence, 6),
                        "rows_evaluated": int(mask.sum()),
                        "violations": int(mask.sum()) - satisfied,
                    })

    datetime_candidates = [c for c in df.columns if float((cols_profile.get(c, {}).get("date_profile") or {}).get("parseable_pct", 0.0) or 0.0) >= 60.0][:12]
    parsed_dt = {c: pd.to_datetime(df[c], errors="coerce", utc=False) for c in datetime_candidates}
    for a, b in combinations(datetime_candidates, 2):
        ma = parsed_dt[a].notna() & parsed_dt[b].notna()
        if int(ma.sum()) < max(30, int(0.1 * n_rows)):
            continue
        ok = int((parsed_dt[b][ma] >= parsed_dt[a][ma]).sum())
        conf = float(ok / max(1, int(ma.sum())))
        if conf >= 0.95:
            rel["deterministic_constraints"]["chronological"].append({
                "constraint": f"{b} >= {a}",
                "start_column": a,
                "end_column": b,
                "confidence": round(conf, 6),
                "rows_evaluated": int(ma.sum()),
                "violations": int(ma.sum()) - ok,
            })

    text_cols = [c for c in df.columns if c not in numeric_candidates][:20]
    yes_tokens = {"yes", "y", "true", "1"}
    for a in text_cols:
        sa = df[a].astype("string").str.strip().str.lower()
        mask_yes = sa.isin(yes_tokens)
        if int(mask_yes.sum()) < max(20, int(0.05 * n_rows)):
            continue
        for b in [c for c in text_cols if c != a][:12]:
            sb = df[b]
            null_ok = int(sb[mask_yes].isna().sum())
            conf = float(null_ok / max(1, int(mask_yes.sum())))
            if conf >= 0.95:
                rel["deterministic_constraints"]["exclusivity"].append({
                    "constraint": f"if {a} is yes then {b} is null",
                    "if_column": a,
                    "then_null_column": b,
                    "confidence": round(conf, 6),
                    "rows_evaluated": int(mask_yes.sum()),
                    "violations": int(mask_yes.sum()) - null_ok,
                })

    for block in rel.get("one_hot_blocks", []):
        rel["deterministic_constraints"]["inclusion"].append({
            "constraint": "one_hot_exclusive",
            "columns": block.get("columns", []),
            "confidence": round(float(block.get("exclusive_pct", 0.0) or 0.0) / 100.0, 6),
            "evidence": {
                "exclusive_pct": block.get("exclusive_pct", 0.0),
                "exactly_one_pct": block.get("exactly_one_pct", 0.0),
            },
        })

    rel["deterministic_constraints"]["arithmetic"] = sorted(
        rel["deterministic_constraints"]["arithmetic"],
        key=lambda x: (-x.get("confidence", 0.0), x.get("violations", 0), str(x.get("formula", ""))),
    )[:40]
    rel["deterministic_constraints"]["chronological"] = sorted(
        rel["deterministic_constraints"]["chronological"],
        key=lambda x: (-x.get("confidence", 0.0), x.get("violations", 0), str(x.get("constraint", ""))),
    )[:30]
    rel["deterministic_constraints"]["exclusivity"] = sorted(
        rel["deterministic_constraints"]["exclusivity"],
        key=lambda x: (-x.get("confidence", 0.0), x.get("violations", 0), str(x.get("constraint", ""))),
    )[:30]
    role_rows = role_scores.get("columns", [])
    role_by_col = {r.get("column"): r for r in role_rows}
    key_seed_map = {k.get("column"): k for k in key_integrity.get("single_column_key_seed_candidates", [])}
    relationship_time_cols = [x.get("column") for x in rel.get("time_column_candidates", []) if x.get("column")]

    def _is_low_signal_col(c: str) -> bool:
        s = df[c].astype("string")
        uniq_ratio = float(s.nunique(dropna=True) / max(1, len(df)))
        long_token_ratio = float(s.str.len().fillna(0).gt(48).mean())
        url_ratio = float(s.str.contains(r"^https?://", na=False, regex=True).mean())
        name_risky = bool(re.search(r"url|uri|link|image|img|photo|avatar|file|path|hash|token", c, re.IGNORECASE))
        return bool(name_risky or url_ratio > 0.25 or (uniq_ratio > 0.95 and long_token_ratio > 0.6))

    def _sample_rows_for_panel(cols: List[str], panel_id: str, *, head_n: int = 5, tail_n: int = 5, rand_n: int = 5) -> List[Dict[str, Any]]:
        if not cols:
            return []
        seed = _stable_int_seed(dataset_sha256, panel_id)
        rows: List[Tuple[int, str]] = []
        rows += [(i, "head") for i in list(df.head(min(head_n, len(df))).index)]
        rows += [(i, "tail") for i in list(df.tail(min(tail_n, len(df))).index)]
        if len(df) > 0:
            rnd_idx = df.sample(n=min(rand_n, len(df)), random_state=seed).index.tolist()
            rows += [(i, "random") for i in rnd_idx]
        seen = set()
        out = []
        for idx, bucket in rows:
            if idx in seen:
                continue
            seen.add(idx)
            rec = {"_row_index": int(idx), "_sample_bucket": bucket}
            for c in cols:
                v = df.at[idx, c]
                rec[c] = None if pd.isna(v) else v
            out.append(rec)
        return out

    panel_specs: List[Dict[str, Any]] = []

    key_cols = [c for c in key_candidate_cols if c in df.columns][:3]
    time_cols = [c for c in list(dict.fromkeys(relationship_time_cols + time_like)) if c in df.columns][:3]
    repeat_cols = [c for c in df.columns if (role_by_col.get(c, {}).get("role_scores", {}).get("repeat_index", 0.0) >= 0.5)][:3]
    grain_cols = list(dict.fromkeys(key_cols + time_cols + repeat_cols))
    grain_uncertainty = bool(any((role_by_col.get(c, {}).get("review_required") for c in grain_cols)) or len(grain_cols) < 2)
    if grain_uncertainty:
        extra = [c for c in df.columns if c not in grain_cols and not _is_low_signal_col(c)][:4]
        grain_cols = list(dict.fromkeys(grain_cols + extra))[:10]
    else:
        grain_cols = grain_cols[:6]
    panel_specs.append({
        "panel_id": "grain_preview",
        "purpose": "Inspect candidate row grain and repeat dimensions",
        "columns": grain_cols,
        "selection_trace": [
            {
                "column": c,
                "reason": "top key candidate" if c in key_cols else ("time-like candidate" if c in time_cols else "repeat index candidate"),
                "source_artifact": "A5" if c in key_cols else "A9",
                "rank": (key_cols.index(c) + 1) if c in key_cols else None,
                "confidence": role_by_col.get(c, {}).get("role_scores", {}).get("time_index") if c in time_cols else role_by_col.get(c, {}).get("role_scores", {}).get("repeat_index"),
                "fallback_rule": "expanded_due_to_low_confidence" if (grain_uncertainty and c not in key_cols + time_cols + repeat_cols) else None,
            }
            for c in grain_cols
        ],
    })

    attr_candidates = [c for c in df.columns if role_by_col.get(c, {}).get("role_scores", {}).get("invariant_attr", 0.0) >= 0.5 and c not in grain_cols and not _is_low_signal_col(c)]
    entity_cols = list(dict.fromkeys(key_cols[:1] + attr_candidates[:7]))
    panel_specs.append({
        "panel_id": "entity_attributes_preview",
        "purpose": "Likely invariant descriptors per entity",
        "columns": entity_cols,
        "selection_trace": [
            {
                "column": c,
                "reason": "entity key anchor" if c in key_cols[:1] else "high invariant_attr role score",
                "source_artifact": "A5" if c in key_cols[:1] else "A9",
                "confidence": role_by_col.get(c, {}).get("role_scores", {}).get("invariant_attr"),
                "rank": key_cols.index(c) + 1 if c in key_cols else None,
                "fallback_rule": None,
            }
            for c in entity_cols
        ],
    })

    measure_candidates = [c for c in df.columns if max(role_by_col.get(c, {}).get("role_scores", {}).get("measure", 0.0), role_by_col.get(c, {}).get("role_scores", {}).get("measure_numeric", 0.0), role_by_col.get(c, {}).get("role_scores", {}).get("measure_item", 0.0)) >= 0.5 and c not in grain_cols and not _is_low_signal_col(c)]
    measure_cols = list(dict.fromkeys(key_cols[:1] + measure_candidates[:8]))
    panel_specs.append({
        "panel_id": "measure_preview",
        "purpose": "Inspect likely numeric/scale/question variables",
        "columns": measure_cols,
        "selection_trace": [
            {
                "column": c,
                "reason": "measure candidate",
                "source_artifact": "A9",
                "confidence": max(role_by_col.get(c, {}).get("role_scores", {}).get("measure", 0.0), role_by_col.get(c, {}).get("role_scores", {}).get("measure_numeric", 0.0), role_by_col.get(c, {}).get("role_scores", {}).get("measure_item", 0.0)) if c not in key_cols[:1] else None,
                "rank": None,
                "fallback_rule": None,
            }
            for c in measure_cols
        ],
    })

    text_candidates = [c for c in df.columns if role_by_col.get(c, {}).get("role_scores", {}).get("coded_categorical", 0.0) >= 0.5 and c not in grain_cols and c not in measure_cols and not _is_low_signal_col(c)]
    text_cols = text_candidates[:8]
    panel_specs.append({
        "panel_id": "text_code_preview",
        "purpose": "Inspect categorical labels/free-text/code fields",
        "columns": text_cols,
        "selection_trace": [
            {
                "column": c,
                "reason": "coded categorical / text role",
                "source_artifact": "A9",
                "confidence": role_by_col.get(c, {}).get("role_scores", {}).get("coded_categorical"),
                "rank": None,
                "fallback_rule": None,
            }
            for c in text_cols
        ],
    })

    family_panels = []
    family_anchor_cols = list(dict.fromkeys(key_cols[:2] + time_cols[:2]))[:2]
    for fam in families:
        fam_cols = [c for c in fam.get("columns", []) if c in df.columns and not _is_low_signal_col(c)]
        members = fam_cols[:8]
        anchors = [c for c in family_anchor_cols if c not in members]
        cols = list(dict.fromkeys(anchors + members))
        family_panels.append({
            "panel_id": f"family_{fam.get('family_id')}_preview",
            "purpose": "Question block / repeated measures preview",
            "columns": cols,
            "columns_structured": {
                "anchors": anchors,
                "members": members,
            },
            "selection_trace": [
                *[
                    {
                        "column": c,
                        "reason": "entity_anchor",
                        "source_artifact": "A5" if c in key_cols else "A10",
                        "confidence": role_by_col.get(c, {}).get("role_scores", {}).get("id_key") if c in key_cols else role_by_col.get(c, {}).get("role_scores", {}).get("time_index"),
                        "rank": (key_cols.index(c) + 1) if c in key_cols else None,
                        "fallback_rule": None,
                    }
                    for c in anchors
                ],
                *[
                    {
                        "column": c,
                        "reason": "group_member",
                        "source_artifact": "A8",
                        "confidence": float(fam.get("detection_confidence", 0.0) or 0.0),
                        "rank": (members.index(c) + 1),
                        "fallback_rule": None,
                    }
                    for c in members
                ],
            ],
            "family_reference": fam,
        })

    if not family_panels:
        stems: Dict[str, List[str]] = defaultdict(list)
        for c in df.columns:
            m = re.match(r"^(.*?)(?:[_\-]?)(\d+)$", c)
            if m:
                stems[m.group(1)].append(c)
        pseudo = sorted([(k, v) for k, v in stems.items() if len(v) >= 3], key=lambda kv: (-len(kv[1]), kv[0]))[:2]
        for stem, cols_raw in pseudo:
            anchors = key_cols[:1]
            members = sorted(cols_raw)[:8]
            cols = list(dict.fromkeys(anchors + members))
            family_panels.append({
                "panel_id": f"pseudo_family_{stem}_preview",
                "purpose": "Fallback patterned column cluster when family detection is weak",
                "columns": cols,
                "columns_structured": {
                    "anchors": anchors,
                    "members": members,
                },
                "selection_trace": [
                    *[
                        {
                            "column": c,
                            "reason": "entity_anchor",
                            "source_artifact": "A5",
                            "confidence": role_by_col.get(c, {}).get("role_scores", {}).get("id_key"),
                            "rank": (key_cols.index(c) + 1) if c in key_cols else None,
                            "fallback_rule": None,
                        }
                        for c in anchors
                    ],
                    *[
                        {
                            "column": c,
                            "reason": "group_member",
                            "source_artifact": "A11_fallback",
                            "confidence": None,
                            "rank": (members.index(c) + 1),
                            "fallback_rule": "no_family_detected_pattern_cluster",
                        }
                        for c in members
                    ],
                ],
                "family_reference": {"family_id": f"pseudo_{stem}", "columns": sorted(cols_raw)},
            })
    panels = []
    for p in panel_specs + family_panels:
        cols = [c for c in p.get("columns", []) if c in df.columns][:10]
        if not cols:
            continue
        rows = _sample_rows_for_panel(
            cols,
            p["panel_id"],
            head_n=6 if (p["panel_id"] == "grain_preview" and grain_uncertainty) else 5,
            tail_n=6 if (p["panel_id"] == "grain_preview" and grain_uncertainty) else 5,
            rand_n=8 if (p["panel_id"] == "grain_preview" and grain_uncertainty) else 5,
        )
        panels.append({
            "panel_id": p["panel_id"],
            "purpose": p["purpose"],
            "columns": cols,
            "columns_structured": p.get("columns_structured"),
            "selection_trace": p["selection_trace"],
            "rows": rows,
            "family_reference": p.get("family_reference"),
            "family_ref": (p.get("family_reference") or {}).get("family_id"),
        })

    glimpses = {
        "artifact": "A11",
        "version": "2",
        "intent": "human_readable_preview_with_audit_trace",
        "inputs": artifact_inputs["A11"],
        "row_samples": {
            "sampling_policy": {
                "head_n": 5,
                "tail_n": 5,
                "random_n": 5,
                "seed": "dataset_sha256-derived",
                "seed_basis": dataset_sha256,
            },
            "selection_policy": {
                "panel_strategy": ["ID/GRAIN", "Entity attributes", "Measure", "Text/code", "Family/pseudo-family"],
                "demotion_rules": ["url_or_path_like", "near_unique_long_metadata_strings"],
                "adaptive_expansion": {
                    "grain_panel_expands_when_confidence_low": True,
                    "pseudo_family_fallback_when_no_families": True,
                },
                "source_artifacts": ["A5", "A8", "A9", "A10"],
            },
            "panels": panels,
        },
    }

    per_family = []
    for p in panels:
        if p.get("family_reference"):
            per_family.append({
                "family_id": p["family_reference"].get("family_id"),
                "columns": p["family_reference"].get("columns", []),
                "rows": p.get("rows", [])[:5],
            })

    family_packets = []
    col_dict_map = {r["column"]: r for r in column_dictionary_rows}
    heatmap_by_col = {r.get("column"): r for r in quality_heatmap.get("columns", [])}

    def _family_token_signature(cols: List[str]) -> set:
        vals = set()
        for c in cols[:4]:
            top_levels = (col_dict_map.get(c, {}) or {}).get("top_levels", [])
            vals.update([str(x).strip().lower() for x in top_levels if x is not None])
        return vals

    family_peer_signatures: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fam in families:
        fam_id = str(fam.get("family_id"))
        fam_cols = [c for c in fam.get("columns", []) if c in df.columns]
        fam_tokens = _family_token_signature(fam_cols)
        fam_pattern = ",".join(fam.get("patterns", []))
        fam_roles = [role_by_col.get(c, {}).get("primary_role") for c in fam_cols if c in role_by_col]
        fam_measure_share = float(sum(1 for r in fam_roles if r in {"measure", "measure_item", "measure_numeric"}) / max(1, len(fam_roles)))

        for other in families:
            other_id = str(other.get("family_id"))
            if other_id == fam_id:
                continue
            other_cols = [c for c in other.get("columns", []) if c in df.columns]
            other_tokens = _family_token_signature(other_cols)
            overlap = len(fam_tokens & other_tokens) / max(1, len(fam_tokens | other_tokens)) if (fam_tokens or other_tokens) else 0.0
            shared_features: List[str] = []
            score = 0.0

            if fam_pattern and fam_pattern == ",".join(other.get("patterns", [])):
                shared_features.append(f"MATCHING_SUFFIX_PATTERN:{fam_pattern}")
                score += 0.25
            if overlap >= 0.6:
                shared_features.append(f"IDENTICAL_TOKEN_SET_OVERLAP:{round(overlap, 4)}")
                score += 0.35

            other_roles = [role_by_col.get(c, {}).get("primary_role") for c in other_cols if c in role_by_col]
            other_measure_share = float(sum(1 for r in other_roles if r in {"measure", "measure_item", "measure_numeric"}) / max(1, len(other_roles)))
            if abs(fam_measure_share - other_measure_share) <= 0.25:
                shared_features.append("SAME_A9_ROLE_SIGNATURE:instrument_item_like")
                score += 0.20

            corr_hit = next((x for x in rel.get("family_screening_correlations", []) if (x.get("a") in fam_cols and x.get("b") in other_cols) or (x.get("b") in fam_cols and x.get("a") in other_cols)), None)
            if corr_hit and abs(float(corr_hit.get("correlation", 0.0) or 0.0)) >= 0.4:
                shared_features.append(f"CORRELATED_MEANS:r={round(float(corr_hit.get('correlation', 0.0) or 0.0), 4)}")
                score += 0.20

            if score >= 0.5:
                family_peer_signatures[fam_id].append({
                    "peer_id": other_id,
                    "confidence": round(min(0.99, score), 6),
                    "shared_features": shared_features,
                })

    for fam in families:
        fam_cols = fam["columns"]
        index_pattern = fam.get("index_pattern") or {}
        present_cols = [c for c in fam_cols if c in df.columns]
        fam_missing = [float(df[c].isna().mean() * 100.0) for c in present_cols]
        common_levels = None
        if present_cols:
            level_sets = []
            for c in present_cols:
                vals = set(df[c].dropna().astype("string").str.strip().str.lower().unique().tolist())
                if len(vals) <= max_categorical_cardinality:
                    level_sets.append(vals)
            if level_sets:
                first = level_sets[0]
                common_levels = all(v == first for v in level_sets[1:])
        idx_tokens = fam.get("extracted_index_set", [])
        numeric_idx = [int(x) for x in idx_tokens if re.fullmatch(r"\d+", str(x))]
        dense = False
        if numeric_idx:
            expected = set(range(min(numeric_idx), max(numeric_idx) + 1))
            dense = set(numeric_idx) == expected

        family_packets.append({
            "inputs": artifact_inputs["B1"],
            "family_id": fam["family_id"],
            "columns": fam_cols,
            "detected_pattern_index_summary": {
                "patterns": fam["patterns"],
                "index": fam.get("extracted_index_set", []),
                "index_type_candidate": fam.get("index_type_candidate") or index_pattern.get("index_type_candidate"),
            },
            "family_summary": {
                "index_is_dense": bool(dense) if numeric_idx else None,
                "index_is_ordinal": bool((index_pattern.get("index_type_candidate") == "ordinal") if index_pattern else False),
                "consistent_levels_across_rows": common_levels,
                "avg_missing_pct": round(sum(fam_missing) / max(1, len(fam_missing)), 6) if fam_missing else None,
                "row_variance_missing_pct": round(float(pd.Series(fam_missing).var(ddof=0)) if fam_missing else 0.0, 6) if fam_missing else None,
                "row_specific_anomalies": [c for c in present_cols if float(df[c].isna().mean() * 100.0) > ((sum(fam_missing) / max(1, len(fam_missing))) + 20.0)] if fam_missing else [],
            },
            "relational_context": {
                "global_grain": (best_test.get("keys_tested", []) if best_test else []),
                "peer_families": [x.get("peer_id") for x in sorted(family_peer_signatures.get(fam["family_id"], []), key=lambda z: -float(z.get("confidence", 0.0)) )[:3]],
                "shared_indices": fam.get("extracted_index_set", [])[:10],
            },
            "peer_signature": sorted(family_peer_signatures.get(fam["family_id"], []), key=lambda z: -float(z.get("confidence", 0.0)) )[:3],
            "evidence_subset": {
                "A2_signals": [
                    {
                        "column": c,
                        "top_candidate_type": (col_dict_map.get(c, {}).get("top_candidate") or {}).get("type"),
                        "top_levels": (col_dict_map.get(c, {}).get("top_levels") or [])[:6],
                        "missing_pct": col_dict_map.get(c, {}).get("missing_pct"),
                        "semantic_anchors": semantic_anchor_map.get(c, []),
                    }
                    for c in fam_cols[:8]
                ],
                "A14_drift_summary": {
                    "status": "Stable" if all(not (heatmap_by_col.get(c, {}) or {}).get("drift_detected", False) for c in fam_cols[:8]) else "Drift_detected",
                    "columns": [
                        {
                            "column": c,
                            "drift_detected": bool((heatmap_by_col.get(c, {}) or {}).get("drift_detected", False)),
                            "global_quality_score": (heatmap_by_col.get(c, {}) or {}).get("global_quality_score"),
                        }
                        for c in fam_cols[:8]
                    ],
                },
            },
            "B_subset": [col_dict_map[c] for c in fam_cols if c in col_dict_map],
            "C_subset": {
                "repeat_candidate": fam,
                "grain_collisions_touching_family": [g for g in grain_tests if any(c in fam_cols for c in g["keys_tested"])],
            },
            "F_subset": next((pf for pf in per_family if pf["family_id"] == fam["family_id"]), {"rows": []}),
        })

    table_layout_candidates = _build_table_layout_candidates(
        key_integrity=key_integrity,
        grain_tests=grain_tests,
        repeat_candidates=repeat_candidates,
        role_scores=role_scores,
        relationships=rel,
        all_columns=list(df.columns),
    )
    table_layout_candidates["artifact"] = "A12"
    table_layout_candidates["purpose"] = "table_layout_candidates"
    table_layout_candidates["inputs"] = artifact_inputs["A12"]
    table_layout_candidates["evidence_primitives"] = {
        "source": "column_signal_map",
        "fields": ["unique_ratio", "top_candidate_type", "top_candidate_confidence", "parsed_as_numeric_pct", "parsed_as_datetime_pct"],
    }

    baseline_column_resolution = _build_baseline_column_resolution_artifact(
        a2_rows=column_dictionary_rows,
        a3t_payload=transform_review_queue,
        a3v_payload=variable_type_review_queue,
        a4_payload=missing_catalog,
        a9_payload=role_scores,
        a13_payload=semantic_anchors,
        a14_payload=quality_heatmap,
        a16_payload=conditional_missingness,
        artifact_inputs=artifact_inputs,
    )

    run_id = uuid4().hex
    base_url = str(request.base_url).rstrip("/")
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    payloads: Dict[str, bytes] = {
        "A2": _jsonl_bytes(column_dictionary_rows),
        "A3-T": _json_bytes(transform_review_queue),
        "A3-V": _json_bytes(variable_type_review_queue),
        "A4": _json_bytes(missing_catalog),
        "A5": _json_bytes(key_integrity),
        "A6": _json_bytes(grain_validation),
        "A7": _json_bytes(duplicates_report),
        "A8": _json_bytes(repeat_candidates),
        "A9": _json_bytes(role_scores),
        "A10": _json_bytes(rel),
        "A11": _json_bytes(glimpses),
        "A12": _json_bytes(table_layout_candidates),
        "A13": _json_bytes(semantic_anchors),
        "A14": _json_bytes(quality_heatmap),
        "A16": _json_bytes(conditional_missingness),
        "A17": _json_bytes(baseline_column_resolution),
        "B1": _jsonl_bytes(family_packets),
    }

    uploaded_meta: Dict[str, Dict[str, Any]] = {}
    for aid, payload in payloads.items():
        spec = ARTIFACT_SPECS[aid]
        object_path = f"runs/{run_id}/{spec['filename']}"
        uploaded_meta[aid] = _upload_artifact(bucket, object_path, payload, spec["content_type"])

    manifest_artifacts = []
    for aid, meta in sorted(uploaded_meta.items(), key=lambda kv: kv[0]):
        manifest_artifacts.append({
            "artifact_id": aid,
            "filename": ARTIFACT_SPECS[aid]["filename"],
            "object_path": meta["object_path"],
            "bucket": bucket_name,
            "content_type": meta["content_type"],
            "sha256": meta["sha256"],
            "size_bytes": meta["size_bytes"],
            "download_url": _build_artifact_url(base_url, aid, run_id, "download"),
            "meta_url": _build_artifact_url(base_url, aid, run_id, "meta"),
            "view_get_url": _build_artifact_view_get_url(base_url, aid, run_id, mode="llm_baseline"),
            "view_post_url": _build_artifact_view_post_url(base_url, aid),
        })

    run_manifest = {
        "artifact": "A1",
        "purpose": "run_manifest",
        "inputs": artifact_inputs["A1"],
        "evidence_primitives": {
            "source": "column_signal_map",
            "fields": [
                "unique_ratio",
                "missing_pct",
                "parsed_as_numeric_pct",
                "parsed_as_datetime_pct",
                "top_candidate_type",
                "top_candidate_confidence",
            ],
        },
        "schema_version": "a1.v1",
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "profiling_limits": {
            "max_categorical_cardinality": int(max_categorical_cardinality),
            "ASSOC_MAX_ROWS": ASSOC_MAX_ROWS,
            "ASSOC_TOP_K_PAIRS": ASSOC_TOP_K_PAIRS,
            "ASSOC_MAX_NUMERIC_COLS": ASSOC_MAX_NUMERIC_COLS,
            "ASSOC_MAX_CAT_COLS": ASSOC_MAX_CAT_COLS,
            "ASSOC_MAX_CAT_CARD": ASSOC_MAX_CAT_CARD,
            "ASSOC_MIN_OVERLAP": ASSOC_MIN_OVERLAP,
            "ASSOC_TOL_ABS": ASSOC_TOL_ABS,
            "ASSOC_TOL_REL": ASSOC_TOL_REL,
        },
        "artifact_registry": manifest_artifacts,
    }

    manifest_object_path = f"runs/{run_id}/manifest.json"
    manifest_payload = _json_bytes(run_manifest)
    manifest_meta = _upload_artifact(bucket, manifest_object_path, manifest_payload, "application/json")

    # Add A1 record into manifest and persist once more under strict single schema
    a1_entry = {
        "artifact_id": "A1",
        "filename": ARTIFACT_SPECS["A1"]["filename"],
        "object_path": manifest_object_path,
        "bucket": bucket_name,
        "content_type": manifest_meta["content_type"],
        "sha256": manifest_meta["sha256"],
        "size_bytes": manifest_meta["size_bytes"],
        "download_url": _build_artifact_url(base_url, "A1", run_id, "download"),
        "meta_url": _build_artifact_url(base_url, "A1", run_id, "meta"),
        "view_get_url": _build_artifact_view_get_url(base_url, "A1", run_id, mode="llm_baseline"),
        "view_post_url": _build_artifact_view_post_url(base_url, "A1"),
    }
    run_manifest["artifact_registry"] = [a1_entry] + manifest_artifacts
    _upload_artifact(bucket, manifest_object_path, _json_bytes(run_manifest), "application/json")

    return {
        "status": "success",
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "artifacts_url": f"{base_url}/artifacts?run_id={run_id}",
        "manifest_url": f"{base_url}/artifacts/A1/meta?run_id={run_id}",
    }


@app.get("/artifacts")
def list_artifacts(
    request: Request,
    run_id: str,
    _=Depends(require_token),
) -> Dict[str, Any]:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"runs/{run_id}/manifest.json")
    if not blob.exists(client):
        raise HTTPException(status_code=404, detail="run_id not found")

    manifest = json.loads(blob.download_as_bytes().decode("utf-8"))
    base_url = str(request.base_url).rstrip("/")
    entries = []
    for item in manifest.get("artifact_registry", []):
        entries.append(_enrich_artifact_entry_with_urls(base_url=base_url, run_id=run_id, item=item))

    return {
        "run_id": run_id,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "artifacts": entries,
    }


@app.get("/artifacts/{artifact_id}/meta")
def artifact_meta(
    request: Request,
    artifact_id: str,
    run_id: str,
    _=Depends(require_token),
) -> Dict[str, Any]:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"runs/{run_id}/manifest.json")
    if not blob.exists(client):
        raise HTTPException(status_code=404, detail="run_id not found")

    manifest = json.loads(blob.download_as_bytes().decode("utf-8"))
    base_url = str(request.base_url).rstrip("/")
    for item in manifest.get("artifact_registry", []):
        if item.get("artifact_id") == artifact_id:
            return _enrich_artifact_entry_with_urls(base_url=base_url, run_id=run_id, item=item)
    raise HTTPException(status_code=404, detail="artifact not found")


@app.get("/artifacts/{artifact_id}/download")
def artifact_download(
    artifact_id: str,
    run_id: str,
    request: Request,
):
    _require_api_key_from_request(request)

    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    manifest_blob = bucket.blob(f"runs/{run_id}/manifest.json")
    if not manifest_blob.exists(client):
        raise HTTPException(status_code=404, detail="run_id not found")

    manifest = json.loads(manifest_blob.download_as_bytes().decode("utf-8"))
    found = None
    for item in manifest.get("artifact_registry", []):
        if item.get("artifact_id") == artifact_id:
            found = item
            break
    if not found:
        raise HTTPException(status_code=404, detail="artifact not found")

    data_blob = bucket.blob(found["object_path"])
    if not data_blob.exists(client):
        raise HTTPException(status_code=404, detail="artifact object missing")

    filename = os.path.basename(found["object_path"])
    stream = io.BytesIO(data_blob.download_as_bytes())
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(stream, media_type=found.get("content_type", "application/octet-stream"), headers=headers)


class ArtifactViewRequest(BaseModel):
    run_id: str
    mode: str = "raw"
    keep: Optional[List[str]] = None
    drop: Optional[List[str]] = None
    limits: Optional[Dict[str, int]] = None
    policy_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    replace_policies: Optional[List[str]] = None
    transform_overrides: Optional[Dict[str, Any]] = None
    replace_transforms: Optional[List[str]] = None
    value_filter: Optional[Any] = None
    debug: bool = False


class ArtifactBundleScope(BaseModel):
    mode: Optional[str] = None
    keep: Optional[List[str]] = None
    drop: Optional[List[str]] = None
    limits: Optional[Dict[str, int]] = None
    value_filter: Optional[Any] = None


class ArtifactBundleRequest(BaseModel):
    run_id: str
    mode: str = "raw"
    artifact_ids: Optional[List[str]] = None
    global_scope: ArtifactBundleScope = Field(default_factory=lambda: ArtifactBundleScope(mode="raw"), alias="global")
    per_artifact: Optional[Dict[str, ArtifactBundleScope]] = None
    light_contract_decisions: Optional[Any] = None
    family_worker_json: Optional[Any] = None
    policy_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    replace_policies: Optional[List[str]] = None
    transform_overrides: Optional[Dict[str, Any]] = None
    replace_transforms: Optional[List[str]] = None
    debug: bool = False

    class Config:
        allow_population_by_field_name = True


LLM_PRUNING_LEDGER_PATH = os.getenv(
    "LLM_PRUNING_LEDGER_PATH",
    str((Path(__file__).resolve().parent / "llm_pruning_ledger.json")),
)
PROFILE_PREFIX = os.getenv("PROFILE_PREFIX", "profiles/")
PROFILE_SOURCE = os.getenv("PROFILE_SOURCE", "gcs").strip().lower()
PROFILE_CACHE_TTL = int(os.getenv("PROFILE_CACHE_TTL", "300"))
LOCAL_PROFILES_DIR = Path(os.getenv("LOCAL_PROFILES_DIR", str(Path(__file__).resolve().parent / "profiles")))


def _normalize_path_hint(path_hint: str) -> str:
    normalized = path_hint.replace("panels[]", "panels").replace("[]", "")
    return normalized.strip(".")


def _load_pruning_ledger(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
    except FileNotFoundError:
        logger.warning("Pruning ledger not found at %s; using empty llm_baseline fallback", path)
        ledger = {"schema_version": "fallback", "modes": {"llm_baseline": {}}}
    except Exception as exc:
        raise RuntimeError(f"Failed to load pruning ledger from {path}: {exc}") from exc

    if not isinstance(ledger, dict):
        raise RuntimeError("Invalid pruning ledger: expected top-level object")
    if "schema_version" not in ledger or "modes" not in ledger:
        raise RuntimeError("Invalid pruning ledger: required keys schema_version and modes")
    modes = ledger.get("modes")
    if not isinstance(modes, dict):
        raise RuntimeError("Invalid pruning ledger: modes must be an object")
    if "llm_baseline" not in modes:        raise RuntimeError("Invalid pruning ledger: llm_baseline mode is required")
    return ledger


def _adapt_pruning_modes_from_ledger(ledger: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_mode = {
        "tier1": {
            "global_drop_keys": [],
            "global_drop_rules": [],
        },
        "tier2": {
            "artifact_drop_keys": {},
        },
        "tier3": {
            "limits": {},
            "policies": {},
            "transforms": {},
        },
    }
    adapted: Dict[str, Dict[str, Any]] = {"raw": raw_mode}

    for mode_name, mode_cfg in (ledger.get("modes") or {}).items():
        if not isinstance(mode_cfg, dict):
            continue
        global_drops = mode_cfg.get("global_drops") or []
        global_drop_rules = mode_cfg.get("global_drop_rules") or []
        artifact_drops = mode_cfg.get("artifact_drops") or {}
        artifact_limits = mode_cfg.get("artifact_limits") or {}
        artifact_policies = mode_cfg.get("artifact_policies") or {}
        artifact_transforms = mode_cfg.get("artifact_transforms") or {}

        flattened_limits: Dict[str, int] = {}

        if isinstance(artifact_limits, dict):
            for artifact_id, key_limits in artifact_limits.items():
                if not isinstance(key_limits, dict):
                    continue

                for key, value in key_limits.items():
                    try:
                        as_int = int(value)
                    except (ValueError, TypeError):
                        continue

                    if as_int >= 0:
                        flattened_limits[f"{artifact_id}.{key}"] = as_int

        normalized_transforms: Dict[str, Dict[str, Any]] = {}
        if isinstance(artifact_transforms, dict):
            for artifact_id, raw_ops in artifact_transforms.items():
                if isinstance(raw_ops, dict):
                    normalized_transforms[artifact_id] = {
                        str(k): dict(v) for k, v in raw_ops.items() if isinstance(v, dict)
                    }
                    continue
                ops: Dict[str, Dict[str, Any]] = {}
                if isinstance(raw_ops, list):
                    idx = 1
                    for op in raw_ops:
                        if not isinstance(op, dict):
                            continue
                        op_name = str(op.get("name") or f"op_{idx}")
                        op_copy = dict(op)
                        op_copy["name"] = op_name
                        ops[op_name] = op_copy
                        idx += 1
                normalized_transforms[artifact_id] = ops

        adapted[mode_name] = {
            "tier1": {
                "global_drop_keys": list(global_drops),
                "global_drop_rules": list(global_drop_rules),
            },
            "tier2": {
                "artifact_drop_keys": dict(artifact_drops),
            },
            "tier3": {
                "limits": flattened_limits,
                "policies": dict(artifact_policies),
                "transforms": normalized_transforms,
            },
        }
    return adapted


PRUNING_LEDGER = _load_pruning_ledger(LLM_PRUNING_LEDGER_PATH)
PRUNING_MODE_LEDGER: Dict[str, Dict[str, Any]] = _adapt_pruning_modes_from_ledger(PRUNING_LEDGER)
_PROFILE_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_profile_prefix(prefix: str) -> str:
    cleaned = (prefix or "profiles/").strip().lstrip("/")
    if cleaned and not cleaned.endswith("/"):
        cleaned += "/"
    return cleaned


def _profile_object_path(profile_name: str) -> str:
    return f"{_normalize_profile_prefix(PROFILE_PREFIX)}{profile_name}.json"


def _extract_drop_entries(drop_entries: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    drop_keys: List[str] = []
    drop_rules: List[Dict[str, Any]] = []
    if not isinstance(drop_entries, list):
        return drop_keys, drop_rules
    for entry in drop_entries:
        if isinstance(entry, str):
            drop_keys.append(entry)
        elif isinstance(entry, dict) and entry.get("key"):
            drop_rules.append(dict(entry))
    return drop_keys, drop_rules


def _normalize_named_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        raise HTTPException(status_code=422, detail="Profile must be a JSON object")

    missing_required = [k for k in ("profile", "artifacts", "pruning") if k not in profile]
    if missing_required:
        raise HTTPException(status_code=422, detail=f"Profile missing required keys: {', '.join(missing_required)}")

    profile_name = str(profile.get("profile") or "").strip()
    artifacts = profile.get("artifacts")
    pruning = profile.get("pruning")
    if not profile_name:
        raise HTTPException(status_code=422, detail="Profile key 'profile' must be a non-empty string")
    if not isinstance(artifacts, list) or not all(isinstance(a, str) and a for a in artifacts):
        raise HTTPException(status_code=422, detail="Profile key 'artifacts' must be a list of artifact IDs")
    if not isinstance(pruning, dict):
        raise HTTPException(status_code=422, detail="Profile key 'pruning' must be an object")

    artifact_drop_keys: Dict[str, List[str]] = {}
    artifact_drop_rules: Dict[str, List[Dict[str, Any]]] = {}
    flattened_limits: Dict[str, int] = {}
    policies: Dict[str, Dict[str, Any]] = {}
    transforms: Dict[str, Dict[str, Any]] = {}

    for artifact_id, artifact_cfg in pruning.items():
        if not isinstance(artifact_cfg, dict):
            raise HTTPException(status_code=422, detail=f"Profile pruning.{artifact_id} must be an object")

        drop_keys, drop_rules = _extract_drop_entries(artifact_cfg.get("drop") or [])
        artifact_drop_keys[artifact_id] = drop_keys
        artifact_drop_rules[artifact_id] = drop_rules

        for key, value in (artifact_cfg.get("limits") or {}).items():
            try:
                as_int = int(value)
            except (ValueError, TypeError):
                continue
            if as_int >= 0:
                flattened_limits[f"{artifact_id}.{key}"] = as_int

        artifact_policies = artifact_cfg.get("policies") or {}
        if artifact_policies and isinstance(artifact_policies, dict):
            policies[artifact_id] = dict(artifact_policies)

        artifact_transforms = artifact_cfg.get("transforms") or {}
        transforms[artifact_id] = _normalize_transform_ops(artifact_transforms)

    return {
        "profile": profile_name,
        "description": profile.get("description", ""),
        "artifacts": artifacts,
        "mode_config": {
            "tier1": {
                "global_drop_keys": [],
                "global_drop_rules": [],
            },
            "tier2": {
                "artifact_drop_keys": artifact_drop_keys,
                "artifact_drop_rules": artifact_drop_rules,
            },
            "tier3": {
                "limits": flattened_limits,
                "policies": policies,
                "transforms": transforms,
            },
        },
    }


def _load_profile_local(profile_name: str) -> Tuple[Dict[str, Any], str]:
    local_path = LOCAL_PROFILES_DIR / f"{profile_name}.json"
    if not local_path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found at {local_path}")
    try:
        profile_raw = json.loads(local_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Malformed profile JSON at {local_path}: {exc}") from exc
    return _normalize_named_profile(profile_raw), "local"


def _load_profile_gcs(profile_name: str) -> Tuple[Dict[str, Any], str]:
    bucket_name = os.getenv("EXPORT_BUCKET")
    object_path = _profile_object_path(profile_name)
    if not bucket_name:
        raise HTTPException(status_code=503, detail="EXPORT_BUCKET is not configured for GCS profile loading")

    now = time.time()
    cached = _PROFILE_CACHE.get(profile_name)
    if cached and now < cached.get("expires_at", 0):
        return cached["profile"], "cache_hit"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        if not blob.exists(client):
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{profile_name}' not found at gs://{bucket_name}/{object_path}. Have you uploaded it?",
            )
        profile_raw = json.loads(blob.download_as_bytes().decode("utf-8"))
    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Malformed profile JSON at gs://{bucket_name}/{object_path}: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"GCS unavailable while loading profile '{profile_name}': {exc}") from exc

    normalized = _normalize_named_profile(profile_raw)
    _PROFILE_CACHE[profile_name] = {
        "profile": normalized,
        "expires_at": now + max(1, PROFILE_CACHE_TTL),
    }
    return normalized, "cache_miss"


def _resolve_mode_config(mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized_mode = str(mode or "raw").strip() or "raw"
    if normalized_mode == "raw":
        return _resolve_pruning_mode("raw"), {
            "mode": "raw",
            "header": "raw; no_pruning",
            "profile_artifacts": None,
            "cache": "none",
            "source": "none",
        }
    if normalized_mode == "llm_baseline":
        return _resolve_pruning_mode("llm_baseline"), {
            "mode": "llm_baseline",
            "header": "llm_baseline; source=local_ledger",
            "profile_artifacts": None,
            "cache": "none",
            "source": "local_ledger",
        }

    use_local = PROFILE_SOURCE == "local" or not os.getenv("EXPORT_BUCKET")
    if use_local:
        profile, source = _load_profile_local(normalized_mode)
        return profile["mode_config"], {
            "mode": profile["profile"],
            "header": f"{profile['profile']}; source={source}",
            "profile_artifacts": profile.get("artifacts") or [],
            "cache": "n/a",
            "source": source,
        }

    profile, cache_state = _load_profile_gcs(normalized_mode)
    cache_value = "hit" if cache_state == "cache_hit" else "miss"
    return profile["mode_config"], {
        "mode": profile["profile"],
        "header": f"{profile['profile']}; cache={cache_value}",
        "profile_artifacts": profile.get("artifacts") or [],
        "cache": cache_value,
        "source": "gcs",
    }

_DECODE_CACHE: Dict[Tuple[str, str, str], Tuple[str, Any]] = {}
_DECODE_CACHE_MAX = int(os.getenv("ARTIFACT_VIEW_CACHE_MAX", "32"))

MAX_POLICY_OVERRIDE_KEYS = int(os.getenv("MAX_POLICY_OVERRIDE_KEYS", "25"))
MAX_POLICY_OVERRIDE_BYTES = int(os.getenv("MAX_POLICY_OVERRIDE_BYTES", "65536"))
MAX_TRANSFORM_OVERRIDE_KEYS = int(os.getenv("MAX_TRANSFORM_OVERRIDE_KEYS", "25"))
MAX_TRANSFORM_OVERRIDE_BYTES = int(os.getenv("MAX_TRANSFORM_OVERRIDE_BYTES", "65536"))
MAX_CAP = int(os.getenv("MAX_CAP", "500"))
MAX_POLICY_INT = int(os.getenv("MAX_POLICY_INT", "1000"))
MAX_POLICY_LIST_LEN = int(os.getenv("MAX_POLICY_LIST_LEN", "200"))
MAX_POLICY_STR_LEN = int(os.getenv("MAX_POLICY_STR_LEN", "2048"))
MAX_LIMITS_VALUE = int(os.getenv("MAX_LIMITS_VALUE", "5000"))

def parse_csv_list(s: str) -> List[str]:
    if not s:
        return []
    return [part.strip() for part in s.split(",") if part and part.strip()]


def parse_limits_csv(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in parse_csv_list(s):
        if ":" not in token:
            raise HTTPException(status_code=422, detail=f"Invalid limits token: {token}")
        key, raw_value = token.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise HTTPException(status_code=422, detail=f"Invalid limits key in token: {token}")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid limit value in token: {token}") from exc
        if value < 0:
            raise HTTPException(status_code=422, detail=f"Limit must be non-negative: {token}")
        out[key] = value
    return out


def _clamp_limit_value(path: str, value: int, report: List[Dict[str, Any]]) -> int:
    if value < 0:
        report.append({"path": path, "from": value, "to": 0, "reason": "negative_not_allowed"})
        return 0
    if value > MAX_LIMITS_VALUE:
        report.append({"path": path, "from": value, "to": MAX_LIMITS_VALUE, "reason": "max_limits_value"})
        return MAX_LIMITS_VALUE
    return value


def _guard_limits_map(limits: Optional[Dict[str, int]], path_prefix: str, clamps: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not limits:
        return limits
    guarded: Dict[str, int] = {}
    for k, v in limits.items():
        try:
            as_int = int(v)
        except (ValueError, TypeError):
            continue
        guarded[str(k)] = _clamp_limit_value(f"{path_prefix}.{k}", as_int, clamps)
    return guarded


def _parse_policy_key(policy_key: str) -> Tuple[str, str]:
    key = str(policy_key or "").strip()
    if key.startswith("artifact_policies."):
        key = key[len("artifact_policies."):]
    parts = key.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(status_code=422, detail=f"Invalid policy key: {policy_key}")
    return parts[0], parts[1]


def _parse_transform_key(transform_key: str) -> Tuple[str, str]:
    key = str(transform_key or "").strip()
    if key.startswith("artifact_transforms."):
        key = key[len("artifact_transforms."):]
    parts = key.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(status_code=422, detail=f"Invalid transform key: {transform_key}")
    return parts[0], parts[1]


def _normalize_transform_ops(raw_ops: Any) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_ops, dict):
        iterable = raw_ops.items()
    elif isinstance(raw_ops, list):
        iterable = ((None, op) for op in raw_ops)
    else:
        return normalized

    next_idx = 1
    for key, op in iterable:
        if not isinstance(op, dict):
            continue
        op_name = str(key or op.get("name") or f"op_{next_idx}").strip()
        if not op_name:
            op_name = f"op_{next_idx}"
        if op_name in normalized:
            suffix = 2
            while f"{op_name}_{suffix}" in normalized:
                suffix += 1
            op_name = f"{op_name}_{suffix}"
        op_copy = dict(op)
        op_copy["name"] = op_name
        normalized[op_name] = op_copy
        next_idx += 1
    return normalized


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _clamp_policy_number(
    path: str,
    number: int,
    field_name: str,
    clamps: List[Dict[str, Any]],
) -> int:
    if number < 0:
        clamps.append({"path": path, "from": number, "to": 0, "reason": "negative_not_allowed"})
        return 0

    key_like_fields = {"k", "cap", "limit", "max_items", "hard_cap_total", "include_review_required_topk"}
    limit = MAX_CAP if field_name in key_like_fields else MAX_POLICY_INT
    reason = "max_policy_k" if field_name in key_like_fields else "max_policy_int"
    if number > limit:
        clamps.append({"path": path, "from": number, "to": limit, "reason": reason})
        return limit
    return number


def _clamp_structure_sizes(node: Any, path: str, clamps: List[Dict[str, Any]]) -> Any:
    if isinstance(node, dict):
        guarded: Dict[str, Any] = {}
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            guarded[key] = _clamp_structure_sizes(value, child_path, clamps)
        return guarded
    if isinstance(node, list):
        if len(node) > MAX_POLICY_LIST_LEN:
            clamps.append({
                "path": path,
                "from": len(node),
                "to": MAX_POLICY_LIST_LEN,
                "reason": "max_policy_list_len",
            })
            node = node[:MAX_POLICY_LIST_LEN]
        return [_clamp_structure_sizes(item, f"{path}[{idx}]", clamps) for idx, item in enumerate(node)]
    if isinstance(node, str) and len(node) > MAX_POLICY_STR_LEN:
        clamps.append({
            "path": path,
            "from": len(node),
            "to": MAX_POLICY_STR_LEN,
            "reason": "max_policy_str_len",
        })
        return node[:MAX_POLICY_STR_LEN]
    return node


def _apply_policy_numeric_guardrails(node: Any, path: str, clamps: List[Dict[str, Any]]) -> Any:
    sized = _clamp_structure_sizes(node, path, clamps)

    if isinstance(sized, dict):
        guarded: Dict[str, Any] = {}

        for key, value in sized.items():
            child_path = f"{path}.{key}" if path else str(key)

            if isinstance(value, bool):
                guarded[key] = value
            elif isinstance(value, int):
                guarded[key] = _clamp_policy_number(child_path, value, str(key), clamps)
            elif isinstance(value, float) and value.is_integer():
                guarded[key] = _clamp_policy_number(child_path, int(value), str(key), clamps)
            else:
                guarded[key] = _apply_policy_numeric_guardrails(value, child_path, clamps)

        return guarded

    if isinstance(sized, list):
        return [
            _apply_policy_numeric_guardrails(item, f"{path}[{idx}]", clamps)
            for idx, item in enumerate(sized)
        ]

    return sized


def _build_effective_policies(    base_policies: Dict[str, Any],
    policy_overrides: Optional[Dict[str, Dict[str, Any]]],
    replace_policies: Optional[Iterable[str]],
    report: Optional[Dict[str, Any]] = None,
    scope_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    effective = json.loads(json.dumps(base_policies or {}))
    override_payload = policy_overrides or {}
    payload_bytes = len(json.dumps(override_payload, default=_json_default).encode("utf-8"))
    if payload_bytes > MAX_POLICY_OVERRIDE_BYTES:
        raise HTTPException(status_code=422, detail="policy_overrides payload too large")
    if len(override_payload) > MAX_POLICY_OVERRIDE_KEYS:
        raise HTTPException(status_code=422, detail="Too many policy_overrides keys")

    replace_set = set(replace_policies or [])
    clamps: List[Dict[str, Any]] = []
    received: Dict[str, Any] = {}

    for policy_key, override in override_payload.items():
        if not isinstance(override, dict):
            raise HTTPException(status_code=422, detail=f"policy_overrides[{policy_key}] must be an object")
        artifact_id, policy_name = _parse_policy_key(policy_key)
        received[policy_key] = override

        artifact_policies = dict(effective.get(artifact_id) or {})
        existing_policy = artifact_policies.get(policy_name)
        if not isinstance(existing_policy, dict):
            existing_policy = {}

        merged = dict(override) if policy_key in replace_set else _deep_merge_dict(existing_policy, override)
        guarded = _apply_policy_numeric_guardrails(merged, policy_key, clamps)
        artifact_policies[policy_name] = guarded
        effective[artifact_id] = artifact_policies

    if report is not None:
        report["policy_overrides_received"] = received
        scoped = effective
        if scope_artifact_id is not None:
            scoped = {scope_artifact_id: dict((effective.get(scope_artifact_id) or {}))}
        report["policy_effective"] = scoped
        report["policy_clamps"] = clamps

    return effective


def _build_effective_transforms(
    base_transforms: Dict[str, Any],
    transform_overrides: Optional[Dict[str, Any]],
    replace_transforms: Optional[Iterable[str]],
    report: Optional[Dict[str, Any]] = None,
    scope_artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    effective: Dict[str, Dict[str, Dict[str, Any]]] = {
        str(artifact_id): _normalize_transform_ops(raw_ops)
        for artifact_id, raw_ops in dict(base_transforms or {}).items()
    }
    override_payload = transform_overrides or {}
    payload_bytes = len(json.dumps(override_payload, default=_json_default).encode("utf-8"))
    if payload_bytes > MAX_TRANSFORM_OVERRIDE_BYTES:
        raise HTTPException(status_code=422, detail="transform_overrides payload too large")
    if len(override_payload) > MAX_TRANSFORM_OVERRIDE_KEYS:
        raise HTTPException(status_code=422, detail="Too many transform_overrides keys")

    replace_set = set(replace_transforms or [])
    received: Dict[str, Any] = {}
    replace_applied: Set[str] = set()

    for transform_key, override in override_payload.items():
        if not isinstance(override, dict):
            raise HTTPException(status_code=422, detail=f"transform_overrides[{transform_key}] must be an object")
        artifact_id, transform_name = _parse_transform_key(transform_key)
        received[transform_key] = override

        artifact_transforms = dict(effective.get(artifact_id) or {})
        existing_transform = artifact_transforms.get(transform_name)
        if not isinstance(existing_transform, dict):
            existing_transform = {}

        should_replace = transform_key in replace_set
        merged = dict(override) if should_replace else _deep_merge_dict(existing_transform, override)
        merged["name"] = transform_name
        artifact_transforms[transform_name] = merged
        effective[artifact_id] = artifact_transforms
        if should_replace:
            replace_applied.add(transform_key)

    if report is not None:
        report["transform_overrides_received"] = received
        scoped: Dict[str, Any] = effective
        if scope_artifact_id is not None:
            scoped = {scope_artifact_id: dict((effective.get(scope_artifact_id) or {}))}
        report["transform_effective"] = scoped
        report["replace_transforms_applied"] = sorted(replace_applied)

    return effective


def _resolve_pruning_mode(mode: str) -> Dict[str, Any]:
    if mode not in PRUNING_MODE_LEDGER:
        raise HTTPException(status_code=422, detail=f"Unsupported mode: {mode}")
    return PRUNING_MODE_LEDGER[mode]


def _load_artifact_blob(run_id: str, artifact_id: str) -> Tuple[storage.Bucket, Dict[str, Any], storage.Blob]:
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    manifest_blob = bucket.blob(f"runs/{run_id}/manifest.json")
    if not manifest_blob.exists(client):
        raise HTTPException(status_code=404, detail="run_id not found")

    manifest = json.loads(manifest_blob.download_as_bytes().decode("utf-8"))
    found = None
    for item in manifest.get("artifact_registry", []):
        if item.get("artifact_id") == artifact_id:
            found = item
            break
    if not found:
        raise HTTPException(status_code=404, detail="artifact not found")
    data_blob = bucket.blob(found["object_path"])
    if not data_blob.exists(client):
        raise HTTPException(status_code=404, detail="artifact object missing")
    return bucket, found, data_blob


def load_artifact_bytes(run_id: str, artifact_id: str) -> Tuple[str, bytes, Dict[str, Any]]:
    _, found, data_blob = _load_artifact_blob(run_id=run_id, artifact_id=artifact_id)
    payload = data_blob.download_as_bytes()
    return found.get("content_type", "application/octet-stream"), payload, found


def decode_payload(content_type: str, payload: bytes) -> Tuple[str, Any]:
    normalized = (content_type or "").split(";")[0].strip().lower()
    if normalized in {"application/json", "text/json"}:
        try:
            return "json", json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail="Invalid JSON payload") from exc

    if normalized in {"application/jsonl", "application/x-ndjson", "application/jsonlines", "text/plain"}:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid JSONL payload") from exc
        rows: List[Any] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"Invalid JSONL payload at line {idx}") from exc
        return "jsonl", rows

    return "other", None


def _iter_jsonl_bytes(rows: Iterable[Any]) -> Iterator[bytes]:
    for row in rows:
        yield (json.dumps(row, separators=(",", ":"), default=_json_default) + "\n").encode("utf-8")


def _is_list_of_objects(value: Any) -> bool:
    return isinstance(value, list) and (len(value) == 0 or all(isinstance(item, dict) for item in value))


def _matches_path_hint(current_path: str, path_hints_any_of: List[str]) -> bool:
    if not current_path:
        return False
    for path_hint in path_hints_any_of:
        normalized_hint = _normalize_path_hint(str(path_hint))
        if not normalized_hint:
            continue
        if current_path == normalized_hint or current_path.endswith(f".{normalized_hint}"):
            return True
    return False


def _should_drop_via_rule(key: str, value: Any, current_path: str, rule: Dict[str, Any]) -> bool:
    if key != rule.get("key"):
        return False

    guards = rule.get("guards") or {}
    value_kinds = set(guards.get("value_kind_any_of") or [])
    path_hints = list(guards.get("path_hints_any_of") or [])

    if not value_kinds and not path_hints:
        return True

    type_match = False
    if "list_of_objects" in value_kinds and _is_list_of_objects(value):
        type_match = True

    path_match = _matches_path_hint(current_path, path_hints)

    if value_kinds and path_hints:
        return type_match or path_match
    if value_kinds:
        return type_match
    return path_match


def _should_drop_key(key: str, value: Any, current_path: str, key_drops: Set[str], drop_rules: List[Dict[str, Any]]) -> bool:
    if key in key_drops:
        return True
    for rule in drop_rules:
        if _should_drop_via_rule(key=key, value=value, current_path=current_path, rule=rule):
            return True
    return False


def _recursive_prune_keys(
    node: Any,
    key_drops: Set[str],
    keep_keys: Set[str],
    drop_rules: List[Dict[str, Any]],
    report: Dict[str, Any],
    path: str = "",
) -> Any:
    if isinstance(node, list):
        return [
            _recursive_prune_keys(
                item,
                key_drops=key_drops,
                keep_keys=keep_keys,
                drop_rules=drop_rules,
                report=report,
                path=path,
            )
            for item in node
        ]
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for key, value in node.items():
            report["seen_keys"][key] += 1
            current_path = f"{path}.{key}" if path else key
            should_drop = _should_drop_key(
                key=key,
                value=value,
                current_path=current_path,
                key_drops=key_drops,
                drop_rules=drop_rules,
            )
            if should_drop and key not in keep_keys:
                report["dropped_keys"][key] += 1
                continue
            if should_drop and key in keep_keys:
                report["kept_key_hits"][key] += 1
            out[key] = _recursive_prune_keys(
                value,
                key_drops=key_drops,
                keep_keys=keep_keys,
                drop_rules=drop_rules,
                report=report,
                path=current_path,
            )
        return out
    return node


def _resolve_limit_for_path(effective_limits: Dict[str, int], artifact_id: str, path: str) -> Optional[int]:
    full_path = f"{artifact_id}.{path}" if artifact_id and path else path
    if full_path in effective_limits:
        return effective_limits[full_path]
    if path in effective_limits:
        return effective_limits[path]
    return None


def _normalize_keys_tested(value: Any) -> Tuple[str, ...]:
    if isinstance(value, list):
        return tuple(sorted(str(v) for v in value))
    if isinstance(value, tuple):
        return tuple(sorted(str(v) for v in value))
    if value is None:
        return tuple()
    return (str(value),)


def _to_sort_value(value: Any, order: str) -> Any:
    if value is None:
        return float("-inf") if order == "desc" else float("inf")
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def _score_test_row(row: Dict[str, Any], rank_order: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    score: List[Any] = []
    for rank in rank_order:
        field = rank.get("field")
        order = str(rank.get("order", "asc")).lower()
        raw = row.get(field)
        sortable = _to_sort_value(raw, order)
        if isinstance(sortable, float):
            score.append(-sortable if order == "desc" else sortable)
        else:
            score.append(sortable if order == "asc" else "".join(chr(255 - ord(c)) for c in sortable))
    score.append(str(_normalize_keys_tested(row.get("keys_tested"))))
    return tuple(score)


def _apply_a6_tests_policy(node: Dict[str, Any], tests: List[Any], policy: Dict[str, Any], report: Dict[str, Any]) -> List[Any]:
    if not isinstance(tests, list):
        return tests

    k = int(policy.get("k", len(tests)))
    must_keep_cfg = policy.get("must_keep") or {}
    best_candidate_path = str(must_keep_cfg.get("best_candidate_keys_path", ""))
    best_keys: Any = node
    for part in [p for p in best_candidate_path.split(".") if p]:
        if isinstance(best_keys, dict):
            best_keys = best_keys.get(part)
        else:
            best_keys = None
            break
    best_key_norm = _normalize_keys_tested(best_keys)

    kept: List[Dict[str, Any]] = []
    remainder: List[Dict[str, Any]] = []
    for item in tests:
        if not isinstance(item, dict):
            continue
        keys_norm = _normalize_keys_tested(item.get("keys_tested"))
        if best_key_norm and keys_norm == best_key_norm:
            kept.append(item)
        else:
            remainder.append(item)

    rank_order = policy.get("rank_order") or []
    remainder_sorted = sorted(remainder, key=lambda row: _score_test_row(row, rank_order))

    selected = kept + remainder_sorted
    if k >= 0:
        selected = selected[:k]

    report["a6_tests_policy"] = {
        "input": len(tests),
        "best_candidate_kept": bool(kept),
        "output": len(selected),
    }
    return selected


def _extract_columns_from_value_filter(
    value_filter: Optional[Any],
    filter_keys: Optional[Iterable[str]] = None,
) -> Set[str]:
    raw_keys = filter_keys if filter_keys is not None else ["force_include_columns", "a9_force_include_columns"]
    keys = [str(key).strip() for key in raw_keys if str(key).strip()]
    forced_names: Set[str] = set()
    if isinstance(value_filter, dict):
        for key in keys:
            forced_names.update(str(v) for v in (value_filter.get(key) or []))
    elif isinstance(value_filter, list):
        for item in value_filter:
            if isinstance(item, dict):
                for key in keys:
                    forced_names.update(str(v) for v in (item.get(key) or []))
    return {name for name in forced_names if name}


def _ordered_nonempty_strings(values: Iterable[Any]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _extract_string_list_from_value_filter(
    value_filter: Optional[Any],
    filter_keys: Iterable[str],
) -> List[str]:
    keys = [str(key).strip() for key in filter_keys if str(key).strip()]
    if not keys:
        return []

    ordered: List[str] = []
    seen: Set[str] = set()

    def _take(raw: Any) -> None:
        iterable: Iterable[Any]
        if raw is None:
            iterable = []
        elif isinstance(raw, (list, tuple, set)):
            iterable = raw
        else:
            iterable = [raw]
        for item in iterable:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)

    if isinstance(value_filter, dict):
        for key in keys:
            _take(value_filter.get(key))
    elif isinstance(value_filter, list):
        for item in value_filter:
            if not isinstance(item, dict):
                continue
            for key in keys:
                _take(item.get(key))

    return ordered


def _extract_force_include_columns(value_filter: Optional[Any]) -> Set[str]:
    return _extract_columns_from_value_filter(value_filter)


def _sorted_columns_for_bucket(columns: List[Tuple[int, Dict[str, Any]]], name_field: str) -> List[Tuple[int, Dict[str, Any]]]:
    return sorted(columns, key=lambda pair: str(pair[1].get(name_field, "")))


def _normalize_group_sample_mode(value: Any) -> str:
    mode = str(value or "all").strip().lower()
    if mode in {"all", "representative_plus_worst_outlier"}:
        return mode
    return "all"


def _column_scope_match_severity(
    row: Dict[str, Any],
    *,
    bool_true_fields: List[str],
    nonempty_fields: List[str],
    numeric_gte: Dict[str, float],
    numeric_lte: Dict[str, float],
) -> Tuple[int, int, int, float]:
    bool_hits = sum(1 for field in bool_true_fields if bool(row.get(field, False)))
    nonempty_hits = sum(1 for field in nonempty_fields if _is_nonempty_field_value(row.get(field)))
    numeric_hits = 0
    numeric_margin = 0.0

    for field, threshold in numeric_gte.items():
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if value >= threshold:
            numeric_hits += 1
            numeric_margin += value - threshold

    for field, threshold in numeric_lte.items():
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if value <= threshold:
            numeric_hits += 1
            numeric_margin += threshold - value

    return bool_hits, nonempty_hits, numeric_hits, numeric_margin


def _compact_pairs_by_inferred_group(
    pairs: List[Tuple[int, Dict[str, Any]]],
    *,
    name_field: str,
    cap_per_group: int,
    group_sample_mode: str,
    score_fn: Callable[[Dict[str, Any]], Tuple[Any, ...]],
) -> List[Tuple[int, Dict[str, Any]]]:
    if cap_per_group <= 0 or len(pairs) <= 1:
        return list(pairs)

    ordered_pairs = list(pairs)
    group_keys_by_name = _resolve_inferred_group_keys(
        str(pair[1].get(name_field, "") or "").strip()
        for pair in ordered_pairs
    )
    if group_sample_mode == "all":
        compacted: List[Tuple[int, Dict[str, Any]]] = []
        group_counts: Dict[str, int] = {}
        for pair in ordered_pairs:
            _, row = pair
            column_name = str(row.get(name_field, "") or "").strip()
            group_key = group_keys_by_name.get(column_name, _infer_column_group_key(column_name))
            if group_key and group_counts.get(group_key, 0) >= cap_per_group:
                continue
            compacted.append(pair)
            if group_key:
                group_counts[group_key] = group_counts.get(group_key, 0) + 1
        return compacted

    grouped_positions: Dict[str, List[Tuple[int, Tuple[int, Dict[str, Any]]]]] = {}
    for pos, pair in enumerate(ordered_pairs):
        _, row = pair
        column_name = str(row.get(name_field, "") or "").strip()
        group_key = group_keys_by_name.get(column_name, _infer_column_group_key(column_name))
        if not group_key:
            group_key = f"__ungrouped__:{pos}"
        grouped_positions.setdefault(group_key, []).append((pos, pair))

    keep_positions: Set[int] = set()
    for members in grouped_positions.values():
        if len(members) <= cap_per_group:
            keep_positions.update(pos for pos, _ in members)
            continue

        representative_count = min(2, max(0, cap_per_group - 1))
        local_keep: List[int] = [members[idx][0] for idx in range(representative_count)]
        local_keep_set: Set[int] = set(local_keep)

        if cap_per_group > representative_count:
            remaining = [(pos, pair) for pos, pair in members if pos not in local_keep_set]
            if remaining:
                worst_pos, _ = max(
                    remaining,
                    key=lambda item: (score_fn(item[1][1]), -item[0]),
                )
                local_keep.append(worst_pos)
                local_keep_set.add(worst_pos)

        if len(local_keep) < cap_per_group:
            for pos, _ in members:
                if pos in local_keep_set:
                    continue
                local_keep.append(pos)
                local_keep_set.add(pos)
                if len(local_keep) >= cap_per_group:
                    break

        keep_positions.update(local_keep)

    return [pair for pos, pair in enumerate(ordered_pairs) if pos in keep_positions]


def _apply_a9_columns_policy(
    columns: List[Any],
    policy: Dict[str, Any],
    value_filter: Optional[Any],
    report: Dict[str, Any],
) -> List[Any]:
    if not isinstance(columns, list):
        return columns

    name_field = str(policy.get("name_field", "column"))
    role_field = str(policy.get("role_field", "primary_role"))
    review_field = str(policy.get("review_field", "review_required"))
    hard_cap = int(policy.get("hard_cap_total", len(columns)))
    exclude_roles = set(policy.get("exclude_roles") or [])
    try:
        selection_cap_per_group = int(policy.get("selection_cap_per_inferred_group", 0) or 0)
    except (TypeError, ValueError):
        selection_cap_per_group = 0
    group_sample_mode = _normalize_group_sample_mode(policy.get("group_sample_mode"))

    valid_columns: List[Tuple[int, Dict[str, Any]]] = [
        (idx, col) for idx, col in enumerate(columns) if isinstance(col, dict)
    ]
    forced_names = _extract_force_include_columns(value_filter)

    selected: List[Tuple[int, Dict[str, Any]]] = []
    selected_idx: Set[int] = set()

    def _take(pair: Tuple[int, Dict[str, Any]]) -> None:
        idx, col = pair
        if idx not in selected_idx and len(selected) < hard_cap:
            selected_idx.add(idx)
            selected.append((idx, col))

    for pair in _sorted_columns_for_bucket(valid_columns, name_field):
        _, col = pair
        if str(col.get(name_field, "")) in forced_names:
            _take(pair)

    include_roles_all = policy.get("include_roles_all") or []
    overflow_flags: Dict[str, bool] = {}
    for rule in include_roles_all:
        role = rule.get("role")
        cap = int(rule.get("cap", len(valid_columns)))
        overflow_flag = rule.get("overflow_flag")
        bucket = [pair for pair in valid_columns if str(pair[1].get(role_field, "")) == str(role)]
        bucket = _sorted_columns_for_bucket(bucket, name_field)
        if overflow_flag:
            overflow_flags[str(overflow_flag)] = len(bucket) > cap
        for pair in bucket[:cap]:
            _take(pair)

    include_roles_topk = policy.get("include_roles_topk") or {}
    for role, cap_val in include_roles_topk.items():
        cap = int(cap_val)
        bucket = [
            pair for pair in valid_columns
            if str(pair[1].get(role_field, "")) == str(role)
        ]
        for pair in _sorted_columns_for_bucket(bucket, name_field)[:cap]:
            _take(pair)

    include_review_required_topk = int(policy.get("include_review_required_topk", 0))
    if include_review_required_topk > 0:
        bucket = [
            pair for pair in valid_columns
            if bool(pair[1].get(review_field, False))
        ]
        for pair in _sorted_columns_for_bucket(bucket, name_field)[:include_review_required_topk]:
            _take(pair)

    final_selected: List[Dict[str, Any]] = []
    filtered_selected: List[Tuple[int, Dict[str, Any]]] = []
    for pair in selected:
        _, col = pair
        role = str(col.get(role_field, ""))
        if role in exclude_roles and str(col.get(name_field, "")) not in forced_names:
            continue
        filtered_selected.append(pair)

    if selection_cap_per_group > 0:
        forced_selected = [
            pair for pair in filtered_selected
            if str(pair[1].get(name_field, "")) in forced_names
        ]
        non_forced_selected = [
            pair for pair in filtered_selected
            if str(pair[1].get(name_field, "")) not in forced_names
        ]
        filtered_selected = forced_selected + _compact_pairs_by_inferred_group(
            non_forced_selected,
            name_field=name_field,
            cap_per_group=selection_cap_per_group,
            group_sample_mode=group_sample_mode,
            score_fn=lambda row: (1 if bool(row.get(review_field, False)) else 0,),
        )

    final_selected = [col for _, col in filtered_selected]

    if len(final_selected) > hard_cap:
        final_selected = final_selected[:hard_cap]

    report["a9_columns_policy"] = {
        "input": len(columns),
        "output": len(final_selected),
        "forced_includes": sorted(forced_names),
        "overflow_flags": overflow_flags,
        "selection_cap_per_inferred_group": selection_cap_per_group,
        "group_sample_mode": group_sample_mode,
    }
    return final_selected


def _is_nonempty_field_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) > 0
    return True


def _infer_column_group_key(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        return ""

    match = re.match(r"^([A-Za-z]\d+)_Q\d+$", text)
    if match:
        return match.group(1)

    match = re.match(r"^([A-Za-z]+\d+(?:Main|Secondary)?_cell_group)Row\d+$", text)
    if match:
        return match.group(1)

    match = re.match(r"^([A-Za-z]+\d+_[A-Za-z0-9]+)Row\d+$", text)
    if match:
        return match.group(1)

    return text


def _parse_dense_numbered_family_candidate(name: Any) -> Optional[Tuple[str, int]]:
    text = str(name or "").strip()
    if not text:
        return None

    match = re.match(r"^([A-Za-z]+(?:_[A-Za-z]+)*)(\d+)$", text)
    if not match:
        return None

    try:
        suffix = int(match.group(2))
    except (TypeError, ValueError):
        return None
    return match.group(1), suffix


def _resolve_inferred_group_keys(names: Iterable[Any]) -> Dict[str, str]:
    ordered_names = _ordered_nonempty_strings(names)
    if not ordered_names:
        return {}

    stem_members: Dict[str, List[str]] = {}
    stem_suffixes: Dict[str, Set[int]] = {}
    resolved: Dict[str, str] = {}

    for name in ordered_names:
        explicit_group = _infer_column_group_key(name)
        if explicit_group != name:
            resolved[name] = explicit_group
            continue

        fallback_candidate = _parse_dense_numbered_family_candidate(name)
        if fallback_candidate is None:
            resolved[name] = explicit_group
            continue

        stem, suffix = fallback_candidate
        stem_members.setdefault(stem, []).append(name)
        stem_suffixes.setdefault(stem, set()).add(suffix)

    dense_stems = {
        stem
        for stem, members in stem_members.items()
        if len(members) >= 5 and len(stem_suffixes.get(stem, set())) >= 4
    }

    for name in ordered_names:
        if name in resolved:
            continue
        fallback_candidate = _parse_dense_numbered_family_candidate(name)
        if fallback_candidate is None:
            resolved[name] = _infer_column_group_key(name)
            continue
        stem, _ = fallback_candidate
        resolved[name] = stem if stem in dense_stems else _infer_column_group_key(name)

    return resolved


def _apply_column_scope_selector_policy(
    items: List[Any],
    policy: Dict[str, Any],
    value_filter: Optional[Any],
    report: Dict[str, Any],
) -> List[Any]:
    if not isinstance(items, list):
        return items

    name_field = str(policy.get("name_field", "column"))
    hard_cap = int(policy.get("hard_cap_total", len(items)))
    bool_true_fields = [str(field) for field in (policy.get("include_bool_true_fields") or []) if str(field).strip()]
    nonempty_fields = [str(field) for field in (policy.get("include_nonempty_fields") or []) if str(field).strip()]
    numeric_gte = {
        str(field): float(threshold)
        for field, threshold in (policy.get("include_numeric_gte") or {}).items()
        if str(field).strip()
    }
    numeric_lte = {
        str(field): float(threshold)
        for field, threshold in (policy.get("include_numeric_lte") or {}).items()
        if str(field).strip()
    }
    raw_filter_keys = policy.get("include_from_filter_keys", None)
    if raw_filter_keys is None:
        raw_filter_keys = ["force_include_columns"]
    filter_keys = [str(field) for field in (raw_filter_keys or []) if str(field).strip()]
    seed_filter_keys = [str(field) for field in (policy.get("seed_from_filter_keys") or []) if str(field).strip()]
    try:
        legacy_hard_cap_per_group = int(policy.get("hard_cap_per_inferred_group", 0) or 0)
    except (TypeError, ValueError):
        legacy_hard_cap_per_group = 0
    try:
        seed_cap_per_group = int(policy.get("seed_cap_per_inferred_group", 0) or 0)
    except (TypeError, ValueError):
        seed_cap_per_group = 0
    try:
        standalone_cap_per_group = int(policy.get("standalone_cap_per_inferred_group", 0) or 0)
    except (TypeError, ValueError):
        standalone_cap_per_group = 0
    if standalone_cap_per_group <= 0:
        standalone_cap_per_group = legacy_hard_cap_per_group
    group_sample_mode = _normalize_group_sample_mode(policy.get("group_sample_mode"))

    valid_items: List[Tuple[int, Dict[str, Any]]] = [
        (idx, row) for idx, row in enumerate(items) if isinstance(row, dict)
    ]
    forced_names = _extract_columns_from_value_filter(value_filter, filter_keys)
    seeded_names = _extract_string_list_from_value_filter(value_filter, seed_filter_keys)
    pair_by_name: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for pair in valid_items:
        _, row = pair
        pair_by_name.setdefault(str(row.get(name_field, "")), pair)

    def _matches(row: Dict[str, Any]) -> bool:
        for field in bool_true_fields:
            if bool(row.get(field, False)):
                return True
        for field in nonempty_fields:
            if _is_nonempty_field_value(row.get(field)):
                return True
        for field, threshold in numeric_gte.items():
            try:
                if float(row.get(field) or 0.0) >= threshold:
                    return True
            except (TypeError, ValueError):
                continue
        for field, threshold in numeric_lte.items():
            try:
                if float(row.get(field) or 0.0) <= threshold:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    ordered_pairs = _sorted_columns_for_bucket(valid_items, name_field)
    forced_pairs = [
        pair for pair in ordered_pairs
        if str(pair[1].get(name_field, "")) in forced_names
    ]
    forced_idx = {idx for idx, _ in forced_pairs}
    forced_name_set = {str(row.get(name_field, "")) for _, row in forced_pairs}
    effective_cap = max(hard_cap, len(forced_pairs))

    seeded_candidates: List[Tuple[int, Dict[str, Any]]] = []
    seeded_seen: Set[int] = set()
    for seeded_name in seeded_names:
        pair = pair_by_name.get(seeded_name)
        if pair is None:
            continue
        idx, row = pair
        if idx in forced_idx or idx in seeded_seen:
            continue
        seeded_seen.add(idx)
        seeded_candidates.append((idx, row))

    compacted_seeded = _compact_pairs_by_inferred_group(
        seeded_candidates,
        name_field=name_field,
        cap_per_group=seed_cap_per_group,
        group_sample_mode=group_sample_mode,
        score_fn=lambda row: _column_scope_match_severity(
            row,
            bool_true_fields=bool_true_fields,
            nonempty_fields=nonempty_fields,
            numeric_gte=numeric_gte,
            numeric_lte=numeric_lte,
        ),
    )
    remaining_budget = max(0, effective_cap - len(forced_pairs))
    seeded_pairs = compacted_seeded[:remaining_budget]
    seeded_idx = {idx for idx, _ in seeded_pairs}

    standalone_candidates = [
        pair for pair in ordered_pairs
        if pair[0] not in forced_idx and pair[0] not in seeded_idx and _matches(pair[1])
    ]
    compacted_standalone = _compact_pairs_by_inferred_group(
        standalone_candidates,
        name_field=name_field,
        cap_per_group=standalone_cap_per_group,
        group_sample_mode=group_sample_mode,
        score_fn=lambda row: _column_scope_match_severity(
            row,
            bool_true_fields=bool_true_fields,
            nonempty_fields=nonempty_fields,
            numeric_gte=numeric_gte,
            numeric_lte=numeric_lte,
        ),
    )
    remaining_budget = max(0, effective_cap - len(forced_pairs) - len(seeded_pairs))
    standalone_pairs = compacted_standalone[:remaining_budget]

    final_selected = [row for _, row in (forced_pairs + seeded_pairs + standalone_pairs)]
    report.setdefault("column_scope_policies", []).append({
        "input": len(items),
        "output": len(final_selected),
        "forced_includes": sorted(forced_names),
        "include_from_filter_keys": filter_keys,
        "seeded_includes": [name for name in seeded_names if name in pair_by_name and name not in forced_name_set],
        "seed_from_filter_keys": seed_filter_keys,
        "bool_true_fields": bool_true_fields,
        "nonempty_fields": nonempty_fields,
        "numeric_gte": numeric_gte,
        "numeric_lte": numeric_lte,
        "hard_cap_total": hard_cap,
        "hard_cap_per_inferred_group": legacy_hard_cap_per_group,
        "seed_cap_per_inferred_group": seed_cap_per_group,
        "standalone_cap_per_inferred_group": standalone_cap_per_group,
        "group_sample_mode": group_sample_mode,
        "forced_output": len(forced_pairs),
        "seeded_output": len(seeded_pairs),
        "standalone_output": len(standalone_pairs),
    })
    return final_selected




def _resolve_policy_target_path(policy_name: str, policy: Dict[str, Any]) -> Optional[str]:
    target = str(policy.get("target_array", "")).strip()
    if target == "__root__":
        return ""
    if target:
        return target
    if policy_name.endswith("_policy"):
        return policy_name[:-7]
    return policy_name or None


def _resolve_policy_type(policy_name: str, policy: Dict[str, Any]) -> str:
    policy_type = str(policy.get("type", "")).strip().lower()
    if policy_type:
        return policy_type
    if policy_name == "columns_policy":
        return "bucket_selector"
    if policy_name == "tests_policy":
        return "ranked_topk"
    return ""


def _apply_ranked_topk_policy(node: Dict[str, Any], items: List[Any], policy: Dict[str, Any], report: Dict[str, Any]) -> List[Any]:
    return _apply_a6_tests_policy(node, items, policy, report)


def _apply_bucket_selector_policy(items: List[Any], policy: Dict[str, Any], value_filter: Optional[Any], report: Dict[str, Any]) -> List[Any]:
    return _apply_a9_columns_policy(items, policy, value_filter, report)


def _apply_typed_array_policy(
    node: Dict[str, Any],
    items: List[Any],
    policy_name: str,
    policy: Dict[str, Any],
    value_filter: Optional[Any],
    report: Dict[str, Any],
) -> List[Any]:
    policy_type = _resolve_policy_type(policy_name, policy)
    if policy_type in {"ranked_topk", "best_candidate_safe_ranked_topk"}:
        return _apply_ranked_topk_policy(node=node, items=items, policy=policy, report=report)
    if policy_type in {"bucket_selector", "role_bucket_selector"}:
        return _apply_bucket_selector_policy(items=items, policy=policy, value_filter=value_filter, report=report)
    if policy_type == "column_scope_selector":
        return _apply_column_scope_selector_policy(items=items, policy=policy, value_filter=value_filter, report=report)
    return items


def _path_parts(path: str) -> List[str]:
    parts: List[str] = []
    for token in [p for p in str(path or "").split(".") if p]:
        if token.endswith("]") and "[" in token:
            base, _, index = token[:-1].partition("[")
            if base:
                parts.append(base)
            parts.append(index)
        else:
            parts.append(token)
    return parts


def _path_get(node: Any, path: str, default: Any = None) -> Any:
    cur = node
    for part in _path_parts(path):
        if isinstance(cur, list):
            if not part.isdigit():
                return default
            idx = int(part)
            if idx < 0 or idx >= len(cur):
                return default
            cur = cur[idx]
            continue
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
            continue
        return default
    return cur


def _path_set(node: Any, path: str, value: Any) -> bool:
    parts = _path_parts(path)
    if not parts:
        return False
    cur = node
    for part in parts[:-1]:
        if isinstance(cur, dict):
            if part not in cur or not isinstance(cur[part], (dict, list)):
                cur[part] = {}
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            if idx < 0 or idx >= len(cur):
                return False
            cur = cur[idx]
        else:
            return False

    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
        return True
    if isinstance(cur, list) and last.isdigit():
        idx = int(last)
        if idx < 0 or idx >= len(cur):
            return False
        cur[idx] = value
        return True
    return False


def _render_template(template: str, context: Dict[str, Any]) -> str:
    out = str(template)
    for key, val in context.items():
        out = out.replace("{" + key + "}", "" if val is None else str(val))
    return out


def _derive_value(spec: Any, root: Dict[str, Any], source: Optional[Dict[str, Any]], target: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(spec, dict):
        return spec

    if "const" in spec:
        return spec.get("const")

    if "from" in spec:
        raw_path = str(spec.get("from") or "")
        if raw_path.startswith("source."):
            value = _path_get(source or {}, raw_path[len("source."):])
        elif raw_path.startswith("target."):
            value = _path_get(target or {}, raw_path[len("target."):])
        else:
            value = _path_get(source or {}, raw_path)
            if value is None:
                value = _path_get(target or {}, raw_path)
            if value is None:
                value = _path_get(root, raw_path)
        if value is None and spec.get("fallback_from"):
            value = _derive_value({"from": spec.get("fallback_from")}, root, source, target)
    else:
        value = None

    if "slice" in spec and isinstance(value, list):
        try:
            k = int(spec.get("slice", len(value)))
        except (TypeError, ValueError):
            k = len(value)
        try:
            k_max = int(spec.get("k_max", k))
        except (TypeError, ValueError):
            k_max = k
        value = value[:max(0, min(k, k_max))]

    transform = str(spec.get("transform", "")).strip().lower()
    if isinstance(value, str):
        if transform == "lower":
            value = value.lower()
        elif transform == "upper":
            value = value.upper()
        elif transform == "title":
            value = value.title()

    if "template" in spec:
        ctx = {
            "value": value,
            "source": source,
            "target": target,
        }
        if isinstance(source, dict):
            for k, v in source.items():
                ctx[f"source.{k}"] = v
        if isinstance(target, dict):
            for k, v in target.items():
                ctx[f"target.{k}"] = v
        value = _render_template(str(spec.get("template")), ctx)

    return value


def _op_join_array_by_key(root: Dict[str, Any], op: Dict[str, Any], report: Dict[str, Any]) -> None:
    target_path = str(op.get("target_array_path") or "")
    source_path = str(op.get("source_array_path") or "")
    join_key = str(op.get("join_key") or "")
    write_field = str(op.get("write_field") or "")
    field_map = op.get("field_map") or {}
    if not target_path or not source_path or not join_key or not write_field or not isinstance(field_map, dict):
        return

    target = _path_get(root, target_path)
    source = _path_get(root, source_path)
    if not isinstance(target, list) or not isinstance(source, list):
        return

    source_index: Dict[Any, Dict[str, Any]] = {}
    for row in source:
        if isinstance(row, dict) and join_key in row and row.get(join_key) is not None:
            source_index[row.get(join_key)] = row

    touched = 0
    for row in target:
        if not isinstance(row, dict):
            continue
        join_val = row.get(join_key)
        if join_val is None:
            continue
        src = source_index.get(join_val)
        if not isinstance(src, dict):
            continue
        derived: Dict[str, Any] = {}
        for out_key, spec in field_map.items():
            derived[str(out_key)] = _derive_value(spec, root=root, source=src, target=row)
        row[write_field] = derived
        touched += 1
    report.setdefault("transform_ops_applied", []).append({
        "name": str(op.get("name") or ""),
        "type": "join_array_by_key",
        "target": target_path,
        "source": source_path,
        "rows_touched": touched,
    })


def _op_derive_fields(root: Dict[str, Any], op: Dict[str, Any], report: Dict[str, Any]) -> None:
    target_path = str(op.get("target_object_path") or "")
    fields = op.get("fields") or {}
    if not target_path or not isinstance(fields, dict):
        return
    target = _path_get(root, target_path)
    if not isinstance(target, dict):
        return
    for field_name, spec in fields.items():
        target[str(field_name)] = _derive_value(spec, root=root, source=target, target=target)
    report.setdefault("transform_ops_applied", []).append({"name": str(op.get("name") or ""), "type": "derive_fields", "target": target_path, "field_count": len(fields)})


def _op_slice_preview(root: Dict[str, Any], op: Dict[str, Any], report: Dict[str, Any]) -> None:
    target_path = str(op.get("target_path") or "")
    if not target_path:
        return
    values = _path_get(root, target_path)
    if not isinstance(values, list):
        return
    try:
        k = int(op.get("k", len(values)))
    except (TypeError, ValueError):
        k = len(values)
    try:
        k_max = int(op.get("k_max", k))
    except (TypeError, ValueError):
        k_max = k
    limit = max(0, min(k, k_max))
    if len(values) <= limit:
        return
    _path_set(root, target_path, values[:limit])
    report.setdefault("transform_ops_applied", []).append({
        "name": str(op.get("name") or ""),
        "type": "slice_preview",
        "target": target_path,
        "before": len(values),
        "after": limit,
    })


def _resolve_transform_target(root: Any, path: str) -> Any:
    normalized = str(path or "").strip()
    if normalized in {"", "__root__"}:
        return root
    if isinstance(root, dict):
        return _path_get(root, normalized)
    return None


def _short_scalar_preview(value: Any, max_chars: int) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _op_compact_keyed_string_arrays(root: Any, op: Dict[str, Any], report: Dict[str, Any]) -> None:
    target = _resolve_transform_target(root, str(op.get("target_array_path") or ""))
    field_name = str(op.get("field") or "")
    if not field_name or not isinstance(target, list):
        return

    try:
        max_items = int(op.get("max_items_per_key", 2))
    except (TypeError, ValueError):
        max_items = 2
    try:
        max_chars = int(op.get("max_chars", 120))
    except (TypeError, ValueError):
        max_chars = 120

    touched = 0
    for row in target:
        if not isinstance(row, dict):
            continue
        raw_map = row.get(field_name)
        if not isinstance(raw_map, dict):
            continue
        compacted: Dict[str, List[str]] = {}
        for key, values in raw_map.items():
            if not isinstance(values, list):
                continue
            previews: List[str] = []
            for item in values:
                short = _short_scalar_preview(item, max_chars=max_chars)
                if short is not None:
                    previews.append(short)
                if len(previews) >= max_items:
                    break
            compacted[str(key)] = previews
        row[field_name] = compacted
        touched += 1

    if touched:
        report.setdefault("transform_ops_applied", []).append({
            "name": str(op.get("name") or ""),
            "type": "compact_keyed_string_arrays",
            "target": str(op.get("target_array_path") or "__root__"),
            "field": field_name,
            "rows_touched": touched,
        })


def _op_derive_role_candidate_preview(root: Any, op: Dict[str, Any], report: Dict[str, Any]) -> None:
    target = _resolve_transform_target(root, str(op.get("target_array_path") or ""))
    if not isinstance(target, list):
        return

    output_field = str(op.get("output_field") or "role_candidate_preview")
    candidates_field = str(op.get("candidates_field") or "role_candidates")
    scores_field = str(op.get("scores_field") or "role_scores")
    role_key = str(op.get("role_key") or "role")
    score_key = str(op.get("score_key") or "score")
    try:
        top_k = int(op.get("top_k", 2))
    except (TypeError, ValueError):
        top_k = 2

    touched = 0
    for row in target:
        if not isinstance(row, dict):
            continue
        preview: List[Dict[str, Any]] = []
        candidates = row.get(candidates_field)
        if isinstance(candidates, list):
            ranked = [
                item for item in candidates
                if isinstance(item, dict) and str(item.get(role_key) or "").strip()
            ]
            ranked = sorted(
                ranked,
                key=lambda item: (-float(item.get(score_key) or 0.0), str(item.get(role_key) or "")),
            )
            for item in ranked[:max(0, top_k)]:
                preview.append({
                    role_key: str(item.get(role_key) or ""),
                    score_key: float(item.get(score_key) or 0.0),
                })
        elif isinstance(row.get(scores_field), dict):
            ranked_pairs = sorted(
                [
                    (str(role), float(score or 0.0))
                    for role, score in row.get(scores_field, {}).items()
                    if str(role).strip()
                ],
                key=lambda pair: (-pair[1], pair[0]),
            )
            preview = [{role_key: role, score_key: score} for role, score in ranked_pairs[:max(0, top_k)]]
        row[output_field] = preview
        touched += 1

    if touched:
        report.setdefault("transform_ops_applied", []).append({
            "name": str(op.get("name") or ""),
            "type": "derive_role_candidate_preview",
            "target": str(op.get("target_array_path") or "__root__"),
            "rows_touched": touched,
        })


def _op_drop_fields_when_empty(root: Any, op: Dict[str, Any], report: Dict[str, Any]) -> None:
    target = _resolve_transform_target(root, str(op.get("target_array_path") or ""))
    if not isinstance(target, list):
        return

    fields = [str(field).strip() for field in (op.get("fields") or []) if str(field).strip()]
    if not fields:
        return

    rows_touched = 0
    fields_removed = 0
    for row in target:
        if not isinstance(row, dict):
            continue
        row_changed = False
        for field in fields:
            if field in row and not _is_nonempty_field_value(row.get(field)):
                row.pop(field, None)
                fields_removed += 1
                row_changed = True
        if row_changed:
            rows_touched += 1

    if rows_touched:
        report.setdefault("transform_ops_applied", []).append({
            "name": str(op.get("name") or ""),
            "type": "drop_fields_when_empty",
            "target": str(op.get("target_array_path") or "__root__"),
            "rows_touched": rows_touched,
            "fields_removed": fields_removed,
        })


def _op_project_a2_semantic_context_grounding(
    root: Any,
    op: Dict[str, Any],
    report: Dict[str, Any],
    value_filter: Optional[Any],
) -> Any:
    if not isinstance(root, list):
        return root

    try:
        low_card_max_unique = int(op.get("low_cardinality_max_unique_count", 12))
    except (TypeError, ValueError):
        low_card_max_unique = 12
    try:
        preview_cap = int(op.get("low_cardinality_preview_cap", 150))
    except (TypeError, ValueError):
        preview_cap = 150
    try:
        top_levels_limit = int(op.get("top_levels_limit", 6))
    except (TypeError, ValueError):
        top_levels_limit = 6

    priority_keys = [
        str(key).strip()
        for key in (op.get("priority_filter_keys") or [])
        if str(key).strip()
    ] or [
        "force_include_columns",
        "review_columns",
        "structural_columns",
        "skip_trigger_columns",
    ]

    priority_buckets: Dict[str, Set[str]] = {
        key: _extract_columns_from_value_filter(value_filter, [key])
        for key in priority_keys
    }

    columns_index: List[Dict[str, Any]] = []
    low_card_candidates: List[Dict[str, Any]] = []

    for row in root:
        if not isinstance(row, dict):
            continue
        column = str(row.get("column") or "").strip()
        if not column:
            continue
        top_candidate = row.get("top_candidate") or {}
        try:
            unique_count = int(row.get("unique_count", 0) or 0)
        except (TypeError, ValueError):
            unique_count = 0
        try:
            missing_pct = float(row.get("missing_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            missing_pct = 0.0

        columns_index.append({
            "column": column,
            "top_candidate_type": str(top_candidate.get("type") or ""),
            "unique_count": unique_count,
            "missing_pct": missing_pct,
            "high_uniqueness_candidate": bool(row.get("high_uniqueness_candidate", False)),
        })

        if 0 < unique_count <= low_card_max_unique:
            preview_row = {
                "column": column,
                "unique_count": unique_count,
                "top_levels": [str(v) for v in (row.get("top_levels") or [])[:max(0, top_levels_limit)] if str(v).strip()],
                "missing_tokens_observed": row.get("missing_tokens_observed") or {},
                "is_one_hot_like": bool(row.get("is_one_hot_like", False)),
            }
            low_card_candidates.append(preview_row)

    def _preview_sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
        column = str(item.get("column") or "")
        bucket_rank = len(priority_keys)
        for idx, key in enumerate(priority_keys):
            if column in priority_buckets.get(key, set()):
                bucket_rank = idx
                break
        return (
            bucket_rank,
            int(item.get("unique_count", 0) or 0),
            column,
        )

    low_card_candidates = sorted(low_card_candidates, key=_preview_sort_key)
    low_cardinality_value_preview = low_card_candidates[:max(0, preview_cap)]

    transformed = {
        "artifact": "A2",
        "columns_index": columns_index,
        "low_cardinality_value_preview": low_cardinality_value_preview,
    }
    report.setdefault("transform_ops_applied", []).append({
        "name": str(op.get("name") or ""),
        "type": "project_a2_semantic_context_grounding",
        "columns_index_count": len(columns_index),
        "low_cardinality_candidates": len(low_card_candidates),
        "low_cardinality_output": len(low_cardinality_value_preview),
        "priority_filter_keys": priority_keys,
    })
    return transformed


def _extract_layout_candidate_grain_keys(candidate: Dict[str, Any]) -> List[str]:
    grain = candidate.get("grain") if isinstance(candidate.get("grain"), dict) else {}
    summary = candidate.get("summary") if isinstance(candidate.get("summary"), dict) else {}

    raw_candidates = [
        grain.get("keys_tested"),
        summary.get("current_row_grain"),
    ]
    for raw in raw_candidates:
        if isinstance(raw, (list, tuple, set)):
            ordered = _ordered_nonempty_strings(raw)
            if ordered:
                return ordered
        elif raw is not None:
            text = str(raw).strip()
            if text:
                return [text]
    return []


def _extract_layout_candidate_score(candidate: Dict[str, Any]) -> float:
    summary = candidate.get("summary") if isinstance(candidate.get("summary"), dict) else {}
    debug_trace = candidate.get("debug_trace") if isinstance(candidate.get("debug_trace"), dict) else {}

    for raw in [candidate.get("score"), summary.get("score"), debug_trace.get("score")]:
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _classify_preferred_grain_match(
    candidate_keys: Iterable[Any],
    preferred_keys: Iterable[Any],
) -> Tuple[int, str]:
    candidate_ordered = _ordered_nonempty_strings(candidate_keys)
    preferred_ordered = _ordered_nonempty_strings(preferred_keys)
    if not preferred_ordered or not candidate_ordered:
        return 3, "none"
    if candidate_ordered == preferred_ordered:
        return 0, "exact"
    if _normalize_keys_tested(candidate_ordered) == _normalize_keys_tested(preferred_ordered):
        return 1, "exact"
    if set(candidate_ordered) & set(preferred_ordered):
        return 2, "overlap"
    return 3, "none"


def _nonempty_sequence_count(value: Any) -> int:
    if not isinstance(value, (list, tuple, set)):
        return 0
    return sum(1 for item in value if str(item or "").strip())


def _compact_a12_family_reference(reference: Any, preview_cap: int) -> Optional[Dict[str, Any]]:
    if not isinstance(reference, dict):
        return None

    compact: Dict[str, Any] = {}
    for key in [
        "family_id",
        "detection_confidence",
        "columns_count",
        "recommended_repeat_dimension_name",
        "flags",
    ]:
        if key in reference:
            compact[key] = copy.deepcopy(reference.get(key))

    columns_preview = reference.get("columns_preview")
    if not isinstance(columns_preview, list):
        columns_preview = reference.get("columns")
    preview_values = _ordered_nonempty_strings(columns_preview or [])[:max(0, preview_cap)]
    if preview_values:
        compact["columns_preview"] = preview_values
    if "columns_count" not in compact and isinstance(reference.get("columns"), list):
        compact["columns_count"] = len(reference.get("columns") or [])

    return compact or None


def _compact_a12_table(table: Any, family_preview_cap: int) -> Optional[Dict[str, Any]]:
    if not isinstance(table, dict):
        return None

    compact: Dict[str, Any] = {}
    for key in [
        "table_id",
        "table_name",
        "kind",
        "grain",
        "repeat_dimension_candidates",
        "column_counts",
        "foreign_keys",
    ]:
        if key in table:
            compact[key] = copy.deepcopy(table.get(key))

    family_reference = _compact_a12_family_reference(table.get("family_reference"), family_preview_cap)
    if family_reference:
        compact["family_reference"] = family_reference

    return compact or None


def _compact_a12_column_placement(
    placement: Any,
    coverage_summary: Any,
    preview_cap: int,
) -> Dict[str, Any]:
    placement_dict = placement if isinstance(placement, dict) else {}
    coverage_dict = coverage_summary if isinstance(coverage_summary, dict) else {}
    compact: Dict[str, Any] = {}

    covered_by_layout = _ordered_nonempty_strings(placement_dict.get("covered_by_layout") or [])
    unmapped = _ordered_nonempty_strings(placement_dict.get("unmapped") or [])

    if covered_by_layout:
        compact["covered_by_layout_preview"] = covered_by_layout[:max(0, preview_cap)]
    if unmapped:
        compact["unmapped_preview"] = unmapped[:max(0, preview_cap)]

    covered_by_role_only_count = _nonempty_sequence_count(placement_dict.get("covered_by_role_only"))
    if covered_by_role_only_count <= 0:
        try:
            covered_by_role_only_count = int(coverage_dict.get("covered_by_role_only_count") or 0)
        except (TypeError, ValueError):
            covered_by_role_only_count = 0
    if covered_by_role_only_count > 0:
        compact["covered_by_role_only_count"] = covered_by_role_only_count

    ambiguous_count = _nonempty_sequence_count(placement_dict.get("ambiguous"))
    if ambiguous_count <= 0:
        try:
            ambiguous_count = int(coverage_dict.get("ambiguous_count") or 0)
        except (TypeError, ValueError):
            ambiguous_count = 0
    if ambiguous_count > 0:
        compact["ambiguous_count"] = ambiguous_count

    return compact


def _compact_a12_ambiguities(ambiguities: Any, cap: int) -> List[Dict[str, Any]]:
    if not isinstance(ambiguities, list):
        return []

    compact: List[Dict[str, Any]] = []
    for row in ambiguities:
        if not isinstance(row, dict):
            continue
        column = str(row.get("column") or "").strip()
        if not column:
            continue
        compact_row: Dict[str, Any] = {"column": column}
        primary_role = str(row.get("primary_role") or "").strip()
        if primary_role:
            compact_row["primary_role"] = primary_role

        alternate_roles: List[Dict[str, Any]] = []
        for alt in (row.get("alternate_roles") or []):
            if not isinstance(alt, dict):
                continue
            role = str(alt.get("role") or "").strip()
            if not role:
                continue
            compact_alt: Dict[str, Any] = {"role": role}
            try:
                if alt.get("score") is not None:
                    compact_alt["score"] = float(alt.get("score"))
            except (TypeError, ValueError):
                pass
            alternate_roles.append(compact_alt)
        if alternate_roles:
            compact_row["alternate_roles"] = alternate_roles

        compact.append(compact_row)
        if len(compact) >= max(0, cap):
            break

    return compact


def _op_project_a12_table_layout_grounding(
    root: Any,
    op: Dict[str, Any],
    report: Dict[str, Any],
    value_filter: Optional[Any],
) -> Any:
    if not isinstance(root, dict):
        return root

    try:
        layout_candidates_top_k = int(op.get("layout_candidates_top_k", 2))
    except (TypeError, ValueError):
        layout_candidates_top_k = 2
    try:
        column_preview_cap = int(op.get("column_placement_preview_cap", 12))
    except (TypeError, ValueError):
        column_preview_cap = 12
    try:
        ambiguities_cap = int(op.get("ambiguities_cap", 8))
    except (TypeError, ValueError):
        ambiguities_cap = 8
    try:
        family_preview_cap = int(op.get("family_columns_preview_cap", 8))
    except (TypeError, ValueError):
        family_preview_cap = 8

    preferred_grain_keys = _extract_string_list_from_value_filter(
        value_filter,
        [str(op.get("preferred_grain_filter_key") or "preferred_primary_grain_keys")],
    )

    ranked_candidates: List[Tuple[Any, ...]] = []
    for idx, candidate in enumerate(root.get("layout_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        candidate_keys = _extract_layout_candidate_grain_keys(candidate)
        match_rank, match_label = _classify_preferred_grain_match(candidate_keys, preferred_grain_keys)
        score = _extract_layout_candidate_score(candidate)

        projected_candidate: Dict[str, Any] = {
            "layout_id": str(candidate.get("layout_id") or ""),
            "score": score,
            "preferred_grain_match": match_label,
            "summary": copy.deepcopy(candidate.get("summary") or {}),
            "grain": copy.deepcopy(candidate.get("grain") or {}),
            "tables": [],
            "coverage_summary": copy.deepcopy(candidate.get("coverage_summary") or {}),
            "column_placement": _compact_a12_column_placement(
                candidate.get("column_placement"),
                candidate.get("coverage_summary"),
                column_preview_cap,
            ),
            "ambiguities": _compact_a12_ambiguities(candidate.get("ambiguities"), ambiguities_cap),
        }

        if not projected_candidate["grain"] and candidate_keys:
            projected_candidate["grain"] = {"keys_tested": candidate_keys}

        for table in candidate.get("tables") or []:
            compact_table = _compact_a12_table(table, family_preview_cap)
            if compact_table:
                projected_candidate["tables"].append(compact_table)

        if preferred_grain_keys:
            ranked_candidates.append((match_rank, -score, idx, projected_candidate))
        else:
            ranked_candidates.append((-score, idx, projected_candidate))

    ranked_candidates = sorted(ranked_candidates)
    if preferred_grain_keys:
        projected_layouts = [row[3] for row in ranked_candidates[:max(0, layout_candidates_top_k)]]
    else:
        projected_layouts = [row[2] for row in ranked_candidates[:max(0, layout_candidates_top_k)]]

    transformed = {
        "artifact": str(root.get("artifact") or "A12"),
        "purpose": str(root.get("purpose") or "table_layout_candidates"),
        "layout_candidates": projected_layouts,
    }
    if "version" in root:
        transformed["version"] = copy.deepcopy(root.get("version"))

    report.setdefault("transform_ops_applied", []).append({
        "name": str(op.get("name") or ""),
        "type": "project_a12_table_layout_grounding",
        "input_layout_candidates": len(root.get("layout_candidates") or []),
        "output_layout_candidates": len(projected_layouts),
        "preferred_primary_grain_keys": preferred_grain_keys,
    })
    return transformed


def _apply_transform_stage(
    node: Any,
    artifact_id: str,
    transforms: Dict[str, Any],
    report: Dict[str, Any],
    value_filter: Optional[Any] = None,
) -> Any:
    if not isinstance(node, (dict, list)):
        return node
    artifact_transforms = _normalize_transform_ops((transforms.get(artifact_id) or {}) if isinstance(transforms, dict) else {})
    for transform_name in sorted(artifact_transforms.keys()):
        op = artifact_transforms.get(transform_name)
        if not isinstance(op, dict):
            continue
        op = dict(op)
        op["name"] = transform_name
        op_type = str(op.get("type", "")).strip().lower()
        if op_type == "join_array_by_key":
            _op_join_array_by_key(node, op, report)
        elif op_type == "derive_fields":
            _op_derive_fields(node, op, report)
        elif op_type == "slice_preview":
            _op_slice_preview(node, op, report)
        elif op_type == "compact_keyed_string_arrays":
            _op_compact_keyed_string_arrays(node, op, report)
        elif op_type == "derive_role_candidate_preview":
            _op_derive_role_candidate_preview(node, op, report)
        elif op_type == "drop_fields_when_empty":
            _op_drop_fields_when_empty(node, op, report)
        elif op_type == "project_a2_semantic_context_grounding":
            node = _op_project_a2_semantic_context_grounding(node, op, report, value_filter)
        elif op_type == "project_a12_table_layout_grounding":
            node = _op_project_a12_table_layout_grounding(node, op, report, value_filter)
    return node


def _build_policies_by_target(artifact_policies: Dict[str, Any]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    by_target: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for policy_name, policy in artifact_policies.items():
        if not isinstance(policy, dict):
            continue
        target = _resolve_policy_target_path(policy_name, policy)
        if target is None:
            continue
        by_target[target] = (policy_name, policy)
    return by_target

def _apply_limits(
    node: Any,    path: str,
    artifact_id: str,
    effective_limits: Dict[str, int],
    policies: Dict[str, Any],
    value_filter: Optional[Any],
    report: Dict[str, Any],
    root_node: Optional[Dict[str, Any]] = None,
) -> Any:
    if root_node is None and isinstance(node, dict):
        root_node = node

    if isinstance(node, list):
        artifact_policies = (policies.get(artifact_id) or {}) if isinstance(policies, dict) else {}
        policies_by_target = _build_policies_by_target(artifact_policies) if isinstance(artifact_policies, dict) else {}
        next_source = node
        if path in policies_by_target:
            policy_name, policy = policies_by_target[path]
            next_source = _apply_typed_array_policy(
                node=root_node or {},
                items=node,
                policy_name=policy_name,
                policy=policy,
                value_filter=value_filter,
                report=report,
            )
        current_limit = _resolve_limit_for_path(effective_limits, artifact_id, path)
        next_list = [
            _apply_limits(
                item,
                path=f"{path}[]" if path else "[]",
                artifact_id=artifact_id,
                effective_limits=effective_limits,
                policies=policies,
                value_filter=value_filter,
                report=report,
                root_node=root_node,
            )
            for item in next_source
        ]
        if current_limit is not None and len(next_list) > current_limit:
            report["truncated_arrays"].append({"path": path or "[]", "before": len(next_list), "after": current_limit})
            next_list = next_list[:current_limit]
        return next_list

    if isinstance(node, dict):
        next_dict: Dict[str, Any] = {}
        artifact_policies = (policies.get(artifact_id) or {}) if isinstance(policies, dict) else {}
        policies_by_target = _build_policies_by_target(artifact_policies) if isinstance(artifact_policies, dict) else {}
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            if isinstance(value, list) and child_path in policies_by_target:
                policy_name, policy = policies_by_target[child_path]
                next_dict[key] = _apply_typed_array_policy(
                    node=root_node or node,
                    items=value,
                    policy_name=policy_name,
                    policy=policy,
                    value_filter=value_filter,
                    report=report,
                )
                continue
            next_dict[key] = _apply_limits(
                value,                child_path,
                artifact_id,
                effective_limits,
                policies,
                value_filter,
                report,
                root_node=root_node,
            )
        return next_dict

    return node


def apply_llm_pruning(
    payload: Any,
    artifact_id: str,
    mode: str,
    keep_keys: Optional[Iterable[str]],
    drop_keys: Optional[Iterable[str]],
    limits: Optional[Dict[str, int]],
    policy_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    replace_policies: Optional[Iterable[str]] = None,
    transform_overrides: Optional[Dict[str, Any]] = None,
    replace_transforms: Optional[Iterable[str]] = None,
    value_filter: Optional[Any] = None,
    debug: bool = False,
    effective_policies: Optional[Dict[str, Any]] = None,
    effective_transforms: Optional[Dict[str, Any]] = None,
    mode_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    mode_config = mode_config or _resolve_pruning_mode(mode)

    keep_set = set(keep_keys or [])
    tier1_cfg = mode_config.get("tier1", {})
    tier2_cfg = mode_config.get("tier2", {})
    tier3_cfg = mode_config.get("tier3", {})

    drop_rules = list(tier1_cfg.get("global_drop_rules", []) or [])
    drop_rules.extend(tier2_cfg.get("artifact_drop_rules", {}).get(artifact_id, []) or [])
    guarded_rule_keys = {str(rule.get("key")) for rule in drop_rules if rule.get("key")}

    effective_drops = set(tier1_cfg.get("global_drop_keys", []))
    effective_drops.update(tier2_cfg.get("artifact_drop_keys", {}).get(artifact_id, []))
    effective_drops.update(drop_keys or [])
    effective_drops = {k for k in effective_drops if k not in guarded_rule_keys}

    default_limits: Dict[str, int] = dict(tier3_cfg.get("limits", {}))
    policy_clamps: List[Dict[str, Any]] = []
    if limits:
        default_limits.update(_guard_limits_map(limits, "limits", policy_clamps) or {})

    requested_drop_keys = [str(k) for k in (drop_keys or [])]
    report: Dict[str, Any] = {
        "mode": mode,
        "artifact_id": artifact_id,
        "dropped_keys": Counter(),
        "truncated_arrays": [],
        "effective_limits": default_limits,
        "seen_keys": Counter(),
        "kept_key_hits": Counter(),
        "keep_requested": sorted(list(keep_set)),
        "drop_requested": requested_drop_keys,
    }
    resolved_policies = effective_policies or _build_effective_policies(
        base_policies=tier3_cfg.get("policies", {}),
        policy_overrides=policy_overrides,
        replace_policies=replace_policies,
        report=report if debug else None,
        scope_artifact_id=artifact_id,
    )
    if debug and effective_policies is not None:
        report["policy_effective"] = {artifact_id: dict((resolved_policies.get(artifact_id) or {}))}
        report["policy_overrides_received"] = policy_overrides or {}
        report["policy_clamps"] = []
    if debug and policy_clamps:
        report.setdefault("policy_clamps", []).extend(policy_clamps)

    resolved_transforms = effective_transforms or _build_effective_transforms(
        base_transforms=tier3_cfg.get("transforms", {}),
        transform_overrides=transform_overrides,
        replace_transforms=replace_transforms,
        report=report if debug else None,
        scope_artifact_id=artifact_id,
    )
    if debug and effective_transforms is not None:
        report["transform_effective"] = {artifact_id: dict((resolved_transforms.get(artifact_id) or {}))}
        report["transform_overrides_received"] = transform_overrides or {}
        replace_set = set(replace_transforms or [])
        report["replace_transforms_applied"] = sorted([
            key for key in (transform_overrides or {}).keys() if key in replace_set
        ])

    if mode == "raw" and not drop_keys and not keep_keys and not limits and not value_filter and not policy_overrides and not transform_overrides:
        report["drop_requested_not_found"] = [k for k in requested_drop_keys if report["seen_keys"].get(k, 0) == 0]
        report["dropped_keys"] = dict(report["dropped_keys"])
        report["kept_key_hits"] = dict(report["kept_key_hits"])
        report.pop("seen_keys", None)
        return payload, report

    working_payload = copy.deepcopy(payload)
    transformed = _apply_transform_stage(
        node=working_payload,
        artifact_id=artifact_id,
        transforms=resolved_transforms,
        report=report,
        value_filter=value_filter,
    )
    pruned = _recursive_prune_keys(
        transformed,
        key_drops=effective_drops,
        keep_keys=keep_set,
        drop_rules=drop_rules,
        report=report,
        path="",
    )

    pruned = _apply_limits(
        pruned,
        path="",
        artifact_id=artifact_id,
        effective_limits=default_limits,
        policies=resolved_policies,
        value_filter=value_filter,
        report=report,
        root_node=pruned if isinstance(pruned, dict) else None,
    )

    report["drop_requested_not_found"] = [k for k in requested_drop_keys if report["seen_keys"].get(k, 0) == 0]
    report["dropped_keys"] = dict(report["dropped_keys"])
    report["kept_key_hits"] = dict(report["kept_key_hits"])
    report.pop("seen_keys", None)
    if not debug:
        report = {
            "mode": report["mode"],
            "artifact_id": report["artifact_id"],
            "dropped_key_count": sum(report["dropped_keys"].values()),
            "truncation_count": len(report["truncated_arrays"]),
        }
    return pruned, report


def _run_pruning_smoke_checks() -> None:
    mode_cfg = _resolve_pruning_mode("llm_baseline")
    assert mode_cfg["tier3"]["limits"].get("A6.tests") == 15
    assert isinstance(mode_cfg["tier3"].get("transforms", {}).get("A8"), dict)

    sample_a9 = {
        "columns": [            {"column": "z_measure", "primary_role": "measure"},
            {"column": "id_a", "primary_role": "id_key"},
            {"column": "id_b", "primary_role": "id_key"},
            {"column": "time_1", "primary_role": "time_index"},
            {"column": "time_1", "primary_role": "time_index"},
        ]
    }
    pruned_a9, _ = apply_llm_pruning(
        payload=sample_a9,
        artifact_id="A9",
        mode="llm_baseline",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
    )
    selected_names = [c.get("column") for c in pruned_a9.get("columns", []) if isinstance(c, dict)]
    assert "id_a" in selected_names and "id_b" in selected_names

    sample_a6 = {
        "row_grain_assessment": {"best_candidate": {"keys": ["order_id"]}},
        "tests": [
            {"keys_tested": ["order_id"], "pivot_safety": 0.5, "collision_severity_score": 0.1, "non_key_conflict_group_pct": 0.1, "uniqueness_rate": 0.8},
            {"keys_tested": ["customer_id"], "pivot_safety": 0.9, "collision_severity_score": 0.5, "non_key_conflict_group_pct": 0.2, "uniqueness_rate": 0.6},
        ],
    }
    pruned_a6, _ = apply_llm_pruning(
        payload=sample_a6,
        artifact_id="A6",
        mode="llm_baseline",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
    )
    first_test = (pruned_a6.get("tests") or [{}])[0]
    assert _normalize_keys_tested(first_test.get("keys_tested")) == ("order_id",)

    sample_a8 = {
        "families": [
            {
                "family_id": "f1",
                "stem_evidence": {"normalized_stem": "metric", "raw_stem": "Metric"},
                "patterns": ["suffix_keyword_numeric"],
                "columns_preview": ["metric_q1", "metric_q2", "metric_q3", "metric_q4"],
                "columns_count": 4,
            }
        ],
        "families_index": [
            {"family_id": "f1", "repeat_dim": "quarter", "patterns": ["suffix_keyword_numeric"]}
        ],
    }
    pruned_a8, _ = apply_llm_pruning(
        payload=sample_a8,
        artifact_id="A8",
        mode="llm_baseline",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
    )
    first_family = (pruned_a8.get("families_index") or [{}])[0]
    assert isinstance(first_family.get("family_signature"), dict)
    assert "families" not in pruned_a8

    grain_profile, profile_source = _load_profile_local("grain_worker")
    assert profile_source == "local"
    assert grain_profile["profile"] == "grain_worker"
    assert grain_profile["artifacts"] == ["A5", "A6", "A7", "A8", "A9", "A10"]

    profile_cfg = grain_profile["mode_config"]
    assert "inputs" in profile_cfg["tier2"]["artifact_drop_keys"]["A5"]

    pruned_a9_profile, _ = apply_llm_pruning(
        payload=sample_a9,
        artifact_id="A9",
        mode="grain_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=profile_cfg,
    )
    selected_names_profile = [c.get("column") for c in pruned_a9_profile.get("columns", []) if isinstance(c, dict)]
    assert "id_a" in selected_names_profile and "id_b" in selected_names_profile

    pruned_a8_profile, _ = apply_llm_pruning(
        payload=sample_a8,
        artifact_id="A8",
        mode="grain_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=profile_cfg,
    )
    profile_first_family = (pruned_a8_profile.get("families_index") or [{}])[0]
    assert isinstance(profile_first_family.get("family_signature"), dict)

    pruned_a6_override, _ = apply_llm_pruning(
        payload=sample_a6,
        artifact_id="A6",
        mode="grain_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        policy_overrides={"A6.tests_policy": {"k": 1}},
        value_filter=None,
        debug=True,
        mode_config=profile_cfg,
    )
    assert len(pruned_a6_override.get("tests", [])) == 1

    try:
        _load_profile_local("missing_profile")
        raise AssertionError("Expected missing profile to raise HTTPException")
    except HTTPException as exc:
        assert exc.status_code == 404

    mode_cfg_bundle, mode_meta_bundle = _resolve_mode_config("grain_worker")
    assert mode_meta_bundle["profile_artifacts"] == ["A5", "A6", "A7", "A8", "A9", "A10"]
    assert mode_cfg_bundle == profile_cfg
    assert parse_csv_list("A5,A8") == ["A5", "A8"]

    skip_df = pd.DataFrame({
        "Q1_Screening": ["No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes"],
        "Q2_Freq": [None, None, None, None, None, "Weekly", "Monthly", "Daily", "Rarely", "Weekly"],
        "Q3_Type": [None, None, None, None, None, "A", "B", "C", "A", "B"],
        "Noise": [None, "x", None, "y", None, "m", None, "n", None, "p"],
    })
    skip_cols_profile = {
        "Q1_Screening": {"missing_pct": 0.0, "unique_count": 2},
        "Q2_Freq": {"missing_pct": 50.0, "unique_count": 5},
        "Q3_Type": {"missing_pct": 50.0, "unique_count": 3},
        "Noise": {"missing_pct": 50.0, "unique_count": 4},
    }
    skip_signal_map = {
        "Q1_Screening": {"missing_pct": 0.0, "unique_count": 2},
        "Q2_Freq": {"missing_pct": 50.0, "unique_count": 5},
        "Q3_Type": {"missing_pct": 50.0, "unique_count": 3},
        "Noise": {"missing_pct": 50.0, "unique_count": 4},
    }
    skip_artifact = _build_conditional_missingness_artifact(
        df=skip_df,
        cols_profile=skip_cols_profile,
        column_signal_map=skip_signal_map,
        artifact_inputs={"A16": {"uses": ["dataset", "A4", "cols_profile", "column_signal_map"]}},
    )
    rules = skip_artifact.get("detected_skip_logic", [])
    matched = next(
        (
            rule for rule in rules
            if rule.get("trigger_column") == "Q1_Screening" and rule.get("trigger_value") == "No"
        ),
        None,
    )
    assert matched is not None
    assert matched.get("affected_column_count") == 2
    assert sorted(matched.get("sample_affected_columns", [])) == ["Q2_Freq", "Q3_Type"]
    assert matched.get("rule_strength") == 1.0
    assert matched.get("directionality") == "bidirectional"
    assert matched.get("missing_explained_pct") == 100.0
    master_switches = skip_artifact.get("master_switch_candidates", [])
    assert master_switches
    assert master_switches[0].get("trigger_column") == "Q1_Screening"
    assert master_switches[0].get("explained_column_count") == 2
    assert skip_artifact.get("audit_assumptions", {}).get("family_member_triggers_excluded_from_master_switch_candidates") is True

    type_transform_profile, type_transform_source = _load_profile_local("type_transform_worker")
    assert type_transform_source == "local"
    assert "A16" in type_transform_profile["artifacts"]
    type_transform_cfg = type_transform_profile["mode_config"]

    missingness_profile, missingness_source = _load_profile_local("missingness_worker")
    assert missingness_source == "local"
    assert missingness_profile["artifacts"] == ["A2", "A4", "A13", "A14", "A16"]
    missingness_cfg = missingness_profile["mode_config"]

    semantic_context_profile, semantic_context_source = _load_profile_local("semantic_context_worker")
    assert semantic_context_source == "local"
    assert semantic_context_profile["artifacts"] == ["A2", "A8", "A9", "A16"]
    semantic_context_cfg = semantic_context_profile["mode_config"]

    family_worker_profile, family_worker_source = _load_profile_local("family_worker")
    assert family_worker_source == "local"
    assert family_worker_profile["artifacts"] == ["A8", "B1"]
    family_worker_cfg = family_worker_profile["mode_config"]

    table_layout_profile, table_layout_source = _load_profile_local("table_layout_worker")
    assert table_layout_source == "local"
    assert table_layout_profile["artifacts"] == ["A2", "A5", "A9", "A10", "A12", "A14"]
    table_layout_cfg = table_layout_profile["mode_config"]

    analysis_layout_profile, analysis_layout_source = _load_profile_local("analysis_layout_worker")
    assert analysis_layout_source == "local"
    assert analysis_layout_profile["artifacts"] == ["A2", "A8", "A10", "A14", "A16", "B1"]
    analysis_layout_cfg = analysis_layout_profile["mode_config"]

    canonical_reviewer_profile, canonical_reviewer_source = _load_profile_local("canonical_contract_reviewer")
    assert canonical_reviewer_source == "local"
    assert canonical_reviewer_profile["artifacts"] == ["A2", "A3-T", "A3-V", "A4", "A9", "A13", "A14", "A16", "A17"]
    canonical_reviewer_cfg = canonical_reviewer_profile["mode_config"]

    mode_cfg_missingness, mode_meta_missingness = _resolve_mode_config("missingness_worker")
    assert mode_meta_missingness["profile_artifacts"] == ["A2", "A4", "A13", "A14", "A16"]
    assert mode_cfg_missingness == missingness_cfg

    mode_cfg_semantic, mode_meta_semantic = _resolve_mode_config("semantic_context_worker")
    assert mode_meta_semantic["profile_artifacts"] == ["A2", "A8", "A9", "A16"]
    assert mode_cfg_semantic == semantic_context_cfg

    mode_cfg_family, mode_meta_family = _resolve_mode_config("family_worker")
    assert mode_meta_family["profile_artifacts"] == ["A8", "B1"]
    assert mode_cfg_family == family_worker_cfg

    mode_cfg_table_layout, mode_meta_table_layout = _resolve_mode_config("table_layout_worker")
    assert mode_meta_table_layout["profile_artifacts"] == ["A2", "A5", "A9", "A10", "A12", "A14"]
    assert mode_cfg_table_layout == table_layout_cfg

    mode_cfg_analysis_layout, mode_meta_analysis_layout = _resolve_mode_config("analysis_layout_worker")
    assert mode_meta_analysis_layout["profile_artifacts"] == ["A2", "A8", "A10", "A14", "A16", "B1"]
    assert mode_cfg_analysis_layout == analysis_layout_cfg

    mode_cfg_canonical_reviewer, mode_meta_canonical_reviewer = _resolve_mode_config("canonical_contract_reviewer")
    assert mode_meta_canonical_reviewer["profile_artifacts"] == ["A2", "A3-T", "A3-V", "A4", "A9", "A13", "A14", "A16", "A17"]
    assert mode_cfg_canonical_reviewer == canonical_reviewer_cfg

    sample_a2_type = [
        {
            "column": "force_col",
            "high_missingness": False,
            "is_one_hot_like": False,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["001", "002"], "tail": ["003"], "random": ["004"]},
            "profiler_samples": {"head": ["too_much"]},
        },
        {
            "column": "wide_flag",
            "high_missingness": True,
            "is_one_hot_like": False,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["A", "B", "C", "D"]},
        },
        {
            "column": "drop_me",
            "high_missingness": False,
            "is_one_hot_like": False,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["x"]},
        },
    ]
    pruned_a2_type, _ = apply_llm_pruning(
        payload=sample_a2_type,
        artifact_id="A2",
        mode="type_transform_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={"force_include_columns": ["force_col"]},
        debug=True,
        mode_config=type_transform_cfg,
    )
    a2_names = [row.get("column") for row in pruned_a2_type if isinstance(row, dict)]
    assert a2_names == ["force_col", "wide_flag"]
    assert "profiler_samples" not in pruned_a2_type[0]
    assert pruned_a2_type[0].get("a2_samples", {}).get("head") == ["001", "002"]

    sample_a3t_type = {
        "items": [
            {
                "column": "consent_text",
                "evidence_snippets": {
                    "parse_failure_examples": ["a" * 140, "b" * 50, "c" * 20],
                    "multi_value_examples": ["x", "y", "z"],
                },
            }
        ]
    }
    pruned_a3t_type, _ = apply_llm_pruning(
        payload=sample_a3t_type,
        artifact_id="A3-T",
        mode="type_transform_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=type_transform_cfg,
    )
    evidence = pruned_a3t_type.get("items", [{}])[0].get("evidence_snippets", {})
    assert len(evidence.get("parse_failure_examples", [])) == 2
    assert evidence.get("parse_failure_examples", [""])[0].endswith("...")
    assert len(evidence.get("multi_value_examples", [])) == 2

    sample_a9_type = {
        "columns": [
            {
                "column": "id_col",
                "primary_role": "id_key",
                "encoding_type": "numeric",
                "role_decision": "keep",
                "review_required": False,
                "review_reasons": [],
                "role_candidates": [
                    {"role": "id_key", "score": 1.0},
                    {"role": "measure", "score": 0.4},
                    {"role": "repeat_index", "score": 0.2},
                ],
                "role_scores": {"id_key": 1.0, "measure": 0.4},
                "audit": {"details": True},
            }
        ]
    }
    pruned_a9_type, _ = apply_llm_pruning(
        payload=sample_a9_type,
        artifact_id="A9",
        mode="type_transform_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=type_transform_cfg,
    )
    a9_row = (pruned_a9_type.get("columns") or [{}])[0]
    assert a9_row.get("role_candidate_preview") == [
        {"role": "id_key", "score": 1.0},
        {"role": "measure", "score": 0.4},
    ]
    assert "role_decision" not in a9_row
    assert "role_scores" not in a9_row

    sample_a14_type = {
        "columns": [
            {
                "column": "force_col",
                "global_quality_score": 1.0,
                "drift_detected": False,
                "segments": [{"row_range": [0, 10]}],
                "interpretation": "Stable",
            },
            {
                "column": "drift_col",
                "global_quality_score": 0.95,
                "drift_detected": True,
                "segments": [{"row_range": [0, 10]}],
                "interpretation": "Drift",
            },
        ]
    }
    pruned_a14_type, _ = apply_llm_pruning(
        payload=sample_a14_type,
        artifact_id="A14",
        mode="type_transform_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={"force_include_columns": ["force_col"]},
        debug=True,
        mode_config=type_transform_cfg,
    )
    a14_names = [row.get("column") for row in pruned_a14_type.get("columns", []) if isinstance(row, dict)]
    assert a14_names == ["force_col", "drift_col"]
    assert all("segments" not in row for row in pruned_a14_type.get("columns", []) if isinstance(row, dict))

    sample_a16_type = {
        "audit_trace": {
            "target_columns_scanned": ["a", "b"],
            "trigger_columns_scanned": ["c"],
            "rows_evaluated": 100,
        }
    }
    pruned_a16_type, _ = apply_llm_pruning(
        payload=sample_a16_type,
        artifact_id="A16",
        mode="type_transform_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=type_transform_cfg,
    )
    assert "target_columns_scanned" not in (pruned_a16_type.get("audit_trace") or {})
    assert "trigger_columns_scanned" not in (pruned_a16_type.get("audit_trace") or {})

    sample_a2_missingness = [
        {
            "column": "force_col",
            "high_missingness": False,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["A", "B", "C"], "tail": ["D"], "random": ["E"]},
        },
        {
            "column": "skip_target",
            "high_missingness": False,
            "missing_tokens_observed": {"token": ["n/a"]},
            "a2_samples": {"head": ["n/a", "x"]},
        },
        {
            "column": "wide_null_col",
            "high_missingness": True,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["", "", "z"]},
        },
        {
            "column": "drop_me",
            "high_missingness": False,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["x"]},
        },
    ]
    pruned_a2_missingness, _ = apply_llm_pruning(
        payload=sample_a2_missingness,
        artifact_id="A2",
        mode="missingness_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={
            "force_include_columns": ["force_col"],
            "skip_affected_preview_columns": ["skip_target"],
        },
        debug=True,
        mode_config=missingness_cfg,
    )
    a2_missing_names = [row.get("column") for row in pruned_a2_missingness if isinstance(row, dict)]
    assert a2_missing_names == ["force_col", "skip_target", "wide_null_col"]
    assert "missing_tokens_observed" not in pruned_a2_missingness[0]
    assert pruned_a2_missingness[0].get("a2_samples", {}).get("head") == ["A", "B"]

    sample_a4_missingness = {
        "summary": {"columns_with_any_missing": 3},
        "global_missingness": {"true_na_total": 5},
        "global_tokens": {"na": 2},
        "token_classification": {"na": "missing_like"},
        "per_column": [
            {
                "column": "force_col",
                "missing_pct": 5.0,
                "token_breakdown": {},
            },
            {
                "column": "high_missing_col",
                "missing_pct": 45.0,
                "token_breakdown": {},
            },
            {
                "column": "token_col",
                "missing_pct": 8.0,
                "token_breakdown": {"n/a": 4},
            },
            {
                "column": "drop_me",
                "missing_pct": 2.0,
                "token_breakdown": {},
            },
        ],
        "audit_trace": {"classification_rules_version": "v2"},
    }
    pruned_a4_missingness, _ = apply_llm_pruning(
        payload=sample_a4_missingness,
        artifact_id="A4",
        mode="missingness_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={"force_include_columns": ["force_col"]},
        debug=True,
        mode_config=missingness_cfg,
    )
    a4_missing_names = [row.get("column") for row in pruned_a4_missingness.get("per_column", []) if isinstance(row, dict)]
    assert a4_missing_names == ["force_col", "high_missing_col", "token_col"]
    assert "global_tokens" not in pruned_a4_missingness
    assert "token_classification" not in pruned_a4_missingness
    assert "token_breakdown" not in pruned_a4_missingness.get("per_column", [])[0]

    sample_a16_missingness = {
        "inputs": {"uses": ["dataset"]},
        "detected_skip_logic": [{"trigger_column": f"Trig{i}", "sample_affected_columns": ["c1", "c2"]} for i in range(60)],
        "master_switch_candidates": [{"trigger_column": f"Master{i}"} for i in range(20)],
        "audit_assumptions": {"family_member_triggers_excluded_from_master_switch_candidates": True},
        "audit_trace": {
            "target_columns_scanned": ["a", "b"],
            "trigger_columns_scanned": ["c"],
            "rows_evaluated": 100,
        },
    }
    pruned_a16_missingness, _ = apply_llm_pruning(
        payload=sample_a16_missingness,
        artifact_id="A16",
        mode="missingness_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=missingness_cfg,
    )
    assert len(pruned_a16_missingness.get("detected_skip_logic", [])) == 50
    assert len(pruned_a16_missingness.get("master_switch_candidates", [])) == 15
    assert "inputs" not in pruned_a16_missingness
    assert "target_columns_scanned" not in (pruned_a16_missingness.get("audit_trace") or {})
    assert "trigger_columns_scanned" not in (pruned_a16_missingness.get("audit_trace") or {})

    sample_a2_semantic = [
        {
            "column": "response_id",
            "top_candidate": {"type": "string"},
            "unique_count": 100,
            "missing_pct": 0.0,
            "high_uniqueness_candidate": True,
            "top_levels": ["r1", "r2"],
            "missing_tokens_observed": {},
            "is_one_hot_like": False,
            "profiler_samples": {"head": ["r1"]},
            "a2_samples": {"head": ["r1"]},
        },
        {
            "column": "gender_code",
            "top_candidate": {"type": "integer"},
            "unique_count": 2,
            "missing_pct": 0.0,
            "high_uniqueness_candidate": False,
            "top_levels": ["0", "1", "2"],
            "missing_tokens_observed": {},
            "is_one_hot_like": False,
        },
        {
            "column": "screen_flag",
            "top_candidate": {"type": "categorical"},
            "unique_count": 2,
            "missing_pct": 5.0,
            "high_uniqueness_candidate": False,
            "top_levels": ["Yes", "No"],
            "missing_tokens_observed": {"UNK": 1},
            "is_one_hot_like": True,
        },
        {
            "column": "large_card_text",
            "top_candidate": {"type": "text"},
            "unique_count": 30,
            "missing_pct": 0.0,
            "high_uniqueness_candidate": False,
            "top_levels": ["a", "b"],
            "missing_tokens_observed": {},
            "is_one_hot_like": False,
        },
    ]
    pruned_a2_semantic, _ = apply_llm_pruning(
        payload=sample_a2_semantic,
        artifact_id="A2",
        mode="semantic_context_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={
            "skip_trigger_columns": ["screen_flag"],
            "force_include_columns": ["response_id"],
        },
        debug=True,
        mode_config=semantic_context_cfg,
    )
    assert pruned_a2_semantic.get("artifact") == "A2"
    semantic_columns = [row.get("column") for row in pruned_a2_semantic.get("columns_index", []) if isinstance(row, dict)]
    assert semantic_columns == ["response_id", "gender_code", "screen_flag", "large_card_text"]
    low_card_columns = [row.get("column") for row in pruned_a2_semantic.get("low_cardinality_value_preview", []) if isinstance(row, dict)]
    assert low_card_columns == ["screen_flag", "gender_code"]
    assert "large_card_text" not in low_card_columns
    assert "response_id" not in low_card_columns
    gender_preview = next(row for row in pruned_a2_semantic.get("low_cardinality_value_preview", []) if row.get("column") == "gender_code")
    assert gender_preview.get("top_levels") == ["0", "1", "2"]

    pruned_a8_semantic, _ = apply_llm_pruning(
        payload=sample_a8,
        artifact_id="A8",
        mode="semantic_context_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=semantic_context_cfg,
    )
    semantic_family = (pruned_a8_semantic.get("families_index") or [{}])[0]
    assert isinstance(semantic_family.get("family_signature"), dict)
    assert "families" not in pruned_a8_semantic

    pruned_a9_semantic, _ = apply_llm_pruning(
        payload=sample_a9_type,
        artifact_id="A9",
        mode="semantic_context_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=semantic_context_cfg,
    )
    semantic_a9_row = (pruned_a9_semantic.get("columns") or [{}])[0]
    assert semantic_a9_row.get("role_candidate_preview") == [
        {"role": "id_key", "score": 1.0},
        {"role": "measure", "score": 0.4},
    ]
    assert "role_scores" not in semantic_a9_row

    pruned_a16_semantic, _ = apply_llm_pruning(
        payload=sample_a16_missingness,
        artifact_id="A16",
        mode="semantic_context_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=semantic_context_cfg,
    )
    assert len(pruned_a16_semantic.get("detected_skip_logic", [])) == 40
    assert len(pruned_a16_semantic.get("master_switch_candidates", [])) == 12
    assert "inputs" not in pruned_a16_semantic

    sample_a8_family = {
        "families": [
            {
                "family_id": "fam_a",
                "stem_evidence": {"normalized_stem": "a", "raw_stem": "A"},
                "patterns": ["suffix_keyword_numeric"],
                "columns_preview": ["A_1", "A_2", "A_3"],
                "columns_count": 3,
            },
            {
                "family_id": "fam_b",
                "stem_evidence": {"normalized_stem": "b", "raw_stem": "B"},
                "patterns": ["suffix_keyword_numeric"],
                "columns_preview": ["B_1", "B_2", "B_3"],
                "columns_count": 3,
            },
        ],
        "families_index": [
            {"family_id": "fam_a", "repeat_dim": "wave", "patterns": ["suffix_keyword_numeric"]},
            {"family_id": "fam_b", "repeat_dim": "wave", "patterns": ["suffix_keyword_numeric"]},
        ],
        "near_misses": [{"family_id": "miss"}],
        "rejected_candidates": [{"family_id": "rej"}],
    }
    pruned_a8_family, _ = apply_llm_pruning(
        payload=sample_a8_family,
        artifact_id="A8",
        mode="family_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={"force_include_family_ids": ["fam_b"]},
        debug=True,
        mode_config=family_worker_cfg,
    )
    family_ids = [row.get("family_id") for row in pruned_a8_family.get("families_index", []) if isinstance(row, dict)]
    assert family_ids == ["fam_b"]
    family_sig = (pruned_a8_family.get("families_index") or [{}])[0].get("family_signature")
    assert isinstance(family_sig, dict)
    assert "families" not in pruned_a8_family
    assert "near_misses" not in pruned_a8_family
    assert "rejected_candidates" not in pruned_a8_family

    sample_b1_family = [
        {
            "inputs": {"uses": ["A8"]},
            "family_id": "fam_a",
            "columns": ["A_1", "A_2"],
            "detected_pattern_index_summary": {"patterns": ["suffix_keyword_numeric"]},
            "family_summary": {"avg_missing_pct": 10.0},
            "relational_context": {"global_grain": ["id"]},
            "peer_signature": [{"peer_id": "fam_b", "confidence": 0.8}],
            "evidence_subset": {"A2_signals": [{"column": "A_1"}]},
            "C_subset": {
                "repeat_candidate": {
                    "family_id": "fam_a",
                    "columns": ["A_1", "A_2"],
                    "index_by_column": {"A_1": ["1"], "A_2": ["2"]},
                },
                "grain_collisions_touching_family": [{"keys_tested": ["id", "A_1"]}],
            },
            "B_subset": [{"column": "A_1"}],
            "F_subset": {"rows": [{"A_1": "x"}]},
        },
        {
            "inputs": {"uses": ["A8"]},
            "family_id": "fam_b",
            "columns": ["B_1", "B_2"],
            "detected_pattern_index_summary": {"patterns": ["suffix_keyword_numeric"]},
            "family_summary": {"avg_missing_pct": 15.0},
            "relational_context": {"global_grain": ["id"]},
            "peer_signature": [{"peer_id": "fam_a", "confidence": 0.7}],
            "evidence_subset": {"A2_signals": [{"column": "B_1"}]},
            "C_subset": {
                "repeat_candidate": {
                    "family_id": "fam_b",
                    "columns": ["B_1", "B_2"],
                    "index_by_column": {"B_1": ["1"], "B_2": ["2"]},
                },
                "grain_collisions_touching_family": [{"keys_tested": ["id", "B_1"]}],
            },
            "B_subset": [{"column": "B_1"}],
            "F_subset": {"rows": [{"B_1": "y"}]},
        },
    ]
    pruned_b1_family, _ = apply_llm_pruning(
        payload=sample_b1_family,
        artifact_id="B1",
        mode="family_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={"force_include_family_ids": ["fam_b"]},
        debug=True,
        mode_config=family_worker_cfg,
    )
    assert isinstance(pruned_b1_family, list)
    assert len(pruned_b1_family) == 1
    assert pruned_b1_family[0].get("family_id") == "fam_b"
    assert "inputs" not in pruned_b1_family[0]
    assert "B_subset" not in pruned_b1_family[0]
    assert "F_subset" not in pruned_b1_family[0]
    c_subset = pruned_b1_family[0].get("C_subset") or {}
    repeat_candidate = c_subset.get("repeat_candidate") or {}
    assert "grain_collisions_touching_family" not in c_subset
    assert "columns" not in repeat_candidate
    assert "index_by_column" not in repeat_candidate

    sample_a2_table_layout = [
        {
            "column": "response_id",
            "high_missingness": False,
            "is_one_hot_like": False,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["r1", "r2", "r3"], "tail": ["r4"]},
            "profiler_samples": {"head": ["too_much"]},
        },
        {
            "column": "one_hot_yes",
            "high_missingness": False,
            "is_one_hot_like": True,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["0", "1", "0"]},
        },
        {
            "column": "null_heavy",
            "high_missingness": True,
            "is_one_hot_like": False,
            "missing_tokens_observed": {},
            "a2_samples": {"head": ["", "", "x"]},
        },
        {
            "column": "drop_me",
            "high_missingness": False,
            "is_one_hot_like": False,
            "missing_tokens_observed": [],
            "a2_samples": {"head": ["x"]},
        },
    ]
    pruned_a2_table_layout, _ = apply_llm_pruning(
        payload=sample_a2_table_layout,
        artifact_id="A2",
        mode="table_layout_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={"force_include_columns": ["response_id"]},
        debug=True,
        mode_config=table_layout_cfg,
    )
    assert pruned_a2_table_layout.get("artifact") == "A2"
    a2_table_layout_names = [
        row.get("column")
        for row in pruned_a2_table_layout.get("columns_index", [])
        if isinstance(row, dict)
    ]
    assert a2_table_layout_names == ["response_id", "one_hot_yes", "null_heavy", "drop_me"]

    sample_a9_table_layout = {
        "columns": [
            {
                "column": "id_col",
                "primary_role": "id_key",
                "review_required": False,
                "role_candidates": [
                    {"role": "id_key", "score": 1.0},
                    {"role": "measure", "score": 0.2},
                ],
                "role_scores": {"id_key": 1.0},
            },
            {
                "column": "measure_col",
                "primary_role": "measure_numeric",
                "review_required": False,
                "role_candidates": [
                    {"role": "measure_numeric", "score": 0.95},
                    {"role": "coded_categorical", "score": 0.2},
                ],
                "role_scores": {"measure_numeric": 0.95},
            },
            {
                "column": "cat_col",
                "primary_role": "coded_categorical",
                "review_required": False,
                "role_candidates": [
                    {"role": "coded_categorical", "score": 0.91},
                    {"role": "invariant_attr", "score": 0.5},
                ],
                "role_scores": {"coded_categorical": 0.91},
            },
            {
                "column": "review_col",
                "primary_role": "unknown",
                "review_required": True,
                "role_candidates": [
                    {"role": "measure_item", "score": 0.6},
                    {"role": "coded_categorical", "score": 0.55},
                ],
                "role_scores": {"measure_item": 0.6},
            },
        ]
    }
    pruned_a9_table_layout, _ = apply_llm_pruning(
        payload=sample_a9_table_layout,
        artifact_id="A9",
        mode="table_layout_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=table_layout_cfg,
    )
    table_layout_role_names = [row.get("column") for row in pruned_a9_table_layout.get("columns", []) if isinstance(row, dict)]
    assert set(table_layout_role_names) == {"id_col", "measure_col", "cat_col", "review_col"}
    assert all("role_scores" not in row for row in pruned_a9_table_layout.get("columns", []) if isinstance(row, dict))
    assert all(isinstance(row.get("role_candidate_preview"), list) for row in pruned_a9_table_layout.get("columns", []) if isinstance(row, dict))

    sample_a10_table_layout = {
        "inputs": {"uses": ["dataset"]},
        "derived_total_candidates": 8,
        "near_duplicate_candidates": [{"a": "b"}],
        "dependency_candidates": [{"determinant": f"id_{i}", "dependent": f"col_{i}", "coverage": 0.9 - i * 0.01} for i in range(25)],
        "time_column_candidates": [{"column": f"dt_{i}", "score": 0.8 - i * 0.02} for i in range(12)],
        "family_screening_correlations": [{"family_id": f"fam_{i}", "trigger_column": f"screen_{i}"} for i in range(12)],
        "one_hot_blocks": [{"group": "g", "columns": ["a", "b"]}],
        "coded_categorical_flags": [{"column": "x"}],
        "audit_trail_signals": {"included_count": 5},
        "deterministic_constraints": {"arithmetic": []},
    }
    pruned_a10_table_layout, _ = apply_llm_pruning(
        payload=sample_a10_table_layout,
        artifact_id="A10",
        mode="table_layout_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=table_layout_cfg,
    )
    assert "inputs" not in pruned_a10_table_layout
    assert "near_duplicate_candidates" not in pruned_a10_table_layout
    assert "one_hot_blocks" not in pruned_a10_table_layout
    assert "coded_categorical_flags" not in pruned_a10_table_layout
    assert "audit_trail_signals" not in pruned_a10_table_layout
    assert "deterministic_constraints" not in pruned_a10_table_layout
    assert len(pruned_a10_table_layout.get("dependency_candidates", [])) == 20
    assert len(pruned_a10_table_layout.get("time_column_candidates", [])) == 10
    assert len(pruned_a10_table_layout.get("family_screening_correlations", [])) == 10

    sample_a12_table_layout = {
        "artifact": "A12",
        "purpose": "table_layout_candidates",
        "version": "2",
        "layout_candidates": [
            {
                "layout_id": "L_wrong",
                "score": 0.95,
                "summary": {"proposed_model": "base_plus_child"},
                "grain": {"keys_tested": ["final_grade", "a1_q10"]},
                "tables": [
                    {
                        "table_id": "T1",
                        "table_name": "base_a",
                        "kind": "entity",
                        "grain": ["final_grade", "a1_q10"],
                        "repeat_dimension_candidates": [],
                        "column_counts": {"keys": 2, "attributes": 5, "measures": 0},
                        "foreign_keys": [],
                        "primary_role_assignments_preview": {"id": {"primary_role": "id_key"}},
                        "family_reference": {
                            "family_id": "fam_a",
                            "detection_confidence": 0.99,
                            "columns": ["q1", "q2", "q3"],
                            "extracted_index_set": ["1", "2", "3"],
                            "index_by_column": {"q1": ["1"]},
                            "recommended_repeat_dimension_name": "q",
                            "flags": {"suspected_item": True},
                        },
                    }
                ],
                "coverage_summary": {
                    "covered_by_layout_count": 5,
                    "covered_by_role_only_count": 2,
                    "ambiguous_count": 2,
                },
                "column_placement": {
                    "covered_by_layout": ["id", "q1", "q2", "q3"],
                    "covered_by_role_only": ["extra_1", "extra_2"],
                    "ambiguous": ["maybe_a", "maybe_b"],
                    "unmapped": ["mystery_1", "mystery_2"],
                },
                "ambiguities": [
                    {
                        "column": "q1",
                        "primary_role": "measure",
                        "alternate_roles": [
                            {"role": "invariant_attr", "score": 0.94},
                            {"role": "measure_item", "score": 0.90},
                        ],
                        "why": "should be dropped",
                    },
                    {"column": "q2", "primary_role": "measure"},
                    {"column": "q3", "primary_role": "measure"},
                ],
                "debug_trace": {"score": 0.95},
            },
            {
                "layout_id": "L_id",
                "score": 0.80,
                "summary": {"proposed_model": "base"},
                "grain": {"keys_tested": ["id"]},
                "tables": [
                    {
                        "table_id": "T2",
                        "table_name": "base_b",
                        "kind": "entity",
                        "grain": ["id"],
                        "repeat_dimension_candidates": [],
                        "column_counts": {"keys": 1, "attributes": 6, "measures": 0},
                        "foreign_keys": [],
                    }
                ],
                "coverage_summary": {"covered_by_layout_count": 6},
                "column_placement": {"covered_by_layout": ["id", "q"]},
                "ambiguities": [],
                "debug_trace": {"score": 0.8},
            },
            {
                "layout_id": "L_overlap",
                "score": 0.85,
                "summary": {"proposed_model": "base_plus_dims"},
                "grain": {"keys_tested": ["id", "wave"]},
                "tables": [{"table_name": "base_c"}],
                "coverage_summary": {"covered_by_layout_count": 7},
                "column_placement": {"covered_by_layout": ["id", "country"]},
                "ambiguities": [],
                "debug_trace": {"score": 0.85},
            },
            {
                "layout_id": "L_low",
                "score": 0.60,
                "summary": {"proposed_model": "mixed"},
                "grain": {"keys_tested": ["legacy_id"]},
                "tables": [{"table_name": "base_d"}],
                "coverage_summary": {"covered_by_layout_count": 8},
                "column_placement": {"covered_by_layout": ["id", "x"]},
                "ambiguities": [],
                "debug_trace": {"score": 0.6},
            },
        ],
        "debug": {"candidate_layout_count": 4},
        "inputs": {"uses": ["A5", "A9"]},
        "evidence_primitives": {"A5": "used"},
    }
    pruned_a12_table_layout, _ = apply_llm_pruning(
        payload=sample_a12_table_layout,
        artifact_id="A12",
        mode="table_layout_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={"preferred_primary_grain_keys": ["id"]},
        debug=True,
        mode_config=table_layout_cfg,
    )
    assert len(pruned_a12_table_layout.get("layout_candidates", [])) == 2
    assert [row.get("layout_id") for row in pruned_a12_table_layout.get("layout_candidates", []) if isinstance(row, dict)] == [
        "L_id",
        "L_overlap",
    ]
    assert "debug" not in pruned_a12_table_layout
    assert "inputs" not in pruned_a12_table_layout
    assert "evidence_primitives" not in pruned_a12_table_layout
    assert all("debug_trace" not in row for row in pruned_a12_table_layout.get("layout_candidates", []) if isinstance(row, dict))
    first_layout = pruned_a12_table_layout.get("layout_candidates", [])[0]
    assert first_layout.get("preferred_grain_match") == "exact"
    assert "covered_by_role_only" not in ((first_layout.get("column_placement") or {}) if isinstance(first_layout, dict) else {})
    assert "ambiguous" not in ((first_layout.get("column_placement") or {}) if isinstance(first_layout, dict) else {})
    assert "covered_by_layout" not in ((first_layout.get("column_placement") or {}) if isinstance(first_layout, dict) else {})
    assert len(((first_layout.get("column_placement") or {}).get("covered_by_layout_preview") or [])) <= 12
    assert len(((first_layout.get("column_placement") or {}).get("unmapped_preview") or [])) <= 12
    assert len(first_layout.get("ambiguities", [])) <= 8

    pruned_a12_table_layout_fallback, _ = apply_llm_pruning(
        payload=sample_a12_table_layout,
        artifact_id="A12",
        mode="table_layout_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter=None,
        debug=True,
        mode_config=table_layout_cfg,
    )
    assert [row.get("layout_id") for row in pruned_a12_table_layout_fallback.get("layout_candidates", []) if isinstance(row, dict)] == [
        "L_wrong",
        "L_overlap",
    ]
    fallback_first_layout = pruned_a12_table_layout_fallback.get("layout_candidates", [])[0]
    fallback_first_table = ((fallback_first_layout.get("tables") or [])[0] if isinstance(fallback_first_layout, dict) else {})
    assert "primary_role_assignments_preview" not in fallback_first_table
    family_reference = fallback_first_table.get("family_reference") or {}
    assert "columns" not in family_reference
    assert "extracted_index_set" not in family_reference
    assert "index_by_column" not in family_reference
    assert set(family_reference.keys()) <= {
        "family_id",
        "detection_confidence",
        "columns_count",
        "columns_preview",
        "recommended_repeat_dimension_name",
        "flags",
    }
    fallback_first_ambiguity = (fallback_first_layout.get("ambiguities") or [])[0]
    assert set(fallback_first_ambiguity.keys()) <= {"column", "primary_role", "alternate_roles"}

    sample_a2_table_layout = [
        {
            "column": "id_col",
            "top_candidate": {"type": "numeric"},
            "unique_count": 100,
            "missing_pct": 0.0,
            "high_uniqueness_candidate": True,
            "top_levels": ["1", "2"],
            "missing_tokens_observed": {},
            "is_one_hot_like": False,
        },
        {
            "column": "country_code",
            "top_candidate": {"type": "categorical"},
            "unique_count": 3,
            "missing_pct": 1.0,
            "high_uniqueness_candidate": False,
            "top_levels": ["CA", "US", "GB", "AU", "NZ"],
            "missing_tokens_observed": {},
            "is_one_hot_like": False,
        },
        {
            "column": "screen_flag",
            "top_candidate": {"type": "categorical"},
            "unique_count": 2,
            "missing_pct": 0.0,
            "high_uniqueness_candidate": False,
            "top_levels": ["0", "1", "2"],
            "missing_tokens_observed": {"UNK": 1},
            "is_one_hot_like": False,
        },
    ]
    pruned_a2_table_layout, _ = apply_llm_pruning(
        payload=sample_a2_table_layout,
        artifact_id="A2",
        mode="table_layout_worker",
        keep_keys=[],
        drop_keys=[],
        limits=None,
        value_filter={
            "force_include_columns": ["id_col"],
            "structural_columns": ["country_code"],
            "skip_trigger_columns": ["screen_flag"],
        },
        debug=True,
        mode_config=table_layout_cfg,
    )
    assert pruned_a2_table_layout.get("artifact") == "A2"
    assert [row.get("column") for row in pruned_a2_table_layout.get("columns_index", []) if isinstance(row, dict)] == [
        "id_col",
        "country_code",
        "screen_flag",
    ]
    assert [row.get("column") for row in pruned_a2_table_layout.get("low_cardinality_value_preview", []) if isinstance(row, dict)] == [
        "screen_flag",
        "country_code",
    ]

    ordered_merge = _merge_value_filter_column_bucket(
        {"reviewer_focus_columns": ["Mjr2", "A2_Q1"]},
        "reviewer_focus_columns",
        ["M1_Q14", "A2_Q1", "Q1"],
    )
    assert ordered_merge == {
        "reviewer_focus_columns": ["Mjr2", "A2_Q1", "M1_Q14", "Q1"],
    }

    grouped_selector_items = [
        {"column": "A1_Q1", "flag": False},
        {"column": "A1_Q2", "flag": False},
        {"column": "A1_Q3", "flag": False},
        {"column": "A1_Q4", "flag": True},
        {"column": "A1_Q5", "flag": True},
        {"column": "A1_Q6", "flag": True},
        {"column": "A1_Q7", "flag": True},
    ]
    grouped_selector_policy = {
        "name_field": "column",
        "seed_from_filter_keys": ["reviewer_focus_columns"],
        "include_bool_true_fields": ["flag"],
        "seed_cap_per_inferred_group": 2,
        "standalone_cap_per_inferred_group": 2,
        "group_sample_mode": "all",
        "hard_cap_total": 10,
    }
    grouped_selector_report: Dict[str, Any] = {}
    grouped_selector_pruned = _apply_column_scope_selector_policy(
        grouped_selector_items,
        grouped_selector_policy,
        {
            "force_include_columns": ["A1_Q2"],
            "reviewer_focus_columns": ["A1_Q1", "A1_Q2", "A1_Q3"],
        },
        grouped_selector_report,
    )
    assert [row.get("column") for row in grouped_selector_pruned if isinstance(row, dict)] == [
        "A1_Q2",
        "A1_Q1",
        "A1_Q3",
        "A1_Q4",
        "A1_Q5",
    ]
    grouped_selector_meta = grouped_selector_report.get("column_scope_policies", [])[-1]
    assert grouped_selector_meta.get("forced_output") == 1
    assert grouped_selector_meta.get("seeded_output") == 2
    assert grouped_selector_meta.get("standalone_output") == 2

    representative_selector_items = [
        {"column": "A1_Q1", "quality": 0.70},
        {"column": "A1_Q2", "quality": 0.69},
        {"column": "A1_Q3", "quality": 0.68},
        {"column": "A1_Q4", "quality": 0.67},
        {"column": "A1_Q5", "quality": 0.10},
    ]
    representative_selector_pruned = _apply_column_scope_selector_policy(
        representative_selector_items,
        {
            "name_field": "column",
            "include_numeric_lte": {"quality": 0.7},
            "standalone_cap_per_inferred_group": 3,
            "group_sample_mode": "representative_plus_worst_outlier",
            "hard_cap_total": 10,
        },
        None,
        {},
    )
    assert [row.get("column") for row in representative_selector_pruned if isinstance(row, dict)] == [
        "A1_Q1",
        "A1_Q2",
        "A1_Q5",
    ]

    unique_group_selector_pruned = _apply_column_scope_selector_policy(
        [
            {"column": "Age", "flag": True},
            {"column": "Grad_Country", "flag": True},
            {"column": "ID", "flag": True},
        ],
        {
            "name_field": "column",
            "include_bool_true_fields": ["flag"],
            "standalone_cap_per_inferred_group": 1,
            "group_sample_mode": "representative_plus_worst_outlier",
            "hard_cap_total": 10,
        },
        None,
        {},
    )
    assert [row.get("column") for row in unique_group_selector_pruned if isinstance(row, dict)] == [
        "Age",
        "Grad_Country",
        "ID",
    ]

    dense_numbered_group_keys = _resolve_inferred_group_keys(
        ["Q1", "Q2", "Q3", "Q4", "Q5", "Q30", "A1", "A2", "Age", "M1_Q1"]
    )
    assert dense_numbered_group_keys.get("Q1") == "Q"
    assert dense_numbered_group_keys.get("Q30") == "Q"
    assert dense_numbered_group_keys.get("A1") == "A1"
    assert dense_numbered_group_keys.get("A2") == "A2"
    assert dense_numbered_group_keys.get("M1_Q1") == "M1"

    role_bucket_pruned = _apply_a9_columns_policy(
        [
            {"column": f"A1_Q{idx}", "primary_role": "measure", "review_required": idx == 12}
            for idx in range(1, 13)
        ],
        {
            "name_field": "column",
            "role_field": "primary_role",
            "review_field": "review_required",
            "include_roles_all": [{"role": "measure", "cap": 12}],
            "selection_cap_per_inferred_group": 6,
            "group_sample_mode": "representative_plus_worst_outlier",
            "hard_cap_total": 20,
        },
        {"force_include_columns": ["A1_Q2"]},
        {},
    )
    assert [row.get("column") for row in role_bucket_pruned if isinstance(row, dict)] == [
        "A1_Q2",
        "A1_Q1",
        "A1_Q10",
        "A1_Q11",
        "A1_Q12",
        "A1_Q3",
        "A1_Q4",
    ]

    dense_q_role_bucket_pruned = _apply_a9_columns_policy(
        (
            [{"column": "ID", "primary_role": "id_key", "review_required": False}]
            + [{"column": f"Q{idx}", "primary_role": "invariant_attr", "review_required": True} for idx in range(1, 31)]
        ),
        {
            "name_field": "column",
            "role_field": "primary_role",
            "review_field": "review_required",
            "include_roles_all": [
                {"role": "id_key", "cap": 5},
                {"role": "invariant_attr", "cap": 30},
            ],
            "include_review_required_topk": 30,
            "selection_cap_per_inferred_group": 6,
            "group_sample_mode": "representative_plus_worst_outlier",
            "hard_cap_total": 40,
        },
        {"force_include_columns": ["ID"]},
        {},
    )
    dense_q_role_bucket_names = [row.get("column") for row in dense_q_role_bucket_pruned if isinstance(row, dict)]
    assert "ID" in dense_q_role_bucket_names
    assert "Q30" not in dense_q_role_bucket_names
    assert sum(1 for name in dense_q_role_bucket_names if re.match(r"^Q\d+$", str(name or ""))) <= 6

    reviewer_a1_columns = [f"A1_Q{idx}" for idx in range(1, 61)]
    reviewer_a2_columns = [f"A2_Q{idx}" for idx in range(1, 61)]
    reviewer_m1_columns = [f"M1_Q{idx}" for idx in range(1, 31)]
    reviewer_q_columns = [f"Q{idx}" for idx in range(1, 31)]
    reviewer_anchor_columns = reviewer_a1_columns[:24] + reviewer_a2_columns[:16]
    reviewer_all_columns = reviewer_a1_columns + reviewer_a2_columns + reviewer_m1_columns + reviewer_q_columns + ["Mjr2", "ID", "ANXATT"]
    reviewer_fixture_payloads: Dict[str, Tuple[str, Any, Dict[str, Any]]] = {
        "A2": (
            "json",
            [
                {
                    "column": column,
                    "high_missingness": False,
                    "missing_tokens_observed": [],
                    "a2_samples": {
                        "head": [f"{column}_v1", f"{column}_v2", f"{column}_v3"],
                        "tail": [f"{column}_v4"],
                        "random": [f"{column}_v5"],
                    },
                }
                for column in reviewer_all_columns
            ],
            {},
        ),
        "A3-T": ("json", {"items": []}, {}),
        "A3-V": ("json", {"items": []}, {}),
        "A4": (
            "json",
            {
                "per_column": [
                    {
                        "column": column,
                        "missing_pct": 0.0,
                        "token_breakdown": {},
                    }
                    for column in reviewer_all_columns
                ]
            },
            {},
        ),
        "A9": (
            "json",
            {
                "columns": (
                    [
                        {"column": "ID", "primary_role": "id_key", "review_required": False},
                        {"column": "ANXATT", "primary_role": "measure", "review_required": True},
                        {"column": "M1_Q14", "primary_role": "invariant_attr", "review_required": True},
                        {"column": "M1_Q18", "primary_role": "invariant_attr", "review_required": True},
                    ]
                    + [
                        {"column": column, "primary_role": "invariant_attr", "review_required": True}
                        for column in reviewer_q_columns
                    ]
                )
            },
            {},
        ),
        "A13": (
            "json",
            {
                "columns": (
                    [
                        {
                            "column": column,
                            "technical_type": "categorical",
                            "detected_anchors": [
                                {
                                    "anchor": "SURVEY_ITEM",
                                    "confidence": 0.91,
                                    "implication": "REVIEW_CONTEXT",
                                }
                            ],
                        }
                        for column in reviewer_anchor_columns
                    ]
                    + [
                        {
                            "column": "ID",
                            "technical_type": "numeric",
                            "detected_anchors": [],
                        }
                    ]
                )
            },
            {},
        ),
        "A14": (
            "json",
            {
                "columns": [
                    {
                        "column": column,
                        "global_quality_score": (
                            0.65 if column == "Mjr2"
                            else 0.8 if column in {"M1_Q14", "M1_Q18", "Q30"}
                            else 0.75 if column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20])
                            else 0.95
                        ),
                        "drift_detected": column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20] + ["Mjr2"]),
                        "interpretation": "Moderate drift detected across segments." if column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20] + ["Mjr2"]) else "Stable",
                    }
                    for column in reviewer_all_columns
                ]
            },
            {},
        ),
        "A16": (
            "json",
            {
                "detected_skip_logic": [
                    {
                        "trigger_column": "ANXATT",
                        "sample_affected_columns": ["A2_Q1"],
                    }
                ],
                "master_switch_candidates": [
                    {"trigger_column": "ANXATT"}
                ],
            },
            {},
        ),
        "A17": (
            "json",
            {
                "columns": [
                    {
                        "column": column,
                        "skip_logic_protected": column == "A2_Q1",
                        "type_review_required": (
                            column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20] + ["Mjr2", "M1_Q14", "M1_Q18"] + reviewer_q_columns[:8])
                        ),
                        "missingness_review_required": (
                            column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20] + ["Mjr2"])
                        ),
                        "needs_human_review": (
                            column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20] + ["Mjr2", "M1_Q14", "M1_Q18"] + reviewer_q_columns[:8])
                        ),
                        "drift_detected": column in set(reviewer_a2_columns[:8] + ["Mjr2"]),
                        "confidence": (
                            0.6 if column == "A2_Q1"
                            else 0.65 if column == "Mjr2"
                            else 0.55 if column in {"M1_Q14", "M1_Q18"}
                            else 0.76 if column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20] + reviewer_q_columns[:8])
                            else 0.96
                        ),
                        "quality_score": (
                            0.4 if column == "Mjr2"
                            else 0.65 if column == "A2_Q1"
                            else 0.68 if column in {"M1_Q14", "M1_Q18"}
                            else 0.82 if column in set(reviewer_a1_columns[:20] + reviewer_a2_columns[:20] + reviewer_q_columns[:8])
                            else 0.97
                        ),
                    }
                    for column in reviewer_all_columns
                ]
            },
            {},
        ),
    }
    original_get_decoded_payload = globals()["_get_decoded_payload"]

    def _fake_reviewer_payload(run_id: str, artifact_id: str) -> Tuple[str, Any, Dict[str, Any]]:
        if artifact_id not in reviewer_fixture_payloads:
            raise HTTPException(status_code=404, detail=f"missing fixture artifact {artifact_id}")
        return reviewer_fixture_payloads[artifact_id]

    globals()["_get_decoded_payload"] = _fake_reviewer_payload
    try:
        reviewer_buckets = _build_post_contract_auto_scope_buckets("run_pruning_smoke")
        reviewer_bundle = _artifact_bundle_view(
            req=ArtifactBundleRequest(
                run_id="run_pruning_smoke",
                mode="canonical_contract_reviewer",
                global_scope=ArtifactBundleScope(
                    mode="canonical_contract_reviewer",
                    value_filter={"force_include_columns": ["ID", "ANXATT", "A2_Q1"]},
                ),
            ),
            response=Response(),
        )
        reviewer_bundle_debug = _artifact_bundle_view(
            req=ArtifactBundleRequest(
                run_id="run_pruning_smoke",
                mode="canonical_contract_reviewer",
                global_scope=ArtifactBundleScope(
                    mode="canonical_contract_reviewer",
                    value_filter={"force_include_columns": ["ID", "ANXATT", "A2_Q1"]},
                ),
                debug=True,
            ),
            response=Response(),
        )
    finally:
        globals()["_get_decoded_payload"] = original_get_decoded_payload

    assert reviewer_buckets["structural_columns"] == ["ID"]
    assert reviewer_buckets["review_priority_columns"][:4] == ["ANXATT", "M1_Q14", "M1_Q18", "Q1"]
    assert "Q30" in reviewer_buckets["review_priority_columns"]
    assert reviewer_buckets["reviewer_focus_columns"][0] == "A2_Q1"
    assert reviewer_buckets["reviewer_focus_columns"][1] == "Mjr2"
    assert len(reviewer_buckets["reviewer_focus_columns"]) == 28
    assert sum(1 for name in reviewer_buckets["reviewer_focus_columns"][:15] if name.startswith("A1_Q")) <= 4
    assert sum(1 for name in reviewer_buckets["reviewer_focus_columns"][:15] if name.startswith("A2_Q")) <= 4
    assert sum(1 for name in reviewer_buckets["reviewer_focus_columns"][:15] if re.match(r"^Q\d+$", name)) <= 4
    assert "Q30" not in reviewer_buckets["reviewer_focus_columns"]

    reviewer_a2_names = [row.get("column") for row in reviewer_bundle.get("artifacts", {}).get("A2", []) if isinstance(row, dict)]
    reviewer_a4_names = [row.get("column") for row in reviewer_bundle.get("artifacts", {}).get("A4", {}).get("per_column", []) if isinstance(row, dict)]
    reviewer_a9_names = [row.get("column") for row in reviewer_bundle.get("artifacts", {}).get("A9", {}).get("columns", []) if isinstance(row, dict)]
    reviewer_a13_rows = [row for row in reviewer_bundle.get("artifacts", {}).get("A13", {}).get("columns", []) if isinstance(row, dict)]
    reviewer_a13_names = [row.get("column") for row in reviewer_a13_rows]
    reviewer_a14_names = [row.get("column") for row in reviewer_bundle.get("artifacts", {}).get("A14", {}).get("columns", []) if isinstance(row, dict)]
    reviewer_a17_names = [row.get("column") for row in reviewer_bundle.get("artifacts", {}).get("A17", {}).get("columns", []) if isinstance(row, dict)]
    reviewer_bundle_report = reviewer_bundle_debug.get("_bundle_prune_report", {})
    reviewer_a13_report = reviewer_bundle_report.get("A13", {})
    reviewer_a13_policy_report = ((reviewer_a13_report.get("column_scope_policies") or [{}])[-1]) if isinstance(reviewer_a13_report, dict) else {}
    assert reviewer_bundle.get("artifacts", {}).get("A3-T", {}).get("items") == []
    assert reviewer_bundle.get("artifacts", {}).get("A3-V", {}).get("items") == []
    assert len(reviewer_a2_names) <= 120
    assert len(reviewer_a4_names) <= 120
    assert len(reviewer_a9_names) <= 100
    assert len(reviewer_a13_names) <= 40
    assert len(reviewer_a14_names) <= 96
    assert len(reviewer_a17_names) <= 120
    assert "A1_Q1" in reviewer_a2_names
    assert "A1_Q1" in reviewer_a4_names
    assert "A1_Q1" in reviewer_a13_names
    assert "A1_Q1" in reviewer_a14_names
    assert "A1_Q1" in reviewer_a17_names
    assert "Mjr2" in reviewer_a14_names
    assert "Mjr2" in reviewer_a17_names
    assert "M1_Q14" in reviewer_a14_names
    assert "M1_Q18" in reviewer_a14_names
    assert "M1_Q14" in reviewer_a17_names
    assert "M1_Q18" in reviewer_a17_names
    assert "ID" in reviewer_a2_names
    assert "ANXATT" in reviewer_a2_names
    assert "A2_Q1" in reviewer_a2_names
    assert "ID" in reviewer_a9_names
    assert "ANXATT" in reviewer_a9_names
    assert "Q30" not in reviewer_a9_names
    assert sum(1 for name in reviewer_a9_names if re.match(r"^Q\d+$", str(name or ""))) <= 6
    assert "ID" in reviewer_a17_names
    assert "ANXATT" in reviewer_a17_names
    assert "A2_Q1" in reviewer_a17_names
    assert "ID" not in reviewer_a13_names
    assert all(_is_nonempty_field_value(row.get("detected_anchors")) for row in reviewer_a13_rows)
    assert reviewer_a13_policy_report.get("forced_output") == 0
    assert "Q30" not in reviewer_a14_names
    assert "Q30" not in reviewer_a2_names
    assert "Q30" not in reviewer_a4_names
    assert "Q30" not in reviewer_a17_names
    assert "A1_Q60" not in reviewer_a2_names
    assert "A1_Q60" not in reviewer_a4_names
    assert "A1_Q60" not in reviewer_a14_names
    assert "A2_Q60" not in reviewer_a17_names
    assert sum(1 for name in reviewer_a2_names if name.startswith("A1_Q")) <= 6
    assert sum(1 for name in reviewer_a2_names if name.startswith("A2_Q")) <= 7
    assert sum(1 for name in reviewer_a4_names if name.startswith("A1_Q")) <= 6
    assert sum(1 for name in reviewer_a4_names if name.startswith("A2_Q")) <= 7
    assert sum(1 for name in reviewer_a14_names if name.startswith("A1_Q")) <= 10
    assert sum(1 for name in reviewer_a14_names if name.startswith("A2_Q")) <= 11
    assert sum(1 for name in reviewer_a17_names if name.startswith("A1_Q")) <= 10
    assert sum(1 for name in reviewer_a17_names if name.startswith("A2_Q")) <= 11
    assert len(reviewer_a2_names) < len(reviewer_all_columns)
    assert len(reviewer_a4_names) < len(reviewer_all_columns)
    assert len(reviewer_a14_names) < len(reviewer_all_columns)
    assert len(json.dumps(reviewer_bundle, sort_keys=True).encode("utf-8")) < (120 * 1024)

    light_contract_override_rows = []
    for default in DEFAULT_OVERRIDE_FIELDS:
        row = {
            "field": default["field"],
            "description": default["description"],
            "user_input": "",
        }
        if default["field"] == "dataset_context_and_collection_notes":
            row["user_input"] = "Survey of customer onboarding. One row is one submitted response. Form logic changed after wave 2."
        elif default["field"] == "semantic_codebook_and_important_variables":
            row["user_input"] = "Q12 is the master switch. StatusCode values 1=active, 2=paused."
        light_contract_override_rows.append(row)

    light_contract_payload = {
        "run_id": "run_smoke",
        "generated_at": "2026-03-17T00:00:00Z",
        "source_endpoint": "/light-contracts/xlsx",
        "column_guide_rows": [],
        "grain_summary_rows": [],
        "primary_grain_rows": [
            {
                "item": "base_grain",
                "recommended_key_1": "response_id",
                "recommended_key_2": "",
                "recommended_key_3": "",
                "your_key_1": "",
                "your_key_2": "",
                "your_key_3": "",
                "status": "accept",
                "comments": "",
            }
        ],
        "reference_rows": [],
        "repeat_family_rows": [],
        "structural_gate_rows": [],
        "scale_mapping_rows": [
            {
                "target_kind": "family",
                "target_id": "q_9_main_cell_group",
                "response_scale_kind": "familiarity_scale",
                "ordered_labels_low_to_high": "Never Heard of It 0|Very Familiar 6",
                "numeric_scores_low_to_high": "",
                "notes": "Optional structured scale mapping.",
            }
        ],
        "override_rows": light_contract_override_rows,
    }
    accepted_handoff = _build_accepted_light_contract_handoff(light_contract_payload)
    assert accepted_handoff.get("reference_decisions") == []
    assert accepted_handoff.get("dimension_decisions") == []
    assert accepted_handoff.get("semantic_context_input") == {
        "dataset_context_and_collection_notes": "Survey of customer onboarding. One row is one submitted response. Form logic changed after wave 2.",
        "semantic_codebook_and_important_variables": "Q12 is the master switch. StatusCode values 1=active, 2=paused.",
    }
    assert accepted_handoff.get("scale_mapping_input", [])[0].get("target_id") == "q_9_main_cell_group"

    legacy_light_contract_payload = dict(light_contract_payload)
    legacy_light_contract_payload.pop("reference_rows", None)
    legacy_light_contract_payload["dimension_rows"] = [
        {
            "table_name": "dim_clinic",
            "recommended_key_1": "clinic_id",
            "recommended_key_2": "",
            "recommended_key_3": "",
            "your_key_1": "",
            "your_key_2": "",
            "your_key_3": "",
            "relationship_to_primary": "many_to_one",
            "status": "accept",
            "comments": "Legacy dimension row should map to reference decisions.",
        }
    ]
    legacy_handoff = _build_accepted_light_contract_handoff(legacy_light_contract_payload)
    assert legacy_handoff.get("reference_decisions") == legacy_handoff.get("dimension_decisions")
    assert legacy_handoff.get("reference_decisions", [])[0].get("table_name") == "dim_clinic"
    assert accepted_handoff.get("override_notes", {}).get("dataset_context_and_collection_notes")

    light_contract_bytes = build_light_contract_xlsx_bytes(light_contract_payload)
    parsed_light_contract = parse_light_contract_xlsx_bytes(light_contract_bytes)
    assert parsed_light_contract.get("metadata", {}).get("run_id") == "run_smoke"
    parsed_handoff = _build_parsed_light_contract_handoff("run_smoke", parsed_light_contract)
    assert parsed_handoff.get("reference_decisions") == parsed_handoff.get("dimension_decisions")
    assert parsed_handoff.get("semantic_context_input") == accepted_handoff.get("semantic_context_input")
    assert parsed_handoff.get("scale_mapping_input") == accepted_handoff.get("scale_mapping_input")
    assert parsed_handoff.get("override_notes", {}).get("semantic_codebook_and_important_variables") == "Q12 is the master switch. StatusCode values 1=active, 2=paused."


def _get_decoded_payload(run_id: str, artifact_id: str) -> Tuple[str, Any, Dict[str, Any]]:
    content_type, data, artifact_meta = load_artifact_bytes(run_id=run_id, artifact_id=artifact_id)
    digest = sha256_hex(data)
    cache_key = (run_id, artifact_id, digest)
    if cache_key in _DECODE_CACHE:
        kind, payload = _DECODE_CACHE[cache_key]
        return kind, payload, artifact_meta

    kind, payload = decode_payload(content_type, data)
    if len(_DECODE_CACHE) >= _DECODE_CACHE_MAX:
        _DECODE_CACHE.pop(next(iter(_DECODE_CACHE)))
    _DECODE_CACHE[cache_key] = (kind, payload)
    return kind, payload, artifact_meta


def _sorted_nonempty_strings(values: Iterable[Any]) -> List[str]:
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _merge_value_filter_column_bucket(
    value_filter: Optional[Any],
    bucket_key: str,
    extra_columns: Iterable[Any],
) -> Optional[Any]:
    extra = _ordered_nonempty_strings(extra_columns)
    if not extra:
        return value_filter

    if value_filter is None:
        return {bucket_key: extra}

    if isinstance(value_filter, dict):
        merged = dict(value_filter)
        combined = _ordered_nonempty_strings(_extract_string_list_from_value_filter(merged, [bucket_key]) + extra)
        merged[bucket_key] = combined
        return merged

    if isinstance(value_filter, list):
        merged_items: List[Any] = []
        combined = _ordered_nonempty_strings(_extract_string_list_from_value_filter(value_filter, [bucket_key]) + extra)
        bucket_written = False
        for item in value_filter:
            if isinstance(item, dict) and bucket_key in item:
                if not bucket_written:
                    replacement = dict(item)
                    replacement[bucket_key] = combined
                    merged_items.append(replacement)
                    bucket_written = True
                continue
            merged_items.append(item)
        if not bucket_written:
            merged_items.append({bucket_key: combined})
        return merged_items

    return {bucket_key: extra}


def _merge_value_filter_force_include_columns(
    value_filter: Optional[Any],
    extra_columns: Iterable[Any],
) -> Optional[Any]:
    return _merge_value_filter_column_bucket(value_filter, "force_include_columns", extra_columns)


def _light_contract_force_include_columns(decisions: Dict[str, Any]) -> List[str]:
    columns: List[str] = []
    primary = decisions.get("primary_grain_decision") or {}
    columns.extend(primary.get("keys") or [])

    for ref in (decisions.get("reference_decisions") or decisions.get("dimension_decisions") or []):
        if isinstance(ref, dict):
            columns.extend(ref.get("keys") or [])

    for family in decisions.get("family_decisions") or []:
        if not isinstance(family, dict):
            continue
        parent_key = str(family.get("parent_key") or "").strip()
        repeat_index_name = str(family.get("repeat_index_name") or "").strip()
        if parent_key:
            columns.append(parent_key)
        if repeat_index_name:
            columns.append(repeat_index_name)

    return _sorted_nonempty_strings(columns)


def _coerce_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _reviewer_focus_severity(row: Dict[str, Any]) -> Tuple[int, float, float]:
    confidence = _coerce_float(row.get("confidence"), 1.0)
    quality_score = _coerce_float(row.get("quality_score"), 1.0)
    type_review_required = bool(row.get("type_review_required", False))
    missingness_review_required = bool(row.get("missingness_review_required", False))

    severity = 0
    if bool(row.get("skip_logic_protected", False)):
        severity += 40
    if bool(row.get("drift_detected", False)):
        severity += 30
    if confidence < 0.8:
        severity += 20
    if confidence < 0.7:
        severity += 10
    if quality_score <= 0.85:
        severity += 20
    if quality_score <= 0.7:
        severity += 10
    if type_review_required:
        severity += 15
    if missingness_review_required:
        severity += 15
    if type_review_required and missingness_review_required:
        severity += 10

    return severity, confidence, quality_score


def _build_ranked_reviewer_focus_columns(rows: Iterable[Any]) -> List[str]:
    candidates: List[Tuple[int, float, float, str]] = []
    total_columns = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        col = str(row.get("column") or "").strip()
        if not col:
            continue
        total_columns += 1
        severity, confidence, quality_score = _reviewer_focus_severity(row)
        if severity >= 40:
            candidates.append((severity, confidence, quality_score, col))

    if not candidates:
        return []

    budget = min(64, max(16, int(math.ceil(total_columns * 0.15))))
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2], item[3]))
    group_keys_by_name = _resolve_inferred_group_keys(item[3] for item in ranked)
    selected: List[str] = []
    selected_seen: Set[str] = set()
    selected_group_counts: Dict[str, int] = {}

    def _take(column_name: str) -> None:
        if column_name in selected_seen or len(selected) >= budget:
            return
        selected_seen.add(column_name)
        selected.append(column_name)
        group_key = group_keys_by_name.get(column_name, _infer_column_group_key(column_name))
        if group_key:
            selected_group_counts[group_key] = selected_group_counts.get(group_key, 0) + 1

    for _, _, _, column_name in ranked:
        group_key = group_keys_by_name.get(column_name, _infer_column_group_key(column_name))
        if group_key and selected_group_counts.get(group_key, 0) >= 4:
            continue
        _take(column_name)
        if len(selected) >= budget:
            return selected

    for _, _, _, column_name in ranked:
        _take(column_name)
        if len(selected) >= budget:
            break

    return selected


def _build_post_contract_auto_scope_buckets(run_id: str) -> Dict[str, List[str]]:
    review_columns: Set[str] = set()
    structural_columns: Set[str] = set()
    skip_trigger_columns: Set[str] = set()
    skip_affected_preview_columns: Set[str] = set()
    review_priority_columns: List[str] = []
    reviewer_focus_columns: List[str] = []

    try:
        kind, payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A3-T")
        if kind == "json" and isinstance(payload, dict):
            for item in payload.get("items") or []:
                if isinstance(item, dict):
                    col = str(item.get("column") or "").strip()
                    if col:
                        review_columns.add(col)
    except HTTPException:
        pass

    try:
        kind, payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A3-V")
        if kind == "json" and isinstance(payload, dict):
            for item in payload.get("items") or []:
                if isinstance(item, dict):
                    col = str(item.get("column") or "").strip()
                    if col:
                        review_columns.add(col)
    except HTTPException:
        pass

    try:
        kind, payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A9")
        if kind == "json" and isinstance(payload, dict):
            for row in payload.get("columns") or []:
                if not isinstance(row, dict):
                    continue
                col = str(row.get("column") or "").strip()
                role = str(row.get("primary_role") or "").strip()
                review_required = bool(row.get("review_required", False))
                if col and role in {"id_key", "time_index", "repeat_index"}:
                    structural_columns.add(col)
                if col and review_required:
                    review_priority_columns.append(col)
    except HTTPException:
        pass

    try:
        kind, payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A16")
        if kind == "json" and isinstance(payload, dict):
            for rule in payload.get("detected_skip_logic") or []:
                if not isinstance(rule, dict):
                    continue
                trigger = str(rule.get("trigger_column") or "").strip()
                if trigger:
                    skip_trigger_columns.add(trigger)
                for col in rule.get("sample_affected_columns") or []:
                    col_text = str(col or "").strip()
                    if col_text:
                        skip_affected_preview_columns.add(col_text)
            for candidate in payload.get("master_switch_candidates") or []:
                if isinstance(candidate, dict):
                    trigger = str(candidate.get("trigger_column") or "").strip()
                    if trigger:
                        skip_trigger_columns.add(trigger)
    except HTTPException:
        pass

    try:
        kind, payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A17")
        if kind == "json" and isinstance(payload, dict):
            reviewer_focus_columns = _build_ranked_reviewer_focus_columns(payload.get("columns") or [])
    except HTTPException:
        pass

    return {
        "review_columns": sorted(review_columns),
        "structural_columns": sorted(structural_columns),
        "skip_trigger_columns": sorted(skip_trigger_columns),
        "skip_affected_preview_columns": sorted(skip_affected_preview_columns),
        "review_priority_columns": _ordered_nonempty_strings(review_priority_columns),
        "reviewer_focus_columns": reviewer_focus_columns,
    }


def _effective_value_filter_for_mode(
    run_id: str,
    mode: str,
    value_filter: Optional[Any],
) -> Optional[Any]:
    if str(mode or "").strip() not in {"type_transform_worker", "missingness_worker", "semantic_context_worker", "table_layout_worker", "analysis_layout_worker", "canonical_contract_reviewer"}:
        return value_filter
    buckets = _build_post_contract_auto_scope_buckets(run_id)
    merged = value_filter
    for bucket_key, columns in buckets.items():
        merged = _merge_value_filter_column_bucket(merged, bucket_key, columns)
    return merged


def _build_view_response(kind: str, payload: Any, report: Dict[str, Any], debug: bool, profile_header: str) -> Response:
    if kind == "json":
        headers: Dict[str, str] = {"X-Pruning-Profile": profile_header}
        if debug:
            headers["X-Prune-Report"] = json.dumps(report, separators=(",", ":"), default=_json_default)[:4096]
        return JSONResponse(content=payload, headers=headers)

    if kind == "jsonl":
        headers = {"X-Pruning-Profile": profile_header}
        if debug:
            headers["X-Prune-Report"] = json.dumps(report, separators=(",", ":"), default=_json_default)[:4096]
        return StreamingResponse(_iter_jsonl_bytes(payload), media_type="application/jsonl", headers=headers)

    raise HTTPException(status_code=415, detail="Artifact view supports only JSON/JSONL. Use /download for raw bytes.")


def _coerce_bundle_input_json(value: Any, field_name: str) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"{field_name} must be valid JSON when provided as a string") from exc
    raise HTTPException(status_code=422, detail=f"{field_name} must be a JSON object, array, stringified JSON, or null")


def _build_scale_mapping_worker_bundle_response(req: ArtifactBundleRequest) -> Dict[str, Any]:
    light_contract_decisions = _coerce_bundle_input_json(req.light_contract_decisions, "light_contract_decisions")
    if light_contract_decisions is None:
        light_contract_decisions = _load_json_from_run_object(req.run_id, "light_contract_decisions.json")
    if not isinstance(light_contract_decisions, dict):
        raise HTTPException(status_code=422, detail="light_contract_decisions must resolve to a JSON object")

    family_worker_json = _coerce_bundle_input_json(req.family_worker_json, "family_worker_json")
    bundle = _build_scale_mapping_bundle(
        run_id=req.run_id,
        light_contract_decisions=light_contract_decisions,
        family_worker_json=family_worker_json or {},
    )
    return {
        "run_id": req.run_id,
        "mode": "scale_mapping_worker",
        "artifact_ids": ["scale_mapping_bundle"],
        "artifacts": {
            "scale_mapping_bundle": bundle,
        },
    }


@app.get("/artifacts/{artifact_id}")
def artifact_view_get(
    artifact_id: str,
    run_id: str,
    mode: str = "raw",
    keep: str = "",
    drop: str = "",
    limits: str = "",
    debug: bool = False,
    _=Depends(require_token),
):
    kind, payload, _artifact_meta = _get_decoded_payload(run_id=run_id, artifact_id=artifact_id)
    if kind == "other":
        raise HTTPException(status_code=415, detail="Artifact view supports only JSON/JSONL. Use /download for raw bytes.")

    keep_keys = parse_csv_list(keep)
    drop_keys = parse_csv_list(drop)
    limits_map = parse_limits_csv(limits)
    resolved_mode_cfg, mode_meta = _resolve_mode_config(mode)
    effective_value_filter = _effective_value_filter_for_mode(run_id=run_id, mode=mode, value_filter=None)
    has_overrides = any([keep_keys, drop_keys, limits_map, effective_value_filter])
    override_tokens = ["keep", "drop", "limits"]
    if effective_value_filter:
        override_tokens.append("value_filter")
    profile_header = mode_meta["header"] if not has_overrides else f"{mode_meta['header']}; overrides={','.join(override_tokens)}"

    pruned, report = apply_llm_pruning(
        payload=payload,
        artifact_id=artifact_id,
        mode=mode,
        keep_keys=keep_keys,
        drop_keys=drop_keys,
        limits=limits_map,
        policy_overrides=None,
        replace_policies=None,
        transform_overrides=None,
        replace_transforms=None,
        value_filter=effective_value_filter,
        debug=debug,
        mode_config=resolved_mode_cfg,
    )

    if debug and kind == "json" and isinstance(pruned, dict):
        with_report = dict(pruned)
        with_report["_prune_report"] = report
        return JSONResponse(content=with_report, headers={"X-Pruning-Profile": profile_header})

    return _build_view_response(kind=kind, payload=pruned, report=report, debug=debug, profile_header=profile_header)


@app.post("/artifacts/{artifact_id}/view")
def artifact_view_post(
    artifact_id: str,
    req: ArtifactViewRequest,
    _=Depends(require_token),
):
    kind, payload, _artifact_meta = _get_decoded_payload(run_id=req.run_id, artifact_id=artifact_id)
    if kind == "other":
        raise HTTPException(status_code=415, detail="Artifact view supports only JSON/JSONL. Use /download for raw bytes.")

    mode_config, mode_meta = _resolve_mode_config(req.mode)
    override_parts: List[str] = []
    if req.policy_overrides:
        override_parts.append("policy_overrides")
    if req.transform_overrides:
        override_parts.append("transform_overrides")
    if req.keep:
        override_parts.append("keep")
    if req.drop:
        override_parts.append("drop")
    if req.limits:
        override_parts.append("limits")
    if req.value_filter is not None:
        override_parts.append("value_filter")
    profile_header = f"{mode_meta['header']}; overrides={','.join(override_parts) if override_parts else 'none'}"
    bundle_effective_policies = _build_effective_policies(
        base_policies=mode_config.get("tier3", {}).get("policies", {}),
        policy_overrides=req.policy_overrides,
        replace_policies=req.replace_policies,
        report=None,
    )
    bundle_effective_transforms = _build_effective_transforms(
        base_transforms=mode_config.get("tier3", {}).get("transforms", {}),
        transform_overrides=req.transform_overrides,
        replace_transforms=req.replace_transforms,
        report=None,
    )

    effective_value_filter = _effective_value_filter_for_mode(
        run_id=req.run_id,
        mode=req.mode,
        value_filter=req.value_filter,
    )
    pruned, report = apply_llm_pruning(
        payload=payload,
        artifact_id=artifact_id,
        mode=req.mode,
        keep_keys=req.keep,
        drop_keys=req.drop,
        effective_policies=bundle_effective_policies,
        limits=req.limits,
        policy_overrides=req.policy_overrides,
        replace_policies=req.replace_policies,
        transform_overrides=req.transform_overrides,
        replace_transforms=req.replace_transforms,
        value_filter=effective_value_filter,
        debug=req.debug,
        effective_transforms=bundle_effective_transforms,
        mode_config=mode_config,
    )

    if req.debug and kind == "json" and isinstance(pruned, dict):
        with_report = dict(pruned)
        with_report["_prune_report"] = report
        return JSONResponse(content=with_report, headers={"X-Pruning-Profile": profile_header})

    return _build_view_response(kind=kind, payload=pruned, report=report, debug=req.debug, profile_header=profile_header)


def _artifact_bundle_view(
    req: ArtifactBundleRequest,
    response: Response,
) -> Dict[str, Any]:
    if str(req.mode or "").strip() == "scale_mapping_worker":
        response.headers["X-Pruning-Profile"] = "scale_mapping_worker; source=backend_bundle; overrides=none"
        return _build_scale_mapping_worker_bundle_response(req)

    artifacts_out: Dict[str, Any] = {}
    bundle_report: Dict[str, Any] = {}

    global_scope = req.global_scope or ArtifactBundleScope(mode=req.mode or "raw")
    if req.mode and req.mode != "raw" and (global_scope.mode is None or global_scope.mode == "raw"):
        global_scope.mode = req.mode
    if not global_scope.mode:
        global_scope.mode = req.mode or "raw"
    per_artifact = req.per_artifact or {}
    profile_mode_cfg: Optional[Dict[str, Any]] = None
    mode_meta: Optional[Dict[str, Any]] = None

    if global_scope.mode not in {"raw", "llm_baseline"}:
        profile_mode_cfg, mode_meta = _resolve_mode_config(global_scope.mode)

    effective_artifact_ids = list(req.artifact_ids or [])
    if not effective_artifact_ids:
        if mode_meta and mode_meta.get("profile_artifacts"):
            effective_artifact_ids = list(mode_meta.get("profile_artifacts") or [])
        else:
            raise HTTPException(status_code=422, detail="artifact_ids are required for mode raw or llm_baseline")

    override_parts: List[str] = []
    if req.policy_overrides:
        override_parts.append("policy_overrides")
    if req.transform_overrides:
        override_parts.append("transform_overrides")
    if req.artifact_ids:
        override_parts.append("artifact_ids")
    if global_scope.value_filter is not None or any(
        scope.value_filter is not None for scope in (per_artifact or {}).values()
    ):
        override_parts.append("value_filter")
    profile_header = (mode_meta or {"header": f"{global_scope.mode}; source=request"})["header"]
    profile_header = f"{profile_header}; overrides={','.join(override_parts) if override_parts else 'none'}"
    response.headers["X-Pruning-Profile"] = profile_header

    for artifact_id in effective_artifact_ids:
        artifact_scope = per_artifact.get(artifact_id) or ArtifactBundleScope()
        effective_mode = artifact_scope.mode or global_scope.mode or "raw"
        resolved_mode_cfg, _effective_mode_meta = (profile_mode_cfg, mode_meta) if (profile_mode_cfg is not None and effective_mode == global_scope.mode) else _resolve_mode_config(effective_mode)
        effective_keep = artifact_scope.keep if artifact_scope.keep is not None else global_scope.keep
        effective_drop = artifact_scope.drop if artifact_scope.drop is not None else global_scope.drop
        effective_limits = artifact_scope.limits if artifact_scope.limits is not None else global_scope.limits
        effective_value_filter = artifact_scope.value_filter if artifact_scope.value_filter is not None else global_scope.value_filter
        effective_value_filter = _effective_value_filter_for_mode(
            run_id=req.run_id,
            mode=effective_mode,
            value_filter=effective_value_filter,
        )

        kind, payload, _ = _get_decoded_payload(run_id=req.run_id, artifact_id=artifact_id)
        if kind == "other":
            raise HTTPException(status_code=415, detail=f"Artifact {artifact_id} is not JSON/JSONL; use /download")

        pruned, report = apply_llm_pruning(
            payload=payload,
            artifact_id=artifact_id,
            mode=effective_mode,
            keep_keys=effective_keep,
            drop_keys=effective_drop,
            limits=effective_limits,
            policy_overrides=req.policy_overrides,
            replace_policies=req.replace_policies,
            transform_overrides=req.transform_overrides,
            replace_transforms=req.replace_transforms,
            value_filter=effective_value_filter,
            debug=req.debug,
            mode_config=resolved_mode_cfg,
        )
        artifacts_out[artifact_id] = pruned
        if req.debug:
            bundle_report[artifact_id] = report

    body: Dict[str, Any] = {
        "run_id": req.run_id,
        "mode": global_scope.mode or "raw",
        "artifact_ids": effective_artifact_ids,
        "artifacts": artifacts_out,
    }
    if req.debug:
        body["_bundle_prune_report"] = bundle_report
    return body


@app.get("/artifact-bundles")
def artifact_bundle_view_get(
    run_id: str,
    response: Response,
    mode: str = "raw",
    artifact_ids: str = "",
    debug: bool = False,
    _=Depends(require_token),
) -> Dict[str, Any]:
    req = ArtifactBundleRequest(
        run_id=run_id,
        mode=mode,
        artifact_ids=parse_csv_list(artifact_ids) or None,
        debug=debug,
    )
    return _artifact_bundle_view(req=req, response=response)


@app.post("/artifact-bundles/view")
def artifact_bundle_view_post(
    req: ArtifactBundleRequest,
    response: Response,
    _=Depends(require_token),
) -> Dict[str, Any]:
    return _artifact_bundle_view(req=req, response=response)


CANONICAL_MODELING_STATUSES = {
    "base_field",
    "child_repeat_member",
    "reference_field",
    "event_field",
    "excluded_from_outputs",
    "unresolved",
}

CANONICAL_TYPE_DECISION_SOURCES = {
    "reviewed_type_worker",
    "scale_mapping_resolver",
    "family_default",
    "a17_baseline",
    "unresolved_no_a2_evidence",
}

CANONICAL_STRUCTURE_DECISION_SOURCES = {
    "table_layout_worker",
    "light_contract_fallback",
    "unresolved",
}

CANONICAL_MISSINGNESS_DECISION_SOURCES = {
    "reviewed_missingness_worker",
    "family_default",
    "a17_baseline",
    "unresolved_no_a2_evidence",
}

CANONICAL_SEMANTIC_DECISION_SOURCES = {
    "semantic_context_worker",
    "family_worker",
    "scale_mapping_resolver",
    "unknown",
}

CANONICAL_ASSIGNMENT_ROLES = {
    "base_key",
    "base_attribute",
    "reference_key",
    "reference_attribute",
    "repeat_parent_key",
    "repeat_index",
    "melt_member",
    "reference_value",
    "exclude_from_outputs",
    "unresolved",
}

TYPE_LOGICAL_TYPES = {
    "identifier",
    "categorical_code",
    "nominal_category",
    "ordinal_category",
    "boolean_flag",
    "date",
    "datetime",
    "numeric_measure",
    "free_text",
    "mixed_or_ambiguous",
}

TYPE_STORAGE_TYPES = {
    "string",
    "integer",
    "decimal",
    "boolean",
    "date",
    "datetime",
}

TYPE_TRANSFORM_ACTIONS = {
    "trim_whitespace",
    "normalize_missing_tokens",
    "normalize_boolean_tokens",
    "cast_to_string",
    "cast_to_integer",
    "cast_to_decimal",
    "cast_to_date",
    "cast_to_datetime",
    "strip_numeric_formatting",
    "normalize_decimal_separator",
    "lowercase_values",
    "uppercase_values",
    "titlecase_values",
    "normalize_category_tokens",
    "extract_numeric_component",
    "strip_unit_suffix",
    "standardize_percent_scale",
}

TYPE_STRUCTURAL_HINTS = {
    "split_multiselect_tokens",
    "requires_multiselect_modeling_decision",
    "split_range_into_start_end",
    "requires_range_semantics_review",
    "requires_unit_normalization_review",
    "requires_wide_to_long_review",
    "requires_child_table_review",
    "requires_multi_column_derivation",
    "requires_start_end_pair_review",
    "requires_codebook_or_label_mapping_review",
}

POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS = set(SHARED_POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS)

TYPE_INTERPRETATION_HINTS = {
    "leading_zero_risk",
    "identifier_not_measure",
    "code_not_quantity",
    "time_index_not_identifier",
    "repeat_context_do_not_use_as_base_key",
    "skip_logic_protected",
    "mixed_content_high_risk",
    "free_text_high_cardinality",
    "numeric_parse_is_misleading",
    "light_contract_override_applied",
}

MISSINGNESS_DISPOSITIONS = {
    "no_material_missingness",
    "token_missingness_present",
    "structurally_valid_missingness",
    "partially_structural_missingness",
    "unexplained_high_missingness",
    "mixed_missingness_risk",
}

MISSINGNESS_HANDLING = {
    "no_action_needed",
    "protect_from_null_penalty",
    "retain_with_caution",
    "review_before_drop",
    "candidate_drop_review",
}

BOOL_TRUE_TOKENS = {"1", "y", "yes", "true", "t"}
BOOL_FALSE_TOKENS = {"0", "n", "no", "false", "f"}
LIKERT_TOKENS = {
    "strongly agree",
    "agree",
    "neutral",
    "disagree",
    "strongly disagree",
    "never",
    "rarely",
    "sometimes",
    "often",
    "always",
    "very satisfied",
    "satisfied",
    "unsatisfied",
    "very unsatisfied",
}

A13_ANCHOR_MEANINGS = {
    "US_ZIP_CODE": "Likely US ZIP or postal code.",
    "PII_LOCATION": "Likely location-sensitive identifier.",
    "EMAIL": "Likely email address.",
    "URL": "Likely URL or resource locator.",
    "PHONE": "Likely phone number.",
    "ISO_COUNTRY": "Likely country code or country label.",
    "STATE_CODE": "Likely state or province code.",
    "CURRENCY": "Likely currency-bearing value.",
    "ICD10_CODE": "Likely ICD-10 diagnosis or medical code.",
    "FISCAL_QUARTER": "Likely fiscal quarter or reporting period.",
}


def _coerce_json_input(value: Any, field_name: str, allow_list: bool = False) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            raise HTTPException(status_code=422, detail=f"{field_name} must not be blank")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"{field_name} is not valid JSON: {exc}") from exc
        value = parsed

    if isinstance(value, dict):
        return value
    if allow_list and isinstance(value, list):
        return value
    expected = "JSON object or array" if allow_list else "JSON object"
    raise HTTPException(status_code=422, detail=f"{field_name} must be a {expected}")


def _load_optional_decoded_payload(run_id: str, artifact_id: str) -> Tuple[Optional[str], Any, Optional[Dict[str, Any]], Optional[str]]:
    try:
        kind, payload, meta = _get_decoded_payload(run_id=run_id, artifact_id=artifact_id)
        return kind, payload, meta, None
    except HTTPException as exc:
        if exc.status_code == 404:
            return None, None, None, artifact_id
        raise


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _dedupe_preserve_order(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _merge_notes(*notes: Any) -> str:
    cleaned = _dedupe_preserve_order(notes)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return " ".join(cleaned)


def _pick_higher_confidence(existing: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> Dict[str, Any]:
    if existing is None:
        return candidate
    if _safe_float(candidate.get("confidence"), 0.0) > _safe_float(existing.get("confidence"), 0.0):
        return candidate
    return existing


def _looks_integer_like(values: Iterable[Any]) -> bool:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return False
    return all(re.fullmatch(r"[+-]?\d+", value) for value in cleaned)


def _looks_numeric_like(values: Iterable[Any]) -> bool:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return False
    return all(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value) for value in cleaned)


def _has_leading_zero_risk(values: Iterable[Any]) -> bool:
    return any(re.fullmatch(r"0\d+", str(value).strip() or "") for value in values)


def _boolean_like(top_levels: Iterable[Any]) -> bool:
    normalized = {str(value).strip().lower() for value in top_levels if str(value).strip()}
    if len(normalized) < 2 or len(normalized) > 3:
        return False
    return normalized <= (BOOL_TRUE_TOKENS | BOOL_FALSE_TOKENS)


def _ordinal_like(top_levels: Iterable[Any], encoding_type: str) -> bool:
    if encoding_type == "ordinal":
        return True
    normalized = {str(value).strip().lower() for value in top_levels if str(value).strip()}
    if not normalized:
        return False
    return any(any(token in level for token in LIKERT_TOKENS) for level in normalized)


def _anchor_fallback_semantics(a13_row: Dict[str, Any]) -> Tuple[str, str]:
    anchors = _coerce_list_of_dicts(a13_row.get("detected_anchors"))
    if not anchors:
        return "", ""
    ranked = sorted(anchors, key=lambda item: (-_safe_float(item.get("confidence"), 0.0), str(item.get("anchor") or "")))
    top_anchor = str(ranked[0].get("anchor") or "").strip()
    if not top_anchor:
        return "", ""
    return A13_ANCHOR_MEANINGS.get(top_anchor, f"Likely semantic anchor: {top_anchor}."), top_anchor


def _column_name_tokens(column: str) -> Set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(column or "").lower()) if token}


def _column_name_has_free_text_hint(column: str) -> bool:
    tokens = _column_name_tokens(column)
    compact = re.sub(r"[^a-z0-9]+", "", str(column or "").lower())
    if not tokens:
        return any(
            marker in compact
            for marker in ("freetext", "openended", "comment", "feedback", "verbatim", "describe", "description", "explain", "specify")
        )
    if "freetext" in tokens or "freetext" in compact or {"free", "text"} <= tokens:
        return True
    if "openended" in tokens or "openended" in compact or {"open", "ended"} <= tokens or {"open", "text"} <= tokens:
        return True
    if {"please", "specify"} <= tokens or {"other", "specify"} <= tokens:
        return True
    if compact and any(marker in compact for marker in ("comment", "comments", "feedback", "verbatim", "describe", "description", "explain", "specify")):
        return True
    return bool(tokens & {"comment", "comments", "feedback", "verbatim", "narrative", "describe", "description", "explain", "note", "notes", "specify"})


def _values_look_textual(values: Iterable[Any]) -> bool:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return False
    return any(
        len(value) >= 12
        or " " in value
        or bool(re.search(r"[A-Za-z]{4,}", value))
        for value in cleaned
    )


def _semantic_text_blob(semantic: Dict[str, Any], family_result: Dict[str, Any]) -> str:
    return " ".join(
        part.strip().lower()
        for part in [
            str(semantic.get("semantic_meaning") or ""),
            str(semantic.get("codebook_note") or ""),
            str(family_result.get("member_semantics_notes") or ""),
            str(family_result.get("recommended_family_role") or ""),
        ]
        if str(part or "").strip()
    )


def _family_semantics_imply_categorical_repeat(
    structure: Dict[str, Any],
    semantic: Dict[str, Any],
    family_result: Dict[str, Any],
) -> bool:
    if structure.get("canonical_modeling_status") != "child_repeat_member":
        return False

    family_role = str(family_result.get("recommended_family_role") or "").strip()
    if family_role == "repeated_measure_set":
        return False
    if family_role == "repeated_survey_block":
        return True

    semantic_blob = _semantic_text_blob(semantic, family_result)
    if not semantic_blob:
        return False

    positive_tokens = (
        "likert",
        "ordinal",
        "survey",
        "questionnaire",
        "categorical",
        "category",
        "rating",
        "familiarity",
        "agreement",
        "frequency",
        "response set",
        "response scale",
        "row-level ordinal",
        "row level ordinal",
    )
    negative_tokens = (
        "continuous",
        "measure set",
        "amount",
        "quantity",
        "duration",
        "concentration",
        "temperature",
        "weight",
        "height",
    )
    return any(token in semantic_blob for token in positive_tokens) and not any(token in semantic_blob for token in negative_tokens)


def _structural_validity_rank(validity: str) -> int:
    return {
        "not_applicable": 0,
        "plausible_structural": 1,
        "confirmed_structural": 2,
    }.get(str(validity or "").strip(), 0)


def _build_a16_column_context(a16_payload: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(a16_payload, dict):
        return {}

    context: Dict[str, Dict[str, Any]] = {}

    def _ensure(col: str) -> Dict[str, Any]:
        row = context.setdefault(
            col,
            {
                "trigger_columns": [],
                "structural_validity": "not_applicable",
                "sample_based": True,
            },
        )
        return row

    for rule in _coerce_list_of_dicts(a16_payload.get("detected_skip_logic")):
        trigger = str(rule.get("trigger_column") or "").strip()
        explained_pct = _safe_float(rule.get("missing_explained_pct"), 0.0)
        directionality = str(rule.get("directionality") or "").strip().lower()
        validity = "confirmed_structural" if (explained_pct >= 95.0 or directionality == "bidirectional") else "plausible_structural"
        for col in rule.get("sample_affected_columns") or []:
            col_name = str(col or "").strip()
            if not col_name:
                continue
            row = _ensure(col_name)
            row["trigger_columns"] = _dedupe_preserve_order(list(row["trigger_columns"]) + ([trigger] if trigger else []))
            if row["structural_validity"] != "confirmed_structural":
                row["structural_validity"] = validity

    for candidate in _coerce_list_of_dicts(a16_payload.get("master_switch_candidates")):
        trigger = str(candidate.get("trigger_column") or "").strip()
        for col in candidate.get("sample_affected_columns") or []:
            col_name = str(col or "").strip()
            if not col_name:
                continue
            row = _ensure(col_name)
            row["trigger_columns"] = _dedupe_preserve_order(list(row["trigger_columns"]) + ([trigger] if trigger else []))
            if row["structural_validity"] == "not_applicable":
                row["structural_validity"] = "plausible_structural"

    return context


def _build_family_missingness_context(
    column_family_map: Dict[str, str],
    reviewed_missingness_by_col: Dict[str, Dict[str, Any]],
    a16_by_col: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    context: Dict[str, Dict[str, Any]] = {}

    def _ensure(family_id: str) -> Dict[str, Any]:
        return context.setdefault(
            family_id,
            {
                "structural_validity": "not_applicable",
                "trigger_columns": [],
                "reviewed_structural_columns": [],
                "evidence_columns": [],
            },
        )

    def _apply(family_id: str, validity: str, trigger_columns: Iterable[Any], evidence_column: str, *, reviewed: bool) -> None:
        if not family_id:
            return
        row = _ensure(family_id)
        if _structural_validity_rank(validity) > _structural_validity_rank(row["structural_validity"]):
            row["structural_validity"] = validity
        row["trigger_columns"] = _dedupe_preserve_order(
            list(row["trigger_columns"]) + [str(trigger).strip() for trigger in trigger_columns if str(trigger or "").strip()]
        )
        if evidence_column:
            row["evidence_columns"] = _dedupe_preserve_order(list(row["evidence_columns"]) + [evidence_column])
            if reviewed:
                row["reviewed_structural_columns"] = _dedupe_preserve_order(list(row["reviewed_structural_columns"]) + [evidence_column])

    for column, a16_row in a16_by_col.items():
        family_id = str(column_family_map.get(column) or "").strip()
        validity = str(a16_row.get("structural_validity") or "not_applicable").strip()
        if family_id and validity in {"plausible_structural", "confirmed_structural"}:
            _apply(
                family_id,
                validity,
                a16_row.get("trigger_columns") or [],
                str(column),
                reviewed=False,
            )

    for column, reviewed_row in reviewed_missingness_by_col.items():
        family_id = str(column_family_map.get(column) or "").strip()
        if not family_id:
            continue
        disposition = str(reviewed_row.get("missingness_disposition") or "").strip()
        validity = "not_applicable"
        if bool(reviewed_row.get("skip_logic_protected", False)) or disposition == "structurally_valid_missingness":
            validity = "confirmed_structural"
        elif disposition == "partially_structural_missingness":
            validity = "plausible_structural"
        if validity != "not_applicable":
            _apply(
                family_id,
                validity,
                reviewed_row.get("trigger_columns") or [],
                str(column),
                reviewed=True,
            )

    return context


def _normalize_transform_review_output(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    decisions: Dict[str, Dict[str, Any]] = {}
    for item in _coerce_list_of_dicts(payload.get("items")):
        col = str(item.get("column") or "").strip()
        if col:
            decisions[col] = item
    return decisions


def _normalize_variable_type_review_output(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    decisions: Dict[str, Dict[str, Any]] = {}
    for item in _coerce_list_of_dicts(payload.get("items")):
        col = str(item.get("column") or "").strip()
        if col:
            decisions[col] = item
    return decisions


def _normalize_missing_catalog_output(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    decisions: Dict[str, Dict[str, Any]] = {}
    for item in _coerce_list_of_dicts(payload.get("per_column")):
        col = str(item.get("column") or "").strip()
        if col:
            decisions[col] = item
    return decisions


def _baseline_type_decision(
    column: str,
    a2_row: Optional[Dict[str, Any]],
    a3t_row: Optional[Dict[str, Any]],
    a3v_row: Optional[Dict[str, Any]],
    a4_row: Optional[Dict[str, Any]],
    a9_row: Optional[Dict[str, Any]],
    a13_row: Optional[Dict[str, Any]],
    a14_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not a2_row:
        return {
            "recommended_logical_type": "mixed_or_ambiguous",
            "recommended_storage_type": "string",
            "transform_actions": [],
            "structural_transform_hints": [],
            "interpretation_hints": ["mixed_content_high_risk"],
            "normalization_notes": "",
            "type_decision_source": "unresolved_no_a2_evidence",
            "type_review_required": True,
            "confidence": 0.5,
            "applied_sources": [],
        }

    top_candidate = a2_row.get("top_candidate") or {}
    top_type = str(top_candidate.get("type") or "").strip()
    top_parse = str(top_candidate.get("parse") or "").strip()
    a2_conf = _safe_float(top_candidate.get("confidence"), _safe_float(a2_row.get("confidence"), 0.6))
    unique_ratio = _safe_float(a2_row.get("unique_ratio"), 0.0)
    unique_count = int(a2_row.get("unique_count") or 0)
    numeric_pct = _safe_float((a2_row.get("numeric_profile") or {}).get("parseable_pct"), 0.0)
    datetime_pct = _safe_float((a2_row.get("datetime_profile") or {}).get("parseable_pct"), 0.0)
    top_levels = [str(value).strip() for value in (a2_row.get("top_levels") or []) if str(value).strip()]
    samples = [str(value).strip() for value in (((a2_row.get("a2_samples") or {}).get("random") or [])[:8]) if str(value).strip()]
    role = str((a9_row or {}).get("primary_role") or "").strip()
    encoding_type = str((a9_row or {}).get("encoding_type") or "").strip()
    a3t_top_candidate = (a3t_row or {}).get("top_candidate") or {}
    transform_type = str(a3t_top_candidate.get("type") or top_type).strip()
    transform_parse = str(a3t_top_candidate.get("parse") or top_parse).strip()
    transform_risk = str((a3t_row or {}).get("risk_level") or "").strip().lower()
    transform_required = bool((a3t_row or {}).get("requires_transform", False))
    ambiguity_gap = _safe_float((a3v_row or {}).get("confidence_gap"), 1.0)
    ambiguity_reason = str((a3v_row or {}).get("ambiguity_reason") or "").strip()
    missing_tokens_present = bool((a4_row or {}).get("token_breakdown") or a2_row.get("missing_tokens_observed"))
    anchor_names = {
        str(item.get("anchor") or "").strip()
        for item in _coerce_list_of_dicts((a13_row or {}).get("detected_anchors"))
        if str(item.get("anchor") or "").strip()
    }
    code_like_anchor = bool(anchor_names & {"US_ZIP_CODE", "ISO_COUNTRY", "STATE_CODE", "ICD10_CODE", "FISCAL_QUARTER"})
    string_like_anchor = bool(anchor_names & {"EMAIL", "URL", "PHONE"})
    free_text_name_hint = _column_name_has_free_text_hint(column)
    textual_value_hint = _values_look_textual(top_levels + samples)
    quality_score = _safe_float((a14_row or {}).get("global_quality_score"), 1.0)
    drift_detected = bool((a14_row or {}).get("drift_detected", False))

    applied_sources: List[str] = ["A2"]
    if a3t_row:
        applied_sources.append("A3-T")
    if a3v_row:
        applied_sources.append("A3-V")
    if a4_row and missing_tokens_present:
        applied_sources.append("A4")
    if a9_row:
        applied_sources.append("A9")
    if a13_row and anchor_names:
        applied_sources.append("A13")
    if a14_row:
        applied_sources.append("A14")

    transform_actions: List[str] = ["trim_whitespace"]
    structural_hints: List[str] = []
    interpretation_hints: List[str] = []
    logical_type = "mixed_or_ambiguous"
    storage_type = "string"
    normalization_notes = ""

    if role == "id_key":
        logical_type = "identifier"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["identifier_not_measure"]
        if numeric_pct >= 80.0:
            interpretation_hints.append("numeric_parse_is_misleading")
        if _has_leading_zero_risk(top_levels + samples):
            interpretation_hints.append("leading_zero_risk")
        normalization_notes = "Baseline typing preserves identifier semantics as string storage."
    elif role == "time_index" or top_type in {"date", "datetime"} or datetime_pct >= 80.0:
        logical_type = "datetime" if top_type == "datetime" or "time" in column.lower() or "timestamp" in column.lower() else "date"
        storage_type = logical_type
        transform_actions = ["trim_whitespace", "cast_to_datetime" if logical_type == "datetime" else "cast_to_date"]
        interpretation_hints = ["time_index_not_identifier"]
        normalization_notes = "Baseline typing preserves temporal semantics from parse evidence and role signals."
    elif role == "repeat_index":
        logical_type = "ordinal_category" if (_looks_integer_like(top_levels + samples) or encoding_type == "ordinal") else "categorical_code"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"] if _looks_numeric_like(top_levels + samples) else ["trim_whitespace", "normalize_category_tokens"]
        interpretation_hints = ["repeat_context_do_not_use_as_base_key"]
        if _looks_numeric_like(top_levels + samples):
            interpretation_hints.append("numeric_parse_is_misleading")
        normalization_notes = "Baseline typing preserves repeat-index semantics without promoting the field to a base identifier."
    elif transform_type in {"numeric_range", "range_like"}:
        logical_type = "mixed_or_ambiguous"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        structural_hints = ["split_range_into_start_end", "requires_range_semantics_review"]
        interpretation_hints = ["mixed_content_high_risk"]
        normalization_notes = "Range-like values need structural follow-up before safe numeric modeling."
    elif transform_type == "numeric_with_unit" or transform_parse == "currency+suffix_possible":
        logical_type = "numeric_measure"
        storage_type = "decimal"
        transform_actions = ["trim_whitespace", "extract_numeric_component", "strip_unit_suffix", "cast_to_decimal"]
        structural_hints = ["requires_unit_normalization_review"]
        normalization_notes = "Baseline typing keeps unit-bearing values numeric while preserving later unit-normalization review."
    elif transform_type == "percent" or transform_parse == "percent_possible":
        logical_type = "numeric_measure"
        storage_type = "decimal"
        transform_actions = ["trim_whitespace", "strip_numeric_formatting", "standardize_percent_scale", "cast_to_decimal"]
        normalization_notes = "Baseline typing standardizes percentage-like values into decimal numeric storage."
    elif transform_type == "categorical_multi":
        logical_type = "categorical_code" if (code_like_anchor or role == "coded_categorical") else "nominal_category"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        structural_hints = ["split_multiselect_tokens", "requires_multiselect_modeling_decision"]
        if code_like_anchor or role == "coded_categorical":
            structural_hints.append("requires_codebook_or_label_mapping_review")
            interpretation_hints.append("code_not_quantity")
        normalization_notes = "Baseline typing keeps multi-value cells as string-backed categories pending structural token-splitting."
    elif _boolean_like(top_levels):
        logical_type = "boolean_flag"
        storage_type = "boolean"
        transform_actions = ["trim_whitespace", "normalize_boolean_tokens"]
        normalization_notes = "Baseline typing treats the column as boolean-like based on observed value vocabulary."
    elif role == "measure_item" and (_ordinal_like(top_levels, encoding_type) or encoding_type == "ordinal" or unique_count <= 7):
        logical_type = "ordinal_category"
        storage_type = "string"
        if _looks_numeric_like(top_levels + samples):
            transform_actions = ["trim_whitespace", "strip_numeric_formatting", "cast_to_string"]
            interpretation_hints = ["numeric_parse_is_misleading"]
        else:
            transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        normalization_notes = "Baseline typing preserves item-style response values as ordered categories."
    elif code_like_anchor or role == "coded_categorical":
        logical_type = "categorical_code"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        structural_hints = ["requires_codebook_or_label_mapping_review"]
        interpretation_hints = ["code_not_quantity"]
        if _looks_numeric_like(top_levels + samples):
            transform_actions.insert(1, "strip_numeric_formatting")
            interpretation_hints.append("numeric_parse_is_misleading")
        normalization_notes = "Baseline typing preserves code-like values as strings rather than quantities."
    elif string_like_anchor:
        logical_type = "free_text"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["free_text_high_cardinality"]
        normalization_notes = "Anchor evidence indicates structured string content that should be preserved as text."
    elif free_text_name_hint and (textual_value_hint or top_type in {"text", "mixed"} or unique_ratio >= 0.35 or unique_count >= 20):
        logical_type = "free_text"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["free_text_high_cardinality"]
        if numeric_pct >= 80.0:
            interpretation_hints.append("numeric_parse_is_misleading")
        normalization_notes = "Column naming and observed samples indicate open-response text, so baseline typing preserves string content even when parser signals are noisy."
    elif top_type in {"numeric"} or (numeric_pct >= 80.0 and role not in {"coded_categorical", "measure_item", "repeat_index"}):
        logical_type = "numeric_measure"
        storage_type = "integer" if _looks_integer_like(top_levels + samples) else "decimal"
        transform_actions = ["trim_whitespace", "strip_numeric_formatting", "cast_to_integer" if storage_type == "integer" else "cast_to_decimal"]
        normalization_notes = "Baseline typing keeps numeric parse evidence as measure semantics."
    elif _ordinal_like(top_levels, encoding_type):
        logical_type = "ordinal_category"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        normalization_notes = "Baseline typing uses ordinal category semantics from value vocabulary and role evidence."
    elif top_type in {"text"} and (unique_ratio >= 0.5 or unique_count >= 50 or textual_value_hint):
        logical_type = "free_text"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["free_text_high_cardinality"]
        normalization_notes = "Baseline typing preserves high-cardinality text as free-form string content."
    elif top_type in {"categorical", "mixed", "text"} or role in {"invariant_attr", "measure_item"}:
        logical_type = "nominal_category"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        normalization_notes = "Baseline typing keeps low-cardinality values as string-backed categories."
    else:
        logical_type = "mixed_or_ambiguous"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["mixed_content_high_risk"]
        normalization_notes = "Baseline typing remains conservative because the profiler evidence is mixed."

    if missing_tokens_present and "normalize_missing_tokens" not in transform_actions:
        transform_actions.append("normalize_missing_tokens")

    if ambiguity_reason:
        interpretation_hints.append("mixed_content_high_risk")
        if logical_type not in {"identifier", "date", "datetime"} and ambiguity_gap <= 0.08 and a2_conf < 0.78:
            logical_type = "mixed_or_ambiguous"
            storage_type = "string"
            transform_actions = ["trim_whitespace", "cast_to_string"]
            structural_hints = []
            normalization_notes = "A3-V shows unresolved semantic ambiguity, so the deterministic baseline remains conservative."

    type_review_required = bool(
        logical_type == "mixed_or_ambiguous"
        or a2_conf < 0.75
        or ambiguity_reason
        or transform_risk in {"high", "medium"}
        or (drift_detected and quality_score < 0.85)
    )

    confidence = min(0.84, max(0.55, a2_conf))
    if ambiguity_reason:
        confidence = min(confidence, 0.68)
    if transform_risk == "high":
        confidence = min(confidence, 0.7)
    elif transform_risk == "medium":
        confidence = min(confidence, 0.74)
    if drift_detected:
        confidence = min(confidence, 0.76)
    if logical_type == "mixed_or_ambiguous":
        confidence = min(confidence, 0.58)

    return {
        "recommended_logical_type": logical_type,
        "recommended_storage_type": storage_type,
        "transform_actions": _dedupe_preserve_order(action for action in transform_actions if action in TYPE_TRANSFORM_ACTIONS),
        "structural_transform_hints": _dedupe_preserve_order(hint for hint in structural_hints if hint in TYPE_STRUCTURAL_HINTS),
        "interpretation_hints": _dedupe_preserve_order(hint for hint in interpretation_hints if hint in TYPE_INTERPRETATION_HINTS),
        "normalization_notes": normalization_notes,
        "type_decision_source": "a17_baseline",
        "type_review_required": type_review_required,
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "applied_sources": _dedupe_preserve_order(applied_sources),
    }


def _baseline_missingness_decision(
    column: str,
    a2_row: Optional[Dict[str, Any]],
    a4_row: Optional[Dict[str, Any]],
    a16_row: Optional[Dict[str, Any]],
    a14_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not a2_row and not a4_row:
        return {
            "missingness_disposition": "mixed_missingness_risk",
            "missingness_handling": "review_before_drop",
            "skip_logic_protected": False,
            "normalization_notes": "",
            "missingness_decision_source": "unresolved_no_a2_evidence",
            "missingness_review_required": True,
            "confidence": 0.5,
            "applied_sources": [],
        }

    missing_pct = _safe_float((a4_row or {}).get("missing_pct"), _safe_float((a2_row or {}).get("missing_pct"), 0.0))
    token_breakdown = (a4_row or {}).get("token_breakdown") or (a2_row or {}).get("missing_tokens_observed") or {}
    token_missing = bool(token_breakdown)
    structural_validity = str((a16_row or {}).get("structural_validity") or "not_applicable").strip()
    drift_detected = bool((a14_row or {}).get("drift_detected", False))

    applied_sources: List[str] = ["A4" if a4_row else "A2"]
    if a16_row:
        applied_sources.append("A16")
    if a14_row and drift_detected:
        applied_sources.append("A14")

    if structural_validity == "confirmed_structural":
        disposition = "structurally_valid_missingness"
        handling = "protect_from_null_penalty"
        skip_logic_protected = True
        notes = "A16 surfaced direct skip-logic evidence for this field."
        review_required = False
        confidence = 0.84
    elif structural_validity == "plausible_structural":
        disposition = "partially_structural_missingness"
        handling = "retain_with_caution"
        skip_logic_protected = False
        notes = "A16 surfaced plausible but not definitive structural missingness evidence."
        review_required = True
        confidence = 0.72
    elif missing_pct < 5.0 and not token_missing:
        disposition = "no_material_missingness"
        handling = "no_action_needed"
        skip_logic_protected = False
        notes = "Missingness is low and no explicit token-based cleanup signal is present."
        review_required = False
        confidence = 0.76
    elif token_missing and missing_pct < 40.0:
        disposition = "token_missingness_present"
        handling = "retain_with_caution"
        skip_logic_protected = False
        notes = "A4 shows explicit missing-like tokens that should be standardized cautiously."
        review_required = False
        confidence = 0.7
    elif missing_pct >= 80.0:
        disposition = "unexplained_high_missingness"
        handling = "candidate_drop_review"
        skip_logic_protected = False
        notes = "Missingness is very high and not structurally explained by available A16 evidence."
        review_required = True
        confidence = 0.62
    elif missing_pct >= 50.0:
        disposition = "mixed_missingness_risk"
        handling = "review_before_drop"
        skip_logic_protected = False
        notes = "Missingness is materially high without direct structural proof."
        review_required = True
        confidence = 0.64
    elif token_missing or missing_pct >= 20.0:
        disposition = "mixed_missingness_risk"
        handling = "retain_with_caution"
        skip_logic_protected = False
        notes = "Missingness is present but not strongly structural in deterministic baseline evidence."
        review_required = False
        confidence = 0.68
    else:
        disposition = "no_material_missingness"
        handling = "no_action_needed"
        skip_logic_protected = False
        notes = ""
        review_required = False
        confidence = 0.72

    if drift_detected and missing_pct >= 20.0:
        confidence = min(confidence, 0.7)
        review_required = True

    return {
        "missingness_disposition": disposition,
        "missingness_handling": handling,
        "skip_logic_protected": skip_logic_protected,
        "normalization_notes": notes,
        "missingness_decision_source": "a17_baseline",
        "missingness_review_required": review_required,
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "applied_sources": _dedupe_preserve_order(applied_sources),
    }


def _build_baseline_column_resolution_artifact(
    a2_rows: List[Dict[str, Any]],
    a3t_payload: Optional[Dict[str, Any]],
    a3v_payload: Optional[Dict[str, Any]],
    a4_payload: Optional[Dict[str, Any]],
    a9_payload: Optional[Dict[str, Any]],
    a13_payload: Optional[Dict[str, Any]],
    a14_payload: Optional[Dict[str, Any]],
    a16_payload: Optional[Dict[str, Any]],
    artifact_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    a3t_by_col = _normalize_transform_review_output(a3t_payload or {})
    a3v_by_col = _normalize_variable_type_review_output(a3v_payload or {})
    a4_by_col = _normalize_missing_catalog_output(a4_payload or {})
    a9_by_col = {
        str(item.get("column") or "").strip(): item
        for item in _coerce_list_of_dicts((a9_payload or {}).get("columns"))
        if str(item.get("column") or "").strip()
    }
    a13_by_col = {
        str(item.get("column") or "").strip(): item
        for item in _coerce_list_of_dicts((a13_payload or {}).get("columns"))
        if str(item.get("column") or "").strip()
    }
    a14_by_col = {
        str(item.get("column") or "").strip(): item
        for item in _coerce_list_of_dicts((a14_payload or {}).get("columns"))
        if str(item.get("column") or "").strip()
    }
    a16_by_col = _build_a16_column_context(a16_payload or {})

    rows: List[Dict[str, Any]] = []
    for a2_row in a2_rows:
        column = str(a2_row.get("column") or "").strip()
        if not column:
            continue
        a3t_row = a3t_by_col.get(column)
        a3v_row = a3v_by_col.get(column)
        a4_row = a4_by_col.get(column)
        a9_row = a9_by_col.get(column)
        a13_row = a13_by_col.get(column)
        a14_row = a14_by_col.get(column)
        a16_row = a16_by_col.get(column)

        type_decision = _baseline_type_decision(
            column=column,
            a2_row=a2_row,
            a3t_row=a3t_row,
            a3v_row=a3v_row,
            a4_row=a4_row,
            a9_row=a9_row,
            a13_row=a13_row,
            a14_row=a14_row,
        )
        missingness_decision = _baseline_missingness_decision(
            column=column,
            a2_row=a2_row,
            a4_row=a4_row,
            a16_row=a16_row,
            a14_row=a14_row,
        )

        row_needs_review = bool(
            type_decision["type_review_required"]
            or missingness_decision["missingness_review_required"]
            or (
                bool((a14_row or {}).get("drift_detected", False))
                and type_decision["recommended_logical_type"] in {"date", "datetime", "mixed_or_ambiguous"}
            )
        )

        rows.append(
            {
                "column": column,
                "a9_primary_role": str((a9_row or {}).get("primary_role") or ""),
                "recommended_logical_type": str(type_decision["recommended_logical_type"]),
                "recommended_storage_type": str(type_decision["recommended_storage_type"]),
                "transform_actions": list(type_decision["transform_actions"]),
                "structural_transform_hints": list(type_decision["structural_transform_hints"]),
                "interpretation_hints": list(type_decision["interpretation_hints"]),
                "missingness_disposition": str(missingness_decision["missingness_disposition"]),
                "missingness_handling": str(missingness_decision["missingness_handling"]),
                "skip_logic_protected": bool(missingness_decision["skip_logic_protected"]),
                "type_normalization_notes": str(type_decision.get("normalization_notes") or ""),
                "missingness_normalization_notes": str(missingness_decision.get("normalization_notes") or ""),
                "quality_score": round(_safe_float((a14_row or {}).get("global_quality_score")), 6) if isinstance((a14_row or {}).get("global_quality_score"), (int, float)) else None,
                "drift_detected": bool((a14_row or {}).get("drift_detected", False)) if isinstance((a14_row or {}).get("drift_detected"), bool) else None,
                "type_decision_source": str(type_decision["type_decision_source"]),
                "missingness_decision_source": str(missingness_decision["missingness_decision_source"]),
                "type_confidence": round(_safe_float(type_decision.get("confidence"), 0.0), 6),
                "missingness_confidence": round(_safe_float(missingness_decision.get("confidence"), 0.0), 6),
                "type_review_required": bool(type_decision["type_review_required"]),
                "missingness_review_required": bool(missingness_decision["missingness_review_required"]),
                "needs_human_review": row_needs_review,
                "confidence": _resolve_contract_confidence(
                    a2_row=a2_row,
                    type_confidence=_safe_float(type_decision.get("confidence"), 0.0),
                    missingness_confidence=_safe_float(missingness_decision.get("confidence"), 0.0),
                    needs_human_review=row_needs_review,
                    conflict_count=0,
                    unresolved=False,
                    drift_detected=(a14_row or {}).get("drift_detected") if isinstance((a14_row or {}).get("drift_detected"), bool) else None,
                ),
                "applied_sources": _dedupe_preserve_order(
                    list(type_decision.get("applied_sources") or []) + list(missingness_decision.get("applied_sources") or [])
                ),
            }
        )

    return {
        "artifact": "A17",
        "purpose": "baseline_column_resolution",
        "inputs": ((artifact_inputs or {}).get("A17") or {"uses": ["A2", "A3-T", "A3-V", "A4", "A9", "A13", "A14", "A16"]}),
        "summary": {
            "total_columns": len(rows),
            "type_review_candidate_count": sum(1 for row in rows if row.get("type_review_required")),
            "missingness_review_candidate_count": sum(1 for row in rows if row.get("missingness_review_required")),
            "skip_logic_protected_count": sum(1 for row in rows if row.get("skip_logic_protected")),
        },
        "columns": rows,
    }


def _normalize_existing_baseline_resolution(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for item in _coerce_list_of_dicts(payload.get("columns")):
        col = str(item.get("column") or "").strip()
        if col:
            rows[col] = item
    return rows


def _load_canonical_support_artifacts(run_id: str) -> Dict[str, Any]:
    kind, a2_payload, _ = _get_decoded_payload(run_id=run_id, artifact_id="A2")
    if kind != "jsonl" or not isinstance(a2_payload, list):
        raise HTTPException(status_code=422, detail="A2 must decode to a JSONL array for canonical contract synthesis")

    a2_rows = [row for row in a2_payload if isinstance(row, dict) and str(row.get("column") or "").strip()]
    if not a2_rows:
        raise HTTPException(status_code=422, detail="A2 does not contain any usable source columns")

    missing_artifacts: List[str] = []

    a3t_kind, a3t_payload, _, missing = _load_optional_decoded_payload(run_id, "A3-T")
    if missing:
        missing_artifacts.append(missing)
    a3v_kind, a3v_payload, _, missing = _load_optional_decoded_payload(run_id, "A3-V")
    if missing:
        missing_artifacts.append(missing)
    a4_kind, a4_payload, _, missing = _load_optional_decoded_payload(run_id, "A4")
    if missing:
        missing_artifacts.append(missing)
    a9_kind, a9_payload, _, missing = _load_optional_decoded_payload(run_id, "A9")
    if missing:
        missing_artifacts.append(missing)
    a13_kind, a13_payload, _, missing = _load_optional_decoded_payload(run_id, "A13")
    if missing:
        missing_artifacts.append(missing)
    a14_kind, a14_payload, _, missing = _load_optional_decoded_payload(run_id, "A14")
    if missing:
        missing_artifacts.append(missing)
    a16_kind, a16_payload, _, missing = _load_optional_decoded_payload(run_id, "A16")
    if missing:
        missing_artifacts.append(missing)
    a17_kind, a17_payload, _, _ = _load_optional_decoded_payload(run_id, "A17")

    a9_rows = _coerce_list_of_dicts((a9_payload or {}).get("columns")) if a9_kind == "json" and isinstance(a9_payload, dict) else []
    a13_rows = _coerce_list_of_dicts((a13_payload or {}).get("columns")) if a13_kind == "json" and isinstance(a13_payload, dict) else []
    a14_rows = _coerce_list_of_dicts((a14_payload or {}).get("columns")) if a14_kind == "json" and isinstance(a14_payload, dict) else []
    a16_payload_obj = a16_payload if a16_kind == "json" and isinstance(a16_payload, dict) else {}
    a16_context = _build_a16_column_context(a16_payload_obj)

    a17_by_col = _normalize_existing_baseline_resolution(a17_payload if a17_kind == "json" and isinstance(a17_payload, dict) else {})
    a17_backfilled = False
    if not a17_by_col:
        a17_backfilled = True
        synthesized_a17 = _build_baseline_column_resolution_artifact(
            a2_rows=a2_rows,
            a3t_payload=a3t_payload if a3t_kind == "json" and isinstance(a3t_payload, dict) else {},
            a3v_payload=a3v_payload if a3v_kind == "json" and isinstance(a3v_payload, dict) else {},
            a4_payload=a4_payload if a4_kind == "json" and isinstance(a4_payload, dict) else {},
            a9_payload=a9_payload if a9_kind == "json" and isinstance(a9_payload, dict) else {},
            a13_payload=a13_payload if a13_kind == "json" and isinstance(a13_payload, dict) else {},
            a14_payload=a14_payload if a14_kind == "json" and isinstance(a14_payload, dict) else {},
            a16_payload=a16_payload_obj,
        )
        a17_by_col = _normalize_existing_baseline_resolution(synthesized_a17)

    return {
        "a2_rows": a2_rows,
        "a2_by_col": {str(row.get("column")).strip(): row for row in a2_rows},
        "a2_order": [str(row.get("column")).strip() for row in a2_rows],
        "a9_by_col": {
            str(row.get("column")).strip(): row
            for row in a9_rows
            if str(row.get("column") or "").strip()
        },
        "a13_by_col": {
            str(row.get("column")).strip(): row
            for row in a13_rows
            if str(row.get("column") or "").strip()
        },
        "a14_by_col": {
            str(row.get("column")).strip(): row
            for row in a14_rows
            if str(row.get("column") or "").strip()
        },
        "a16_by_col": a16_context,
        "a17_by_col": a17_by_col,
        "family_by_column": _family_column_map(run_id),
        "missing_artifacts": missing_artifacts,
        "a17_backfilled": a17_backfilled,
    }


def _normalize_light_contract_maps(decisions: Dict[str, Any]) -> Dict[str, Any]:
    primary_keys = _dedupe_preserve_order((decisions.get("primary_grain_decision") or {}).get("keys") or [])

    reference_table_by_key: Dict[str, str] = {}
    for ref in _coerce_list_of_dicts(decisions.get("reference_decisions") or decisions.get("dimension_decisions")):
        table_name = str(ref.get("table_name") or "").strip()
        for key in ref.get("keys") or []:
            col = str(key or "").strip()
            if col and table_name and col not in reference_table_by_key:
                reference_table_by_key[col] = table_name

    family_by_id: Dict[str, Dict[str, Any]] = {}
    for family in _coerce_list_of_dicts(decisions.get("family_decisions")):
        family_id = str(family.get("family_id") or "").strip()
        if family_id:
            family_by_id[family_id] = family

    return {
        "primary_keys": primary_keys,
        "reference_table_by_key": reference_table_by_key,
        "family_by_id": family_by_id,
    }


def _build_mapping_entry_from_human_input(entry: Dict[str, Any]) -> Dict[str, Any]:
    ordered_labels = [str(label).strip() for label in (entry.get("ordered_labels") or []) if str(label or "").strip()]
    numeric_scores = [float(score) for score in (entry.get("numeric_scores") or [])]
    numeric_map = (
        {label: score for label, score in zip(ordered_labels, numeric_scores)}
        if ordered_labels and numeric_scores and len(ordered_labels) == len(numeric_scores)
        else {}
    )
    response_scale_kind = str(entry.get("response_scale_kind") or "").strip() or _infer_scale_kind(ordered_labels, str(entry.get("notes") or ""))
    mapping_status = "human_confirmed" if ordered_labels or numeric_map or response_scale_kind else "unresolved"
    return {
        "target_kind": str(entry.get("target_kind") or "").strip(),
        "target_id": str(entry.get("target_id") or "").strip(),
        "mapping_status": mapping_status,
        "response_scale_kind": response_scale_kind,
        "ordered_labels": ordered_labels,
        "label_to_ordinal_position": {label: idx + 1 for idx, label in enumerate(ordered_labels)},
        "label_to_numeric_score": numeric_map,
        "numeric_score_semantics_confirmed": bool(numeric_map),
        "source": "light_contract_scale_mapping",
        "notes": str(entry.get("notes") or "").strip() or "Structured mapping supplied through the light contract workbook.",
        "confidence": 0.99 if ordered_labels else 0.85,
    }


def _collect_scale_mapping_candidates(
    *,
    support: Dict[str, Any],
    family_results_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reverse_family_map = _reverse_family_column_map(support["family_by_column"])

    family_candidates: List[Dict[str, Any]] = []
    for family_id, member_columns in sorted(reverse_family_map.items()):
        observed_values: List[str] = []
        for column in member_columns[:6]:
            observed_values.extend(_preview_values_from_a2_row(support["a2_by_col"].get(column) or {}, limit=4))
        observed_values = _dedupe_preserve_order(observed_values)
        if not observed_values or not _looks_scale_like_values(observed_values):
            continue
        family_result = family_results_by_id.get(family_id) or {}
        family_candidates.append(
            {
                "target_kind": "family",
                "target_id": family_id,
                "observed_values": observed_values,
                "semantic_text": " ".join(
                    bit
                    for bit in [
                        str(family_result.get("member_semantics_notes") or "").strip(),
                        str(family_result.get("recommended_family_role") or "").strip(),
                    ]
                    if bit
                ),
            }
        )

    column_candidates: List[Dict[str, Any]] = []
    for a2_row in support["a2_rows"]:
        column = str(a2_row.get("column") or "").strip()
        if not column or column in support["family_by_column"]:
            continue
        observed_values = _preview_values_from_a2_row(a2_row, limit=8)
        if not observed_values or not _looks_scale_like_values(observed_values):
            continue
        column_candidates.append(
            {
                "target_kind": "column",
                "target_id": column,
                "observed_values": observed_values,
                "semantic_text": "",
            }
        )

    return family_candidates, column_candidates


def _build_scale_mapping_contract(
    *,
    run_id: str,
    light_contract_decisions: Dict[str, Any],
    family_worker_json: Any,
    scale_mapping_extractor_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    support = _load_canonical_support_artifacts(run_id)
    light_contract_maps = _normalize_light_contract_maps(light_contract_decisions)
    family_results_by_id = _extract_family_results(family_worker_json)
    known_family_ids = set((light_contract_maps.get("family_by_id") or {}).keys()) | set(family_results_by_id.keys()) | set(support["family_by_column"].values())
    known_columns = set(support["a2_by_col"].keys())

    resolved_by_target: Dict[Tuple[str, str], Dict[str, Any]] = {}
    review_flags: List[Dict[str, Any]] = []

    human_entries = _normalize_light_contract_scale_mapping_input(light_contract_decisions.get("scale_mapping_input") or [])
    for entry in human_entries:
        target_kind = str(entry.get("target_kind") or "").strip()
        target_id = str(entry.get("target_id") or "").strip()
        if target_kind == "family" and target_id not in known_family_ids:
            _add_review_flag(review_flags, target_id, "unknown_scale_mapping_target", "Structured light-contract mapping target does not resolve to a known family id.")
            continue
        if target_kind == "column" and target_id not in known_columns:
            _add_review_flag(review_flags, target_id, "unknown_scale_mapping_target", "Structured light-contract mapping target does not resolve to a known source column.")
            continue
        resolved_by_target[(target_kind, target_id)] = _build_mapping_entry_from_human_input(entry)

    extractor_index = _normalize_scale_mapping_contract(
        scale_mapping_extractor_json or {},
        known_columns=known_columns,
        known_family_ids=known_family_ids,
    )
    for extra_target in sorted(extractor_index.get("extra_targets") or set()):
        _add_review_flag(review_flags, extra_target, "scale_mapping_target_not_in_scope", "Extractor proposed a mapping target that does not resolve to a known family id or source column.")
    for target_key, mapping in (extractor_index.get("by_target") or {}).items():
        if target_key in resolved_by_target:
            continue
        if mapping.get("mapping_status") == "unresolved" and not mapping.get("ordered_labels"):
            continue
        resolved_by_target[target_key] = dict(mapping)

    family_candidates, column_candidates = _collect_scale_mapping_candidates(
        support=support,
        family_results_by_id=family_results_by_id,
    )
    for candidate in family_candidates + column_candidates:
        target_key = (candidate["target_kind"], candidate["target_id"])
        if target_key in resolved_by_target:
            continue
        inferred = _deterministic_scale_inference(
            target_kind=candidate["target_kind"],
            target_id=candidate["target_id"],
            observed_values=candidate["observed_values"],
            semantic_text=candidate["semantic_text"],
        )
        if inferred:
            resolved_by_target[target_key] = inferred
        else:
            resolved_by_target[target_key] = {
                "target_kind": candidate["target_kind"],
                "target_id": candidate["target_id"],
                "mapping_status": "unresolved",
                "response_scale_kind": _infer_scale_kind(candidate["observed_values"], candidate["semantic_text"]),
                "ordered_labels": [],
                "label_to_ordinal_position": {},
                "label_to_numeric_score": {},
                "numeric_score_semantics_confirmed": False,
                "source": "resolver_unresolved",
                "notes": "Observed values suggest ordered-scale semantics, but direction or scoring remains ambiguous.",
                "confidence": 0.45,
            }

    mappings = sorted(
        resolved_by_target.values(),
        key=lambda item: (str(item.get("target_kind") or ""), str(item.get("target_id") or "")),
    )
    output = {
        "worker": "scale_mapping_resolver",
        "summary": {
            "overview": f"Resolved {len(mappings)} family- or column-scoped scale mappings before canonical synthesis using light-contract mappings, optional codebook extraction, and deterministic inference.",
            "mapping_count": len(mappings),
            "human_confirmed_count": sum(1 for item in mappings if item.get("mapping_status") == "human_confirmed"),
            "codebook_confirmed_count": sum(1 for item in mappings if item.get("mapping_status") == "codebook_confirmed"),
            "deterministic_inferred_count": sum(1 for item in mappings if item.get("mapping_status") == "deterministic_inferred"),
            "unresolved_count": sum(1 for item in mappings if item.get("mapping_status") == "unresolved"),
            "key_points": [
                "Scale mappings are reused by canonical synthesis and later analysis derivation planning.",
                "Raw source response labels remain string-backed in canon even when score mappings are confirmed.",
                "Unresolved mappings remain non-blocking and preserve later review cues.",
            ],
        },
        "mappings": mappings,
        "review_flags": review_flags,
        "assumptions": [
            {
                "assumption": "human_codebook_deterministic_precedence",
                "explanation": "Resolver precedence is light-contract structured mappings first, validated extractor output second, deterministic safe inference third, unresolved last.",
            }
        ],
    }
    errors = _validate_scale_mapping_contract(
        output,
        known_columns=known_columns,
        known_family_ids=known_family_ids,
    )
    if errors:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "scale mapping resolver produced an invalid payload",
                "errors": errors,
            },
        )
    return output


def _normalize_type_output(payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    decisions: Dict[str, Dict[str, Any]] = {}
    for item in _coerce_list_of_dicts(payload.get("column_decisions")):
        col = str(item.get("column") or "").strip()
        if col:
            decisions[col] = item

    deduped_rules: List[Dict[str, Any]] = []
    seen_rule_names: Set[str] = set()
    for item in _coerce_list_of_dicts(payload.get("global_transform_rules")):
        rule_name = str(item.get("rule_name") or "").strip()
        applies_when = str(item.get("applies_when") or "").strip()
        description = str(item.get("rule_description") or "").strip()
        if not (rule_name and applies_when and description) or rule_name in seen_rule_names:
            continue
        seen_rule_names.add(rule_name)
        deduped_rules.append(
            {
                "rule_name": rule_name,
                "applies_when": applies_when,
                "rule_description": description,
            }
        )

    return decisions, deduped_rules


def _normalize_missingness_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    decisions: Dict[str, Dict[str, Any]] = {}
    for item in _coerce_list_of_dicts(payload.get("column_decisions")):
        col = str(item.get("column") or "").strip()
        if col:
            decisions[col] = item

    global_contract_raw = payload.get("global_contract") if isinstance(payload, dict) else {}
    global_contract = {
        "token_missing_placeholders_detected": (
            global_contract_raw.get("token_missing_placeholders_detected")
            if isinstance(global_contract_raw, dict) and isinstance(global_contract_raw.get("token_missing_placeholders_detected"), bool)
            else None
        ),
        "notes": (
            str(global_contract_raw.get("notes") or "").strip()
            if isinstance(global_contract_raw, dict)
            else ""
        ),
    }

    return {
        "column_decisions": decisions,
        "global_contract": global_contract,
        "global_findings": _coerce_list_of_dicts(payload.get("global_findings")),
    }


def _extract_family_results(payload: Any) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    def _maybe_add(item: Any) -> None:
        if not isinstance(item, dict):
            return
        candidate = item.get("family_result") if isinstance(item.get("family_result"), dict) else item
        family_id = str(candidate.get("family_id") or "").strip()
        if family_id and family_id not in results:
            results[family_id] = candidate

    if isinstance(payload, dict):
        if isinstance(payload.get("family_results"), list):
            for item in payload.get("family_results") or []:
                _maybe_add(item)
        else:
            _maybe_add(payload)
        for key in ("items", "results"):
            if isinstance(payload.get(key), list):
                for item in payload.get(key) or []:
                    _maybe_add(item)
    elif isinstance(payload, list):
        for item in payload:
            _maybe_add(item)

    return results


def _normalize_family_member_defaults(payload: Any) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    def _maybe_add(item: Any) -> None:
        if not isinstance(item, dict):
            return
        candidate = item.get("family_result") if isinstance(item.get("family_result"), dict) else item
        family_id = str(candidate.get("family_id") or item.get("family_id") or "").strip()
        raw_defaults = item.get("member_defaults")
        if not isinstance(raw_defaults, dict) and isinstance(candidate.get("member_defaults"), dict):
            raw_defaults = candidate.get("member_defaults")
        if not family_id or not isinstance(raw_defaults, dict):
            return

        provided_fields: List[str] = []
        normalized: Dict[str, Any] = {}

        if str(raw_defaults.get("recommended_logical_type") or "").strip() in TYPE_LOGICAL_TYPES:
            normalized["recommended_logical_type"] = str(raw_defaults.get("recommended_logical_type")).strip()
            provided_fields.append("recommended_logical_type")
        if str(raw_defaults.get("recommended_storage_type") or "").strip() in TYPE_STORAGE_TYPES:
            normalized["recommended_storage_type"] = str(raw_defaults.get("recommended_storage_type")).strip()
            provided_fields.append("recommended_storage_type")
        if "transform_actions" in raw_defaults and isinstance(raw_defaults.get("transform_actions"), list):
            normalized["transform_actions"] = _dedupe_preserve_order(
                action for action in raw_defaults.get("transform_actions") or [] if str(action or "").strip() in TYPE_TRANSFORM_ACTIONS
            )
            provided_fields.append("transform_actions")
        if "structural_transform_hints" in raw_defaults and isinstance(raw_defaults.get("structural_transform_hints"), list):
            normalized["structural_transform_hints"] = _dedupe_preserve_order(
                hint for hint in raw_defaults.get("structural_transform_hints") or [] if str(hint or "").strip() in TYPE_STRUCTURAL_HINTS
            )
            provided_fields.append("structural_transform_hints")
        if "interpretation_hints" in raw_defaults and isinstance(raw_defaults.get("interpretation_hints"), list):
            normalized["interpretation_hints"] = _dedupe_preserve_order(
                hint for hint in raw_defaults.get("interpretation_hints") or [] if str(hint or "").strip() in TYPE_INTERPRETATION_HINTS
            )
            provided_fields.append("interpretation_hints")
        raw_missingness_disposition = str(raw_defaults.get("missingness_disposition") or "").strip()
        family_missingness_allowed = raw_missingness_disposition in FAMILY_DEFAULT_ALLOWED_MISSINGNESS_DISPOSITIONS
        if family_missingness_allowed:
            normalized["missingness_disposition"] = raw_missingness_disposition
            provided_fields.append("missingness_disposition")
        if family_missingness_allowed and str(raw_defaults.get("missingness_handling") or "").strip() in MISSINGNESS_HANDLING:
            normalized["missingness_handling"] = str(raw_defaults.get("missingness_handling")).strip()
            provided_fields.append("missingness_handling")
        if family_missingness_allowed and "skip_logic_protected" in raw_defaults and isinstance(raw_defaults.get("skip_logic_protected"), bool):
            normalized["skip_logic_protected"] = bool(raw_defaults.get("skip_logic_protected"))
            provided_fields.append("skip_logic_protected")
        if "normalization_notes" in raw_defaults:
            normalized["normalization_notes"] = str(raw_defaults.get("normalization_notes") or "").strip()
            provided_fields.append("normalization_notes")

        normalized["confidence"] = round(max(0.0, min(1.0, _safe_float(raw_defaults.get("confidence"), 0.74))), 6)
        normalized["needs_human_review"] = bool(raw_defaults.get("needs_human_review", False))
        normalized["applied_sources"] = ["family_worker.member_defaults"]
        normalized["provided_fields"] = provided_fields

        if provided_fields:
            results[family_id] = normalized

    if isinstance(payload, dict):
        if isinstance(payload.get("family_results"), list):
            for item in payload.get("family_results") or []:
                _maybe_add(item)
        else:
            _maybe_add(payload)
        for key in ("items", "results"):
            if isinstance(payload.get(key), list):
                for item in payload.get(key) or []:
                    _maybe_add(item)
    elif isinstance(payload, list):
        for item in payload:
            _maybe_add(item)

    return results


def _normalize_table_layout_output(payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    assignments_by_col: Dict[str, Dict[str, Any]] = {}
    for row in _coerce_list_of_dicts(payload.get("column_table_assignments")):
        col = str(row.get("column") or "").strip()
        if col:
            assignments_by_col[col] = row

    tables_by_name: Dict[str, Dict[str, Any]] = {}
    for table in _coerce_list_of_dicts(payload.get("table_suggestions")):
        name = str(table.get("table_name") or "").strip()
        if name:
            tables_by_name[name] = table

    return assignments_by_col, tables_by_name


def _normalize_semantic_context(
    payload: Dict[str, Any],
    known_columns: Set[str],
    known_family_ids: Set[str],
) -> Dict[str, Any]:
    if payload.get("status") == "skipped":
        return {
            "skip": True,
            "skip_reason": str(payload.get("reason") or "").strip(),
            "important_by_column": {},
            "important_by_family": {},
            "codebook_by_column": {},
            "extra_columns": set(),
        }

    important_by_column: Dict[str, Dict[str, Any]] = {}
    important_by_family: Dict[str, Dict[str, Any]] = {}
    codebook_by_column: Dict[str, Dict[str, Any]] = {}
    extra_columns: Set[str] = set()

    for item in _coerce_list_of_dicts(payload.get("important_variables")):
        target = str(item.get("column_or_family") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not target:
            continue
        if kind == "family_context" or (target in known_family_ids and target not in known_columns):
            important_by_family[target] = _pick_higher_confidence(important_by_family.get(target), item)
        elif target in known_columns:
            important_by_column[target] = _pick_higher_confidence(important_by_column.get(target), item)
        else:
            extra_columns.add(target)

    for item in _coerce_list_of_dicts(payload.get("codebook_hints")):
        target = str(item.get("column") or "").strip()
        if not target:
            continue
        if target in known_columns:
            codebook_by_column[target] = _pick_higher_confidence(codebook_by_column.get(target), item)
        else:
            extra_columns.add(target)

    return {
        "skip": False,
        "skip_reason": "",
        "important_by_column": important_by_column,
        "important_by_family": important_by_family,
        "codebook_by_column": codebook_by_column,
        "extra_columns": extra_columns,
    }


def _resolve_scale_mapping_for_column(
    column: str,
    source_family_id: str,
    scale_mapping_index: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    by_target = scale_mapping_index.get("by_target") or {}
    column_mapping = by_target.get(("column", column))
    if column_mapping:
        return column_mapping
    if source_family_id:
        return by_target.get(("family", source_family_id))
    return None


def _rewrite_notes_with_scale_mapping(existing_notes: str, mapping: Dict[str, Any]) -> str:
    text = str(existing_notes or "").strip()
    stale_patterns = [
        r"\bpost-codebook review\b\.?",
        r"\baplicar validated label->score mapping post-codebook review\b\.?",
        r"\bvalidate numeric-label mapping before casting to numeric\b\.?",
        r"\bvalidate numeric-label mapping post-codebook review\b\.?",
    ]
    for pattern in stale_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" .")

    label_preview = " | ".join((mapping.get("ordered_labels") or [])[:5])
    if mapping.get("mapping_status") in {"human_confirmed", "codebook_confirmed"}:
        suffix = (
            f"Structured scale mapping confirmed ({mapping.get('mapping_status')}); preserve raw labels in canon"
            f"{' and defer numeric scoring to derived transforms' if mapping.get('numeric_score_semantics_confirmed') else ''}."
        )
        if label_preview:
            suffix = f"{suffix} Ordered labels: {label_preview}."
    elif mapping.get("mapping_status") == "deterministic_inferred":
        suffix = "Deterministic ordered-scale inference is available, but numeric scoring remains advisory until human or codebook confirmation."
    else:
        suffix = ""

    return _merge_notes(text, suffix)


def _apply_scale_mapping_to_type_and_semantics(
    *,
    type_decision: Dict[str, Any],
    semantic: Dict[str, Any],
    mapping: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    if not mapping:
        return dict(type_decision), dict(semantic), False

    updated_type = dict(type_decision)
    updated_semantic = dict(semantic)
    ordered_labels = [str(label).strip() for label in (mapping.get("ordered_labels") or []) if str(label or "").strip()]
    mapping_status = str(mapping.get("mapping_status") or "").strip()
    confirmed = mapping_status in {"human_confirmed", "codebook_confirmed"}
    materially_applied = False

    if ordered_labels or str(mapping.get("response_scale_kind") or "").strip():
        updated_type["recommended_logical_type"] = "ordinal_category"
        updated_type["recommended_storage_type"] = "string"
        preserved_actions = [
            action
            for action in (updated_type.get("transform_actions") or [])
            if action in {"trim_whitespace", "normalize_missing_tokens", "lowercase_values", "uppercase_values", "titlecase_values"}
        ]
        updated_type["transform_actions"] = _dedupe_preserve_order(
            ["trim_whitespace", "normalize_category_tokens"] + preserved_actions
        )
        structural_hints = list(updated_type.get("structural_transform_hints") or [])
        if confirmed:
            structural_hints = [hint for hint in structural_hints if hint != "requires_codebook_or_label_mapping_review"]
        updated_type["structural_transform_hints"] = _dedupe_preserve_order(structural_hints)
        interpretation_hints = list(updated_type.get("interpretation_hints") or [])
        if any(re.search(r"\d", label) for label in ordered_labels):
            interpretation_hints.append("numeric_parse_is_misleading")
        updated_type["interpretation_hints"] = _dedupe_preserve_order(interpretation_hints)
        updated_type["normalization_notes"] = _rewrite_notes_with_scale_mapping(
            str(updated_type.get("normalization_notes") or ""),
            mapping,
        )
        updated_type["confidence"] = round(max(_safe_float(updated_type.get("confidence"), 0.0), _safe_float(mapping.get("confidence"), 0.0)), 6)
        updated_type["applied_sources"] = _dedupe_preserve_order(list(updated_type.get("applied_sources") or []) + ["scale_mapping_resolver"])
        updated_type["type_decision_source"] = "scale_mapping_resolver"
        materially_applied = True

    if ordered_labels:
        response_scale_kind = str(mapping.get("response_scale_kind") or "").strip().replace("_", " ")
        if not str(updated_semantic.get("semantic_meaning") or "").strip():
            updated_semantic["semantic_meaning"] = f"Ordered response scale ({response_scale_kind or 'ordinal scale'}) with labels preserved as raw categories."
        label_note = f"Ordered labels: {' | '.join(ordered_labels[:6])}."
        if confirmed:
            note_prefix = "Confirmed scale mapping."
        elif mapping_status == "deterministic_inferred":
            note_prefix = "Deterministically inferred scale ordering."
        else:
            note_prefix = "Scale mapping remains unresolved."
        updated_semantic["codebook_note"] = _merge_notes(str(updated_semantic.get("codebook_note") or ""), f"{note_prefix} {label_note}")
        updated_semantic["semantic_decision_source"] = "scale_mapping_resolver"
        materially_applied = True

    return updated_type, updated_semantic, materially_applied


def _resolve_structure_for_column(
    column: str,
    assignments_by_col: Dict[str, Dict[str, Any]],
    tables_by_name: Dict[str, Dict[str, Any]],
    light_contract_maps: Dict[str, Any],
    family_by_column: Dict[str, str],
) -> Dict[str, Any]:
    assignment = assignments_by_col.get(column)
    if assignment:
        assigned_table = str(assignment.get("assigned_table") or "").strip()
        assignment_role = str(assignment.get("assignment_role") or "unresolved").strip() or "unresolved"
        source_family_id = str(assignment.get("source_family_id") or "").strip() or family_by_column.get(column, "")
        table_kind = str((tables_by_name.get(assigned_table) or {}).get("kind") or "").strip()
        modeling_status = "base_field"
        if assignment_role == "exclude_from_outputs":
            modeling_status = "excluded_from_outputs"
            assigned_table = ""
        elif assignment_role == "unresolved":
            modeling_status = "unresolved"
            assigned_table = ""
        elif table_kind == "event_table":
            modeling_status = "event_field"
        elif assignment_role in {"reference_key", "reference_attribute", "reference_value"} or table_kind == "reference_lookup":
            modeling_status = "reference_field"
        elif assignment_role in {"repeat_parent_key", "repeat_index", "melt_member"} or table_kind == "child_repeat":
            modeling_status = "child_repeat_member"
        return {
            "canonical_modeling_status": modeling_status,
            "canonical_table_name": assigned_table,
            "canonical_assignment_role": assignment_role if assignment_role in CANONICAL_ASSIGNMENT_ROLES else "unresolved",
            "source_family_id": source_family_id,
            "structure_decision_source": "table_layout_worker",
            "table_kind": table_kind,
        }

    reference_table = str((light_contract_maps.get("reference_table_by_key") or {}).get(column) or "").strip()
    if reference_table:
        return {
            "canonical_modeling_status": "reference_field",
            "canonical_table_name": reference_table,
            "canonical_assignment_role": "reference_key",
            "source_family_id": "",
            "structure_decision_source": "light_contract_fallback",
            "table_kind": "reference_lookup",
        }

    family_id = family_by_column.get(column, "")
    family_decision = (light_contract_maps.get("family_by_id") or {}).get(family_id) or {}
    family_table_name = str(family_decision.get("table_name") or "").strip()
    if family_id and family_table_name:
        return {
            "canonical_modeling_status": "child_repeat_member",
            "canonical_table_name": family_table_name,
            "canonical_assignment_role": "melt_member",
            "source_family_id": family_id,
            "structure_decision_source": "light_contract_fallback",
            "table_kind": "child_repeat",
        }

    return {
        "canonical_modeling_status": "unresolved",
        "canonical_table_name": "",
        "canonical_assignment_role": "unresolved",
        "source_family_id": family_id,
        "structure_decision_source": "unresolved",
        "table_kind": "",
    }


def _resolve_semantic_enrichment(
    column: str,
    source_family_id: str,
    semantic_index: Dict[str, Any],
    family_results_by_id: Dict[str, Dict[str, Any]],
    a13_row: Dict[str, Any],
) -> Dict[str, Any]:
    codebook_hint = semantic_index["codebook_by_column"].get(column)
    if codebook_hint:
        return {
            "semantic_meaning": str(codebook_hint.get("meaning") or "").strip(),
            "codebook_note": str(codebook_hint.get("codes_or_labels_note") or "").strip(),
            "semantic_decision_source": "semantic_context_worker",
            "semantic_kind": "codebook_hint",
        }

    important = semantic_index["important_by_column"].get(column)
    if important:
        return {
            "semantic_meaning": str(important.get("meaning") or "").strip(),
            "codebook_note": "",
            "semantic_decision_source": "semantic_context_worker",
            "semantic_kind": str(important.get("kind") or "").strip(),
        }

    if source_family_id:
        family_important = semantic_index["important_by_family"].get(source_family_id)
        if family_important:
            return {
                "semantic_meaning": str(family_important.get("meaning") or "").strip(),
                "codebook_note": "",
                "semantic_decision_source": "semantic_context_worker",
                "semantic_kind": str(family_important.get("kind") or "").strip(),
            }
        family_result = family_results_by_id.get(source_family_id) or {}
        family_note = str(family_result.get("member_semantics_notes") or "").strip()
        if family_note:
            return {
                "semantic_meaning": family_note,
                "codebook_note": "",
            "semantic_decision_source": "family_worker",
            "semantic_kind": "family_result",
        }

    return {
        "semantic_meaning": "",
        "codebook_note": "",
        "semantic_decision_source": "unknown",
        "semantic_kind": "",
    }


def _baseline_row_to_type_decision(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {
            "recommended_logical_type": "mixed_or_ambiguous",
            "recommended_storage_type": "string",
            "transform_actions": [],
            "structural_transform_hints": [],
            "interpretation_hints": ["mixed_content_high_risk"],
            "normalization_notes": "",
            "type_decision_source": "unresolved_no_a2_evidence",
            "type_review_required": True,
            "confidence": 0.5,
            "applied_sources": [],
        }

    return {
        "recommended_logical_type": str(row.get("recommended_logical_type") or "mixed_or_ambiguous"),
        "recommended_storage_type": str(row.get("recommended_storage_type") or "string"),
        "transform_actions": list(row.get("transform_actions") or []),
        "structural_transform_hints": list(row.get("structural_transform_hints") or []),
        "interpretation_hints": list(row.get("interpretation_hints") or []),
        "normalization_notes": str(row.get("type_normalization_notes") or ""),
        "type_decision_source": str(row.get("type_decision_source") or "a17_baseline"),
        "type_review_required": bool(row.get("type_review_required", False)),
        "confidence": _safe_float(row.get("type_confidence"), _safe_float(row.get("confidence"), 0.0)),
        "applied_sources": list(row.get("applied_sources") or []),
    }


def _baseline_row_to_missingness_decision(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {
            "missingness_disposition": "mixed_missingness_risk",
            "missingness_handling": "review_before_drop",
            "skip_logic_protected": False,
            "normalization_notes": "",
            "missingness_decision_source": "unresolved_no_a2_evidence",
            "missingness_review_required": True,
            "confidence": 0.5,
            "applied_sources": [],
        }

    return {
        "missingness_disposition": str(row.get("missingness_disposition") or "mixed_missingness_risk"),
        "missingness_handling": str(row.get("missingness_handling") or "review_before_drop"),
        "skip_logic_protected": bool(row.get("skip_logic_protected", False)),
        "normalization_notes": str(row.get("missingness_normalization_notes") or ""),
        "missingness_decision_source": str(row.get("missingness_decision_source") or "a17_baseline"),
        "missingness_review_required": bool(row.get("missingness_review_required", False)),
        "confidence": _safe_float(row.get("missingness_confidence"), _safe_float(row.get("confidence"), 0.0)),
        "applied_sources": list(row.get("applied_sources") or []),
    }


def _apply_family_type_defaults(base_decision: Dict[str, Any], family_defaults: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if not family_defaults:
        return dict(base_decision), False

    provided = set(family_defaults.get("provided_fields") or [])
    type_fields = {
        "recommended_logical_type",
        "recommended_storage_type",
        "transform_actions",
        "structural_transform_hints",
        "interpretation_hints",
        "normalization_notes",
    }
    type_substantive_fields = {
        "recommended_logical_type",
        "recommended_storage_type",
        "transform_actions",
        "structural_transform_hints",
        "interpretation_hints",
    }
    if not (provided & type_substantive_fields):
        return dict(base_decision), False

    merged = dict(base_decision)
    if "recommended_logical_type" in provided:
        merged["recommended_logical_type"] = str(family_defaults.get("recommended_logical_type") or merged["recommended_logical_type"])
    if "recommended_storage_type" in provided:
        merged["recommended_storage_type"] = str(family_defaults.get("recommended_storage_type") or merged["recommended_storage_type"])
    if "transform_actions" in provided:
        merged["transform_actions"] = list(family_defaults.get("transform_actions") or [])
    if "structural_transform_hints" in provided:
        merged["structural_transform_hints"] = list(family_defaults.get("structural_transform_hints") or [])
    if "interpretation_hints" in provided:
        merged["interpretation_hints"] = list(family_defaults.get("interpretation_hints") or [])
    if "normalization_notes" in provided:
        merged["normalization_notes"] = str(family_defaults.get("normalization_notes") or "")

    merged["type_decision_source"] = "family_default"
    merged["type_review_required"] = bool(family_defaults.get("needs_human_review", merged.get("type_review_required", False)))
    merged["confidence"] = round(max(_safe_float(merged.get("confidence"), 0.0), _safe_float(family_defaults.get("confidence"), 0.74)), 6)
    merged["applied_sources"] = list(family_defaults.get("applied_sources") or ["family_worker.member_defaults"])
    return merged, True


def _apply_family_missingness_defaults(
    base_decision: Dict[str, Any],
    family_defaults: Dict[str, Any],
    *,
    global_token_missing_placeholders_detected: Optional[bool] = None,
) -> Tuple[Dict[str, Any], bool]:
    if not family_defaults:
        return dict(base_decision), False

    provided = set(family_defaults.get("provided_fields") or [])
    missingness_fields = {
        "missingness_disposition",
        "missingness_handling",
        "skip_logic_protected",
        "normalization_notes",
    }
    missingness_substantive_fields = {
        "missingness_disposition",
        "missingness_handling",
        "skip_logic_protected",
    }
    if not (provided & missingness_substantive_fields):
        return dict(base_decision), False

    merged = dict(base_decision)
    if "missingness_disposition" in provided:
        merged["missingness_disposition"] = str(family_defaults.get("missingness_disposition") or merged["missingness_disposition"])
    if "missingness_handling" in provided:
        merged["missingness_handling"] = str(family_defaults.get("missingness_handling") or merged["missingness_handling"])
    if "skip_logic_protected" in provided:
        merged["skip_logic_protected"] = bool(family_defaults.get("skip_logic_protected", merged["skip_logic_protected"]))
    if "normalization_notes" in provided:
        merged["normalization_notes"] = str(family_defaults.get("normalization_notes") or "")

    merged["missingness_decision_source"] = "family_default"
    merged["missingness_review_required"] = bool(family_defaults.get("needs_human_review", merged.get("missingness_review_required", False)))
    merged["confidence"] = round(max(_safe_float(merged.get("confidence"), 0.0), _safe_float(family_defaults.get("confidence"), 0.74)), 6)
    merged["applied_sources"] = list(family_defaults.get("applied_sources") or ["family_worker.member_defaults"])
    merged, _, _ = _normalize_canonical_missingness_decision(
        merged,
        base_decision=base_decision,
        global_token_missing_placeholders_detected=global_token_missing_placeholders_detected,
    )
    return merged, True


def _sync_skip_logic_interpretation_hints(interpretation_hints: List[str], skip_logic_protected: bool) -> List[str]:
    cleaned = [hint for hint in interpretation_hints if hint != "skip_logic_protected"]
    if skip_logic_protected:
        cleaned.append("skip_logic_protected")
    return _dedupe_preserve_order(cleaned)


def _fallback_type_decision(
    column: str,
    a2_row: Dict[str, Any],
    a9_row: Dict[str, Any],
    structure: Dict[str, Any],
    semantic: Dict[str, Any],
    family_result: Dict[str, Any],
) -> Dict[str, Any]:
    top_candidate = a2_row.get("top_candidate") or {}
    top_type = str(top_candidate.get("type") or "").strip()
    unique_ratio = _safe_float(a2_row.get("unique_ratio"), 0.0)
    unique_count = int(a2_row.get("unique_count") or 0)
    numeric_pct = _safe_float((a2_row.get("numeric_profile") or {}).get("parseable_pct"), 0.0)
    datetime_pct = _safe_float((a2_row.get("datetime_profile") or {}).get("parseable_pct"), 0.0)
    top_levels = [str(value).strip() for value in (a2_row.get("top_levels") or []) if str(value).strip()]
    samples = ((a2_row.get("a2_samples") or {}).get("random") or [])[:8]
    role = str(a9_row.get("primary_role") or "").strip()
    encoding_type = str(a9_row.get("encoding_type") or "").strip()
    code_semantics = bool(semantic.get("codebook_note") or semantic.get("semantic_kind") in {"codebook_hint", "code_column"})
    placeholder_context = semantic.get("semantic_kind") == "placeholder_value_context"
    explicit_missing_tokens = bool(a2_row.get("missing_tokens_observed")) and not placeholder_context
    semantic_blob = _semantic_text_blob(semantic, family_result)
    free_text_name_hint = _column_name_has_free_text_hint(column)
    textual_value_hint = _values_look_textual(top_levels + [str(value) for value in samples])
    repeat_survey_semantics = _family_semantics_imply_categorical_repeat(structure, semantic, family_result)

    transform_actions: List[str] = ["trim_whitespace"]
    structural_hints: List[str] = []
    interpretation_hints: List[str] = []
    normalization_notes = ""
    logical_type = "mixed_or_ambiguous"
    storage_type = "string"

    if structure.get("canonical_assignment_role") in {"base_key", "reference_key", "repeat_parent_key"} or role == "id_key":
        logical_type = "identifier"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["identifier_not_measure"]
        if numeric_pct >= 80.0:
            interpretation_hints.append("numeric_parse_is_misleading")
        if _has_leading_zero_risk(top_levels + [str(value) for value in samples]):
            interpretation_hints.append("leading_zero_risk")
        normalization_notes = "Fallback typing preserves identifier semantics as string storage."
    elif structure.get("canonical_assignment_role") == "repeat_index" or role == "repeat_index":
        logical_type = "ordinal_category" if _looks_integer_like(top_levels) else "categorical_code"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"] if _looks_numeric_like(top_levels) else ["trim_whitespace", "normalize_category_tokens"]
        interpretation_hints = ["repeat_context_do_not_use_as_base_key"]
        if _looks_numeric_like(top_levels):
            interpretation_hints.append("numeric_parse_is_misleading")
        normalization_notes = "Fallback typing preserves repeat-index semantics without promoting the field to a base identifier."
    elif role == "time_index" or top_type in {"date", "datetime"} or datetime_pct >= 80.0:
        logical_type = "datetime" if top_type == "datetime" or "time" in column.lower() or "timestamp" in column.lower() else "date"
        storage_type = logical_type
        transform_actions = ["trim_whitespace", "cast_to_datetime" if logical_type == "datetime" else "cast_to_date"]
        interpretation_hints = ["time_index_not_identifier"]
        normalization_notes = "Fallback typing preserves temporal semantics from parse evidence and role signals."
    elif top_type in {"numeric_range", "range_like"}:
        logical_type = "mixed_or_ambiguous"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        structural_hints = ["split_range_into_start_end", "requires_range_semantics_review"]
        interpretation_hints = ["mixed_content_high_risk"]
        normalization_notes = "Range-like values need structural follow-up before safe numeric modeling."
    elif top_type == "numeric_with_unit":
        logical_type = "numeric_measure"
        storage_type = "decimal"
        transform_actions = ["trim_whitespace", "extract_numeric_component", "strip_unit_suffix", "cast_to_decimal"]
        structural_hints = ["requires_unit_normalization_review"]
        normalization_notes = "Fallback typing keeps unit-bearing values numeric while preserving later unit-normalization review."
    elif top_type == "percent":
        logical_type = "numeric_measure"
        storage_type = "decimal"
        transform_actions = ["trim_whitespace", "strip_numeric_formatting", "standardize_percent_scale", "cast_to_decimal"]
        normalization_notes = "Fallback typing standardizes percentage-like values into decimal numeric storage."
    elif free_text_name_hint and (
        textual_value_hint
        or top_type in {"text", "mixed"}
        or unique_ratio >= 0.35
        or unique_count >= 20
    ):
        logical_type = "free_text"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["free_text_high_cardinality"]
        if numeric_pct >= 80.0:
            interpretation_hints.append("numeric_parse_is_misleading")
        normalization_notes = "Column naming and observed samples indicate open-response text, so fallback typing preserves string content even when parser signals are noisy."
    elif repeat_survey_semantics:
        semantic_suggests_ordinal = (
            _ordinal_like(top_levels, encoding_type)
            or "likert" in semantic_blob
            or "ordinal" in semantic_blob
            or "rating" in semantic_blob
            or "familiarity" in semantic_blob
            or "frequency" in semantic_blob
            or "agreement" in semantic_blob
        )
        logical_type = "ordinal_category" if semantic_suggests_ordinal else "nominal_category"
        storage_type = "string"
        if _looks_numeric_like(top_levels):
            transform_actions = ["trim_whitespace", "strip_numeric_formatting", "cast_to_string"]
            interpretation_hints = ["numeric_parse_is_misleading"]
        else:
            transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        normalization_notes = "Family semantics indicate a repeated survey-response block, so numeric-looking codes are preserved as categories rather than promoted to measures."
    elif top_type == "categorical_multi":
        logical_type = "categorical_code" if code_semantics else "nominal_category"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        structural_hints = ["split_multiselect_tokens", "requires_multiselect_modeling_decision"]
        if code_semantics:
            structural_hints.append("requires_codebook_or_label_mapping_review")
            interpretation_hints.append("code_not_quantity")
        normalization_notes = "Fallback typing keeps multi-value cells as string-backed categories pending structural token-splitting."
    elif code_semantics:
        logical_type = "categorical_code"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        structural_hints = ["requires_codebook_or_label_mapping_review"]
        interpretation_hints = ["code_not_quantity"]
        if _looks_numeric_like(top_levels):
            transform_actions.insert(1, "strip_numeric_formatting")
            interpretation_hints.append("numeric_parse_is_misleading")
        normalization_notes = "Structured semantic evidence indicates code semantics, so numeric-looking values are preserved as strings."
    elif _boolean_like(top_levels):
        logical_type = "boolean_flag"
        storage_type = "boolean"
        transform_actions = ["trim_whitespace", "normalize_boolean_tokens"]
        normalization_notes = "Fallback typing treats the column as boolean-like based on observed value vocabulary."
    elif top_type in {"numeric"} or numeric_pct >= 80.0 or role in {"measure", "measure_numeric"}:
        logical_type = "numeric_measure"
        storage_type = "integer" if _looks_integer_like(top_levels + [str(value) for value in samples]) else "decimal"
        transform_actions = ["trim_whitespace", "strip_numeric_formatting", "cast_to_integer" if storage_type == "integer" else "cast_to_decimal"]
        normalization_notes = "Fallback typing keeps numeric parse evidence as measure semantics."
    elif _ordinal_like(top_levels, encoding_type):
        logical_type = "ordinal_category"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        normalization_notes = "Fallback typing uses ordinal category semantics from value vocabulary and role evidence."
    elif unique_ratio >= 0.5 or unique_count >= 50 or top_type == "text":
        logical_type = "free_text"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["free_text_high_cardinality"]
        normalization_notes = "Fallback typing preserves high-cardinality text as free-form string content."
    elif top_type in {"categorical", "mixed", "text"} or role in {"coded_categorical", "invariant_attr", "measure_item"}:
        logical_type = "nominal_category"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "normalize_category_tokens"]
        normalization_notes = "Fallback typing keeps low-cardinality values as string-backed nominal categories."
    else:
        logical_type = "mixed_or_ambiguous"
        storage_type = "string"
        transform_actions = ["trim_whitespace", "cast_to_string"]
        interpretation_hints = ["mixed_content_high_risk"]
        normalization_notes = "Fallback typing remains conservative because the baseline evidence is mixed."

    if explicit_missing_tokens and "normalize_missing_tokens" not in transform_actions:
        transform_actions.append("normalize_missing_tokens")

    if structure.get("source_family_id") and structure.get("canonical_modeling_status") != "child_repeat_member":
        structural_hints.append("requires_child_table_review")

    return {
        "recommended_logical_type": logical_type,
        "recommended_storage_type": storage_type,
        "transform_actions": _dedupe_preserve_order(action for action in transform_actions if action in TYPE_TRANSFORM_ACTIONS),
        "structural_transform_hints": _dedupe_preserve_order(hint for hint in structural_hints if hint in TYPE_STRUCTURAL_HINTS),
        "interpretation_hints": _dedupe_preserve_order(hint for hint in interpretation_hints if hint in TYPE_INTERPRETATION_HINTS),
        "normalization_notes": normalization_notes,
        "type_decision_source": "a2_fallback",
        "type_review_required": (
            logical_type == "mixed_or_ambiguous"
            or _safe_float(a2_row.get("confidence"), 0.0) < 0.75
            or bool((a2_row.get("high_missingness") or False) and logical_type in {"free_text", "mixed_or_ambiguous"})
        ),
        "confidence": min(0.7, _safe_float(a2_row.get("confidence"), 0.6)),
    }


def _fallback_missingness_decision(
    column: str,
    a2_row: Optional[Dict[str, Any]],
    a16_context: Dict[str, Any],
    family_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not a2_row:
        return {
            "missingness_disposition": "mixed_missingness_risk",
            "missingness_handling": "review_before_drop",
            "skip_logic_protected": False,
            "normalization_notes": "",
            "missingness_decision_source": "unresolved_no_a2_evidence",
            "missingness_review_required": True,
            "confidence": 0.5,
        }

    missing_pct = _safe_float(a2_row.get("missing_pct"), 0.0)
    token_missing = bool(a2_row.get("missing_tokens_observed"))
    a16_row = a16_context.get(column) or {}
    structural_validity = str(a16_row.get("structural_validity") or "not_applicable")
    family_context = family_context or {}
    family_structural_validity = str(family_context.get("structural_validity") or "not_applicable")
    family_trigger_columns = [
        str(trigger).strip()
        for trigger in (family_context.get("trigger_columns") or [])
        if str(trigger or "").strip()
    ]
    family_reviewed_columns = [
        str(value).strip()
        for value in (family_context.get("reviewed_structural_columns") or [])
        if str(value or "").strip()
    ]

    if structural_validity == "confirmed_structural":
        return {
            "missingness_disposition": "structurally_valid_missingness",
            "missingness_handling": "protect_from_null_penalty",
            "skip_logic_protected": True,
            "normalization_notes": "A16 surfaced direct skip-logic evidence for this field in sampled affected columns.",
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": False,
            "confidence": 0.78,
        }

    if structural_validity == "plausible_structural":
        return {
            "missingness_disposition": "partially_structural_missingness",
            "missingness_handling": "retain_with_caution",
            "skip_logic_protected": False,
            "normalization_notes": "A16 surfaced sample-based gating evidence, but fallback handling remains conservative without reviewed adjudication.",
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": True,
            "confidence": 0.68,
        }

    if family_structural_validity == "confirmed_structural" and missing_pct >= 20.0:
        evidence_note = "Sibling columns in the same family show reviewed or sampled structural missingness evidence."
        if family_trigger_columns:
            evidence_note = f"{evidence_note} Trigger context: {', '.join(family_trigger_columns[:3])}."
        if family_reviewed_columns:
            evidence_note = f"{evidence_note} Reviewed sibling evidence was present for {', '.join(family_reviewed_columns[:3])}."
        return {
            "missingness_disposition": "structurally_valid_missingness",
            "missingness_handling": "protect_from_null_penalty",
            "skip_logic_protected": True,
            "normalization_notes": evidence_note,
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": False,
            "confidence": 0.76,
        }

    if family_structural_validity == "plausible_structural" and missing_pct >= 20.0:
        evidence_note = "Sibling columns in the same family show plausible structural missingness evidence, so fallback handling propagates that family-level caution."
        if family_trigger_columns:
            evidence_note = f"{evidence_note} Trigger context: {', '.join(family_trigger_columns[:3])}."
        return {
            "missingness_disposition": "partially_structural_missingness",
            "missingness_handling": "retain_with_caution",
            "skip_logic_protected": False,
            "normalization_notes": evidence_note,
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": True,
            "confidence": 0.7,
        }

    if missing_pct < 5.0 and not token_missing:
        return {
            "missingness_disposition": "no_material_missingness",
            "missingness_handling": "no_action_needed",
            "skip_logic_protected": False,
            "normalization_notes": "Missingness is low and no explicit token-based cleanup signal is present.",
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": False,
            "confidence": 0.72,
        }

    if token_missing and missing_pct < 40.0:
        return {
            "missingness_disposition": "token_missingness_present",
            "missingness_handling": "retain_with_caution",
            "skip_logic_protected": False,
            "normalization_notes": "A2 shows explicit missing-like tokens that should be standardized cautiously.",
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": False,
            "confidence": 0.69,
        }

    if missing_pct >= 80.0:
        return {
            "missingness_disposition": "unexplained_high_missingness",
            "missingness_handling": "candidate_drop_review",
            "skip_logic_protected": False,
            "normalization_notes": "Missingness is very high and not structurally explained by the sampled A16 evidence.",
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": True,
            "confidence": 0.62,
        }

    if missing_pct >= 50.0:
        return {
            "missingness_disposition": "mixed_missingness_risk",
            "missingness_handling": "review_before_drop",
            "skip_logic_protected": False,
            "normalization_notes": "Missingness is materially high without direct structural proof.",
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": True,
            "confidence": 0.64,
        }

    if token_missing or missing_pct >= 20.0:
        return {
            "missingness_disposition": "mixed_missingness_risk",
            "missingness_handling": "retain_with_caution",
            "skip_logic_protected": False,
            "normalization_notes": "Missingness is present but not strongly structural in fallback evidence.",
            "missingness_decision_source": "a2_a16_fallback",
            "missingness_review_required": False,
            "confidence": 0.67,
        }

    return {
        "missingness_disposition": "no_material_missingness",
        "missingness_handling": "no_action_needed",
        "skip_logic_protected": False,
        "normalization_notes": "",
        "missingness_decision_source": "a2_a16_fallback",
        "missingness_review_required": False,
        "confidence": 0.7,
    }


def _add_review_flag(flags: List[Dict[str, Any]], item: str, issue: str, why: str) -> None:
    candidate = {
        "item": str(item or "").strip(),
        "issue": str(issue or "").strip(),
        "why": str(why or "").strip(),
    }
    if not candidate["item"] or not candidate["issue"] or not candidate["why"]:
        return
    if candidate not in flags:
        flags.append(candidate)


def _review_reason_summary(rows: List[Dict[str, Any]]) -> str:
    reasons: List[str] = []
    if any(row.get("canonical_modeling_status") == "unresolved" for row in rows):
        reasons.append("unresolved placement")
    if any(row.get("type_decision_source") == "family_default" for row in rows):
        reasons.append("family-default typing")
    elif any(row.get("type_decision_source") == "a17_baseline" for row in rows):
        reasons.append("deterministic baseline typing")
    if any(
        row.get("missingness_decision_source") == "family_default"
        and row.get("missingness_disposition") in {"mixed_missingness_risk", "unexplained_high_missingness", "partially_structural_missingness", "structurally_valid_missingness"}
        for row in rows
    ):
        reasons.append("family-default missingness adjudication")
    elif any(
        row.get("missingness_decision_source") == "a17_baseline"
        and row.get("missingness_disposition") in {"mixed_missingness_risk", "unexplained_high_missingness", "partially_structural_missingness", "structurally_valid_missingness"}
        for row in rows
    ):
        reasons.append("deterministic baseline missingness adjudication")
    if any(row.get("drift_detected") is True for row in rows):
        reasons.append("drift follow-up")
    if not reasons:
        reasons.append("manual adjudication")
    if len(reasons) == 1:
        return reasons[0]
    if len(reasons) == 2:
        return f"{reasons[0]} and {reasons[1]}"
    return ", ".join(reasons[:-1]) + f", and {reasons[-1]}"


def _append_review_summary_flags(flags: List[Dict[str, Any]], column_contracts: List[Dict[str, Any]]) -> None:
    review_rows = [row for row in column_contracts if row.get("needs_human_review")]
    if review_rows:
        family_hotspots = len({str(row.get("source_family_id") or "").strip() for row in review_rows if str(row.get("source_family_id") or "").strip()})
        singleton_hotspots = sum(1 for row in review_rows if not str(row.get("source_family_id") or "").strip())
        _add_review_flag(
            flags,
            "contract_summary",
            "review_required_columns_present",
            f"{len(review_rows)} column contracts require human review across {family_hotspots + singleton_hotspots} hotspots.",
        )

    baseline_count = sum(1 for row in column_contracts if row.get("type_decision_source") == "a17_baseline")
    if column_contracts and (baseline_count / len(column_contracts)) >= 0.5:
        _add_review_flag(
            flags,
            "contract_summary",
            "high_deterministic_baseline_share",
            f"{baseline_count} of {len(column_contracts)} columns relied on A17 baseline typing; downstream automation should treat unresolved semantics as provisional until reviewed coverage improves.",
        )

    family_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    standalone_rows: List[Dict[str, Any]] = []
    for row in review_rows:
        family_id = str(row.get("source_family_id") or "").strip()
        if family_id:
            family_groups[family_id].append(row)
        else:
            standalone_rows.append(row)

    for family_id, rows in sorted(family_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(rows) == 1:
            standalone_rows.extend(rows)
            continue
        _add_review_flag(
            flags,
            family_id,
            "family_columns_need_review",
            f"{len(rows)} columns in family '{family_id}' require review due to {_review_reason_summary(rows)}.",
        )

    for row in sorted(standalone_rows, key=lambda item: str(item.get("column") or ""))[:12]:
        column = str(row.get("column") or "").strip()
        if not column:
            continue
        _add_review_flag(
            flags,
            column,
            "column_needs_review",
            f"Column '{column}' requires review due to {_review_reason_summary([row])}.",
        )


def _cleanup_structural_hints(hints: Iterable[Any], canonical_modeling_status: str) -> List[str]:
    cleaned = _dedupe_preserve_order(hint for hint in hints if str(hint or "").strip() in TYPE_STRUCTURAL_HINTS)
    if canonical_modeling_status == "child_repeat_member":
        cleaned = [
            hint
            for hint in cleaned
            if hint not in POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS
        ]
    return cleaned


def _build_conflict_flags(
    column: str,
    row: Dict[str, Any],
    light_contract_maps: Dict[str, Any],
    family_results_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []

    if column in set(light_contract_maps.get("primary_keys") or []) and row["canonical_assignment_role"] in {"exclude_from_outputs", "unresolved"}:
        flags.append(
            {
                "item": column,
                "issue": "light_contract_primary_key_unplaced",
                "why": "The finalized light contract marks this column as a primary-grain key, but the canonical structure does not place it in a resolved table role.",
            }
        )

    expected_reference_table = str((light_contract_maps.get("reference_table_by_key") or {}).get(column) or "").strip()
    if expected_reference_table and row["canonical_assignment_role"] in {"exclude_from_outputs", "unresolved", "base_key", "base_attribute"}:
        flags.append(
            {
                "item": column,
                "issue": "light_contract_reference_conflict",
                "why": f"The finalized light contract associates this key with reference table '{expected_reference_table}', but the canonical structure did not preserve that placement.",
            }
        )

    family_id = row.get("source_family_id") or ""
    family_result = family_results_by_id.get(family_id) or {}
    family_role = str(family_result.get("recommended_family_role") or "").strip()
    if family_role == "answer_key_or_reference_block" and row["canonical_modeling_status"] == "child_repeat_member":
        flags.append(
            {
                "item": column,
                "issue": "family_semantics_vs_layout_conflict",
                "why": "Reviewed family semantics suggest a reference-style block, but canonical layout still models this column as a child-repeat member.",
            }
        )

    if row["recommended_logical_type"] == "identifier" and row["canonical_assignment_role"] == "melt_member":
        flags.append(
            {
                "item": column,
                "issue": "identifier_vs_repeat_member_conflict",
                "why": "Identifier semantics conflict with a melt-member placement and should be reviewed.",
            }
        )

    if row["recommended_logical_type"] == "numeric_measure" and row["canonical_assignment_role"] in {"base_key", "reference_key", "repeat_parent_key", "repeat_index"}:
        flags.append(
            {
                "item": column,
                "issue": "measure_vs_key_conflict",
                "why": "Numeric-measure semantics conflict with a key-like structural assignment.",
            }
        )

    return flags


def _resolve_contract_confidence(
    a2_row: Optional[Dict[str, Any]],
    type_confidence: float,
    missingness_confidence: float,
    needs_human_review: bool,
    conflict_count: int,
    unresolved: bool,
    drift_detected: Optional[bool],
) -> float:
    base_candidates = [type_confidence, missingness_confidence]
    if a2_row:
        base_candidates.append(min(0.7, _safe_float(a2_row.get("confidence"), 0.0)))
    base = max(base_candidates) if any(base_candidates) else 0.5

    if drift_detected and not a2_row is None:
        base = min(base, 0.78)
    if needs_human_review:
        base = min(base, 0.75)
    if conflict_count:
        base = min(base, 0.6)
    if unresolved:
        base = min(base, 0.5)

    return round(max(0.0, min(1.0, base)), 6)


def _validate_canonical_column_contract_output(
    payload: Dict[str, Any],
    expected_source_columns: List[str],
    *,
    global_token_missing_placeholders_detected: Optional[bool] = None,
) -> List[str]:
    errors: List[str] = []
    required_top_level = [
        "worker",
        "summary",
        "column_contracts",
        "global_value_rules",
        "review_flags",
        "assumptions",
    ]
    for key in required_top_level:
        if key not in payload:
            errors.append(f"Missing required top-level key: {key}")

    if payload.get("worker") != "canonical_column_contract_builder":
        errors.append("worker must be 'canonical_column_contract_builder'")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        for key in [
            "overview",
            "total_source_columns",
            "included_column_count",
            "excluded_column_count",
            "unresolved_column_count",
            "reviewed_type_count",
            "fallback_type_count",
            "key_contract_principles",
        ]:
            if key not in summary:
                errors.append(f"summary.{key} is required")
        if not isinstance(summary.get("overview"), str) or not summary.get("overview", "").strip():
            errors.append("summary.overview must be a non-empty string")
        if not isinstance(summary.get("key_contract_principles"), list):
            errors.append("summary.key_contract_principles must be an array")
        for key in ["reviewed_override_count", "family_default_count", "deterministic_baseline_count"]:
            if key not in summary:
                errors.append(f"summary.{key} is required")
            elif not isinstance(summary.get(key), int):
                errors.append(f"summary.{key} must be an integer")

    rows = payload.get("column_contracts")
    if not isinstance(rows, list) or not rows:
        errors.append("column_contracts must be a non-empty array")
        return errors

    seen_columns: Set[str] = set()
    expected_set = set(expected_source_columns)
    seen_expected: Set[str] = set()

    for idx, row in enumerate(rows):
        path = f"column_contracts[{idx}]"
        if not isinstance(row, dict):
            errors.append(f"{path} must be an object")
            continue

        required_row_fields = [
            "column",
            "canonical_modeling_status",
            "canonical_table_name",
            "canonical_assignment_role",
            "source_family_id",
            "a9_primary_role",
            "recommended_logical_type",
            "recommended_storage_type",
            "transform_actions",
            "structural_transform_hints",
            "interpretation_hints",
            "missingness_disposition",
            "missingness_handling",
            "skip_logic_protected",
            "confidence",
            "needs_human_review",
            "type_decision_source",
            "structure_decision_source",
            "missingness_decision_source",
            "semantic_decision_source",
            "applied_sources",
        ]
        for key in required_row_fields:
            if key not in row:
                errors.append(f"{path}.{key} is required")

        column = str(row.get("column") or "").strip()
        if not column:
            errors.append(f"{path}.column must be a non-empty string")
        elif column in seen_columns:
            errors.append(f"Duplicate column_contracts entry for column: {column}")
        else:
            seen_columns.add(column)
            if column in expected_set:
                seen_expected.add(column)

        if row.get("canonical_modeling_status") not in CANONICAL_MODELING_STATUSES:
            errors.append(f"{path}.canonical_modeling_status is invalid")
        if row.get("canonical_assignment_role") not in CANONICAL_ASSIGNMENT_ROLES:
            errors.append(f"{path}.canonical_assignment_role is invalid")
        if row.get("recommended_logical_type") not in TYPE_LOGICAL_TYPES:
            errors.append(f"{path}.recommended_logical_type is invalid")
        if row.get("recommended_storage_type") not in TYPE_STORAGE_TYPES:
            errors.append(f"{path}.recommended_storage_type is invalid")
        if row.get("missingness_disposition") not in MISSINGNESS_DISPOSITIONS:
            errors.append(f"{path}.missingness_disposition is invalid")
        if row.get("missingness_handling") not in MISSINGNESS_HANDLING:
            errors.append(f"{path}.missingness_handling is invalid")
        if row.get("type_decision_source") not in CANONICAL_TYPE_DECISION_SOURCES:
            errors.append(f"{path}.type_decision_source is invalid")
        if row.get("structure_decision_source") not in CANONICAL_STRUCTURE_DECISION_SOURCES:
            errors.append(f"{path}.structure_decision_source is invalid")
        if row.get("missingness_decision_source") not in CANONICAL_MISSINGNESS_DECISION_SOURCES:
            errors.append(f"{path}.missingness_decision_source is invalid")
        if row.get("semantic_decision_source") not in CANONICAL_SEMANTIC_DECISION_SOURCES:
            errors.append(f"{path}.semantic_decision_source is invalid")

        if row.get("canonical_assignment_role") in {"exclude_from_outputs", "unresolved"}:
            if str(row.get("canonical_table_name") or "").strip():
                errors.append(f"{path}.canonical_table_name must be blank for excluded or unresolved assignments")
        elif not str(row.get("canonical_table_name") or "").strip():
            errors.append(f"{path}.canonical_table_name must be non-blank for resolved assignments")

        if not isinstance(row.get("transform_actions"), list):
            errors.append(f"{path}.transform_actions must be an array")
        else:
            for item in row.get("transform_actions") or []:
                if item not in TYPE_TRANSFORM_ACTIONS:
                    errors.append(f"{path}.transform_actions contains invalid value: {item}")

        if not isinstance(row.get("structural_transform_hints"), list):
            errors.append(f"{path}.structural_transform_hints must be an array")
        else:
            for item in row.get("structural_transform_hints") or []:
                if item not in TYPE_STRUCTURAL_HINTS:
                    errors.append(f"{path}.structural_transform_hints contains invalid value: {item}")
                elif (
                    row.get("canonical_modeling_status") == "child_repeat_member"
                    and item in POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS
                ):
                    errors.append(
                        f"{path}.structural_transform_hints must not contain {item} when canonical_modeling_status is child_repeat_member"
                    )

        if not isinstance(row.get("interpretation_hints"), list):
            errors.append(f"{path}.interpretation_hints must be an array")
        else:
            for item in row.get("interpretation_hints") or []:
                if item not in TYPE_INTERPRETATION_HINTS:
                    errors.append(f"{path}.interpretation_hints contains invalid value: {item}")

        if not isinstance(row.get("applied_sources"), list):
            errors.append(f"{path}.applied_sources must be an array")
        else:
            for item in row.get("applied_sources") or []:
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"{path}.applied_sources contains an invalid value")

        if not isinstance(row.get("skip_logic_protected"), bool):
            errors.append(f"{path}.skip_logic_protected must be boolean")
        for invariant_error in _find_canonical_row_invariant_errors(
            row,
            path,
            global_token_missing_placeholders_detected=global_token_missing_placeholders_detected,
        ):
            errors.append(invariant_error)
        if not isinstance(row.get("needs_human_review"), bool):
            errors.append(f"{path}.needs_human_review must be boolean")
        if not isinstance(row.get("confidence"), (int, float)) or isinstance(row.get("confidence"), bool):
            errors.append(f"{path}.confidence must be numeric")
        elif not (0.0 <= float(row["confidence"]) <= 1.0):
            errors.append(f"{path}.confidence must be between 0 and 1")

        for key in ["semantic_meaning", "codebook_note", "normalization_notes", "a9_primary_role", "source_family_id", "canonical_table_name"]:
            if key in row and row[key] is not None and not isinstance(row[key], str):
                errors.append(f"{path}.{key} must be a string or null")
        if "quality_score" in row and row["quality_score"] is not None:
            if not isinstance(row["quality_score"], (int, float)) or isinstance(row["quality_score"], bool):
                errors.append(f"{path}.quality_score must be numeric or null")
        if "drift_detected" in row and row["drift_detected"] is not None and not isinstance(row["drift_detected"], bool):
            errors.append(f"{path}.drift_detected must be boolean or null")

    missing_expected = sorted(expected_set - seen_expected)
    if missing_expected:
        errors.append(f"column_contracts is missing A2 source columns: {', '.join(missing_expected)}")

    for key in ["global_value_rules", "review_flags", "assumptions"]:
        if not isinstance(payload.get(key), list):
            errors.append(f"{key} must be an array")

    return errors


def _synthesize_canonical_column_contract(
    run_id: str,
    light_contract_decisions: Dict[str, Any],
    semantic_context_json: Dict[str, Any],
    type_transform_worker_json: Dict[str, Any],
    missingness_worker_json: Dict[str, Any],
    family_worker_json: Any,
    table_layout_worker_json: Dict[str, Any],
    scale_mapping_json: Dict[str, Any],
    support: Dict[str, Any],
) -> Dict[str, Any]:
    a2_by_col = support["a2_by_col"]
    a2_order = support["a2_order"]
    a17_by_col = support["a17_by_col"]
    family_by_column = support["family_by_column"]

    light_contract_maps = _normalize_light_contract_maps(light_contract_decisions)
    type_by_col, global_rules = _normalize_type_output(type_transform_worker_json)
    missingness_context = _normalize_missingness_output(missingness_worker_json)
    missingness_by_col = missingness_context["column_decisions"]
    missingness_global_contract = missingness_context["global_contract"]
    global_token_missing_placeholders_detected = missingness_global_contract.get("token_missing_placeholders_detected")
    family_results_by_id = _extract_family_results(family_worker_json)
    family_defaults_by_id = _normalize_family_member_defaults(family_worker_json)
    assignments_by_col, tables_by_name = _normalize_table_layout_output(table_layout_worker_json)
    column_family_map = dict(family_by_column)
    for col, assignment in assignments_by_col.items():
        family_id = str(assignment.get("source_family_id") or "").strip()
        if family_id:
            column_family_map[col] = family_id

    known_family_ids = set(column_family_map.values()) | set(family_results_by_id.keys()) | set((light_contract_maps.get("family_by_id") or {}).keys())
    for row in assignments_by_col.values():
        family_id = str(row.get("source_family_id") or "").strip()
        if family_id:
            known_family_ids.add(family_id)

    semantic_index = _normalize_semantic_context(
        semantic_context_json,
        known_columns=set(a2_by_col.keys()),
        known_family_ids=known_family_ids,
    )
    scale_mapping_index = _normalize_scale_mapping_contract(
        scale_mapping_json,
        known_columns=set(a2_by_col.keys()),
        known_family_ids=known_family_ids,
    )

    extra_columns = set(type_by_col.keys()) | set(missingness_by_col.keys()) | set(assignments_by_col.keys()) | set(semantic_index["extra_columns"])
    extra_columns.update(light_contract_maps.get("primary_keys") or [])
    extra_columns.update((light_contract_maps.get("reference_table_by_key") or {}).keys())
    extra_columns = {col for col in extra_columns if col and col not in a2_by_col}

    top_level_review_flags: List[Dict[str, Any]] = []
    for extra in sorted(extra_columns):
        _add_review_flag(
            top_level_review_flags,
            extra,
            "reviewed_column_not_in_a2",
            "A reviewed or structural input referenced this column, but it does not exist in the baseline A2 column dictionary for the run.",
        )
    for extra_target in sorted(scale_mapping_index.get("extra_targets") or set()):
        _add_review_flag(
            top_level_review_flags,
            extra_target,
            "scale_mapping_target_not_in_a2_scope",
            "scale_mapping_json referenced a family or column target that does not resolve against the run's reviewed family inventory or A2 source columns.",
        )

    column_contracts: List[Dict[str, Any]] = []
    all_columns = list(a2_order) + sorted(extra_columns)

    for column in all_columns:
        a2_row = a2_by_col.get(column)
        baseline_row = a17_by_col.get(column) or {}

        structure = _resolve_structure_for_column(
            column=column,
            assignments_by_col=assignments_by_col,
            tables_by_name=tables_by_name,
            light_contract_maps=light_contract_maps,
            family_by_column=column_family_map,
        )
        family_id = str(structure.get("source_family_id") or "")
        family_result = family_results_by_id.get(family_id) or {}
        family_defaults = family_defaults_by_id.get(family_id) or {}
        semantic = _resolve_semantic_enrichment(
            column=column,
            source_family_id=family_id,
            semantic_index=semantic_index,
            family_results_by_id=family_results_by_id,
            a13_row={},
        )

        type_decision = _baseline_row_to_type_decision(baseline_row)
        type_decision, family_type_used = _apply_family_type_defaults(type_decision, family_defaults)
        reviewed_type = type_by_col.get(column)
        if reviewed_type:
            type_decision = {
                "recommended_logical_type": str(reviewed_type.get("recommended_logical_type") or "mixed_or_ambiguous"),
                "recommended_storage_type": str(reviewed_type.get("recommended_storage_type") or "string"),
                "transform_actions": _dedupe_preserve_order(
                    item for item in (reviewed_type.get("transform_actions") or []) if str(item or "").strip() in TYPE_TRANSFORM_ACTIONS
                ),
                "structural_transform_hints": _dedupe_preserve_order(
                    item for item in (reviewed_type.get("structural_transform_hints") or []) if str(item or "").strip() in TYPE_STRUCTURAL_HINTS
                ),
                "interpretation_hints": _dedupe_preserve_order(
                    item for item in (reviewed_type.get("interpretation_hints") or []) if str(item or "").strip() in TYPE_INTERPRETATION_HINTS
                ),
                "normalization_notes": str(reviewed_type.get("normalization_notes") or "").strip(),
                "type_decision_source": "reviewed_type_worker",
                "type_review_required": bool(reviewed_type.get("needs_human_review", False)),
                "confidence": _safe_float(reviewed_type.get("confidence"), 0.0),
                "applied_sources": ["type_transform_worker"],
            }
        scale_mapping = _resolve_scale_mapping_for_column(
            column=column,
            source_family_id=family_id,
            scale_mapping_index=scale_mapping_index,
        )
        type_decision, semantic, scale_mapping_applied = _apply_scale_mapping_to_type_and_semantics(
            type_decision=type_decision,
            semantic=semantic,
            mapping=scale_mapping,
        )

        baseline_missingness_decision = _baseline_row_to_missingness_decision(baseline_row)
        missingness_decision = dict(baseline_missingness_decision)
        missingness_decision, family_missingness_used = _apply_family_missingness_defaults(
            missingness_decision,
            family_defaults,
            global_token_missing_placeholders_detected=global_token_missing_placeholders_detected,
        )
        reviewed_missingness = missingness_by_col.get(column)
        if reviewed_missingness:
            missingness_decision = {
                "missingness_disposition": str(reviewed_missingness.get("missingness_disposition") or "mixed_missingness_risk"),
                "missingness_handling": str(reviewed_missingness.get("recommended_handling") or "retain_with_caution"),
                "skip_logic_protected": bool(reviewed_missingness.get("skip_logic_protected", False)),
                "normalization_notes": str(reviewed_missingness.get("normalization_notes") or "").strip(),
                "missingness_decision_source": "reviewed_missingness_worker",
                "missingness_review_required": bool(reviewed_missingness.get("needs_human_review", False)),
                "confidence": _safe_float(reviewed_missingness.get("confidence"), 0.0),
                "applied_sources": ["missingness_worker"],
            }

        interpretation_hints = list(type_decision["interpretation_hints"])
        interpretation_hints = _sync_skip_logic_interpretation_hints(
            interpretation_hints,
            bool(missingness_decision["skip_logic_protected"]),
        )

        structural_transform_hints = _cleanup_structural_hints(
            type_decision["structural_transform_hints"],
            str(structure.get("canonical_modeling_status") or "unresolved"),
        )

        quality_score = baseline_row.get("quality_score") if isinstance(baseline_row, dict) else None
        drift_detected = baseline_row.get("drift_detected") if isinstance(baseline_row, dict) else None

        applied_sources: List[str] = []
        structure_source = str(structure.get("structure_decision_source") or "unresolved")
        if structure_source == "table_layout_worker":
            applied_sources.append("table_layout_worker")
        elif structure_source == "light_contract_fallback":
            applied_sources.append("light_contract_decisions")
        if baseline_row:
            applied_sources.append("A17")
        if semantic.get("semantic_decision_source") == "semantic_context_worker":
            applied_sources.append("semantic_context_worker")
        elif semantic.get("semantic_decision_source") == "family_worker":
            applied_sources.append("family_worker.family_result")
        elif semantic.get("semantic_decision_source") == "scale_mapping_resolver":
            applied_sources.append("scale_mapping_resolver")
        if family_type_used or family_missingness_used:
            applied_sources.append("family_worker.member_defaults")
        applied_sources.extend(type_decision.get("applied_sources") or [])
        applied_sources.extend(missingness_decision.get("applied_sources") or [])
        if scale_mapping_applied:
            applied_sources.append("scale_mapping_resolver")

        row = {
            "column": column,
            "canonical_modeling_status": str(structure.get("canonical_modeling_status") or "unresolved"),
            "canonical_table_name": str(structure.get("canonical_table_name") or ""),
            "canonical_assignment_role": str(structure.get("canonical_assignment_role") or "unresolved"),
            "source_family_id": str(structure.get("source_family_id") or ""),
            "a9_primary_role": str((baseline_row or {}).get("a9_primary_role") or ""),
            "recommended_logical_type": str(type_decision["recommended_logical_type"]),
            "recommended_storage_type": str(type_decision["recommended_storage_type"]),
            "transform_actions": list(type_decision["transform_actions"]),
            "structural_transform_hints": structural_transform_hints,
            "interpretation_hints": interpretation_hints,
            "missingness_disposition": str(missingness_decision["missingness_disposition"]),
            "missingness_handling": str(missingness_decision["missingness_handling"]),
            "skip_logic_protected": bool(missingness_decision["skip_logic_protected"]),
            "semantic_meaning": str(semantic.get("semantic_meaning") or ""),
            "codebook_note": str(semantic.get("codebook_note") or ""),
            "normalization_notes": _merge_notes(type_decision.get("normalization_notes"), missingness_decision.get("normalization_notes")),
            "quality_score": round(_safe_float(quality_score), 6) if isinstance(quality_score, (int, float)) else None,
            "drift_detected": bool(drift_detected) if isinstance(drift_detected, bool) else None,
            "type_decision_source": str(type_decision["type_decision_source"]),
            "structure_decision_source": str(structure.get("structure_decision_source") or "unresolved"),
            "missingness_decision_source": str(missingness_decision["missingness_decision_source"]),
            "semantic_decision_source": str(semantic.get("semantic_decision_source") or "unknown"),
            "applied_sources": _dedupe_preserve_order(applied_sources),
            "confidence": 0.0,
            "needs_human_review": False,
        }

        normalized_missingness, reconciliation_notes, invariant_errors = _normalize_canonical_missingness_decision(
            {
                "canonical_modeling_status": row["canonical_modeling_status"],
                "structural_transform_hints": row["structural_transform_hints"],
                "missingness_disposition": row["missingness_disposition"],
                "missingness_handling": row["missingness_handling"],
                "missingness_decision_source": row["missingness_decision_source"],
                "skip_logic_protected": row["skip_logic_protected"],
            },
            base_decision=baseline_missingness_decision,
            global_token_missing_placeholders_detected=global_token_missing_placeholders_detected,
        )
        row["missingness_disposition"] = str(normalized_missingness["missingness_disposition"])
        row["missingness_handling"] = str(normalized_missingness["missingness_handling"])
        row["skip_logic_protected"] = bool(normalized_missingness["skip_logic_protected"])
        row["interpretation_hints"] = _sync_skip_logic_interpretation_hints(
            list(row["interpretation_hints"]),
            row["skip_logic_protected"],
        )
        row["normalization_notes"] = _merge_notes(
            row.get("normalization_notes"),
            "Deterministic invariant reconciliation applied."
            if reconciliation_notes
            else "",
        )
        if invariant_errors:
            raise ValueError(
                f"Canonical column contract row for {column} is not safely reconcilable: {invariant_errors[0]}"
            )

        row_conflicts = _build_conflict_flags(
            column=column,
            row=row,
            light_contract_maps=light_contract_maps,
            family_results_by_id=family_results_by_id,
        )
        for flag in row_conflicts:
            _add_review_flag(
                top_level_review_flags,
                str(flag.get("item") or ""),
                str(flag.get("issue") or ""),
                str(flag.get("why") or ""),
            )

        needs_review = bool(
            type_decision["type_review_required"]
            or missingness_decision["missingness_review_required"]
            or row["canonical_modeling_status"] == "unresolved"
            or row_conflicts
            or (
                row["drift_detected"] is True
                and row["type_decision_source"] != "reviewed_type_worker"
                and row["recommended_logical_type"] in {"date", "datetime", "mixed_or_ambiguous"}
            )
        )

        row["needs_human_review"] = needs_review
        row["confidence"] = _resolve_contract_confidence(
            a2_row=a2_row,
            type_confidence=_safe_float(type_decision.get("confidence"), 0.0),
            missingness_confidence=_safe_float(missingness_decision.get("confidence"), 0.0),
            needs_human_review=needs_review,
            conflict_count=len(row_conflicts),
            unresolved=row["canonical_modeling_status"] == "unresolved",
            drift_detected=row["drift_detected"],
        )

        column_contracts.append(row)

    _append_review_summary_flags(top_level_review_flags, column_contracts)

    summary = {
        "overview": (
            f"Synthesized a deterministic canonical column contract for {len(column_contracts)} columns by "
            f"merging structure decisions, semantic context, reviewed overrides, scale mappings, family defaults, and the A17 baseline layer."
        ),
        "total_source_columns": len(column_contracts),
        "included_column_count": sum(1 for row in column_contracts if row["canonical_modeling_status"] not in {"excluded_from_outputs", "unresolved"}),
        "excluded_column_count": sum(1 for row in column_contracts if row["canonical_modeling_status"] == "excluded_from_outputs"),
        "unresolved_column_count": sum(1 for row in column_contracts if row["canonical_modeling_status"] == "unresolved"),
        "reviewed_override_count": sum(
            1
            for row in column_contracts
            if row["type_decision_source"] == "reviewed_type_worker" or row["missingness_decision_source"] == "reviewed_missingness_worker"
        ),
        "family_default_count": sum(
            1
            for row in column_contracts
            if row["type_decision_source"] == "family_default" or row["missingness_decision_source"] == "family_default"
        ),
        "deterministic_baseline_count": sum(
            1
            for row in column_contracts
            if row["type_decision_source"] == "a17_baseline" and row["missingness_decision_source"] == "a17_baseline"
        ),
        "reviewed_type_count": sum(1 for row in column_contracts if row["type_decision_source"] == "reviewed_type_worker"),
        "fallback_type_count": sum(1 for row in column_contracts if row["type_decision_source"] != "reviewed_type_worker"),
        "key_contract_principles": [
            "Control fields are complete for every contract row; semantic enrichment stays blank unless evidence exists.",
            "Reviewed table layout, semantic context, reviewed specialist outputs, and family defaults outrank the A17 deterministic baseline.",
            "Structured scale mappings, when present, refine ordinal semantics before canon and are reused downstream for derivation planning.",
            "Raw prose is not allowed to directly override structured contract fields in v1.",
        ],
    }

    assumptions: List[Dict[str, Any]] = []
    if semantic_index["skip"]:
        assumptions.append(
            {
                "assumption": "semantic_context_skipped",
                "explanation": f"semantic_context_json used the skip sentinel ({semantic_index['skip_reason'] or 'unspecified reason'}), so semantic enrichment remains blank unless reviewed family semantics exist.",
            }
        )
    if support.get("a17_backfilled"):
        assumptions.append(
            {
                "assumption": "a17_backfilled_during_contract_build",
                "explanation": "A17 was not present for this run, so the service deterministically rebuilt the baseline column-resolution layer before merging the final contract.",
            }
        )
    if any(artifact in {"A3-T", "A3-V", "A4", "A9", "A13", "A14", "A16"} for artifact in support["missing_artifacts"]):
        assumptions.append(
            {
                "assumption": "optional_artifact_fallback_applied",
                "explanation": f"Some optional artifacts were unavailable ({', '.join(sorted(support['missing_artifacts']))}), so the A17 baseline used conservative fallback behavior for the affected dimensions.",
            }
        )
    if any(row["type_decision_source"] == "a17_baseline" or row["missingness_decision_source"] == "a17_baseline" for row in column_contracts):
        assumptions.append(
            {
                "assumption": "a17_baseline_applied_for_unreviewed_columns",
                "explanation": "Columns not covered by reviewed specialist outputs or family defaults inherited deterministic baseline decisions from A17 rather than ad hoc endpoint heuristics.",
            }
        )
    if any(row["type_decision_source"] == "family_default" or row["missingness_decision_source"] == "family_default" for row in column_contracts):
        assumptions.append(
            {
                "assumption": "family_defaults_propagated",
                "explanation": "Reviewed family defaults were propagated to sibling columns where explicit per-column reviewed overrides were absent.",
            }
        )
    if any("scale_mapping_resolver" in (row.get("applied_sources") or []) for row in column_contracts):
        assumptions.append(
            {
                "assumption": "scale_mapping_semantics_applied",
                "explanation": "Resolved scale mappings were applied before canon so ordered response semantics could be confirmed without changing raw canonical storage away from strings.",
            }
        )

    output = {
        "worker": "canonical_column_contract_builder",
        "summary": summary,
        "column_contracts": column_contracts,
        "global_value_rules": global_rules,
        "review_flags": top_level_review_flags,
        "assumptions": assumptions,
    }

    validation_errors = _validate_canonical_column_contract_output(
        output,
        expected_source_columns=a2_order,
        global_token_missing_placeholders_detected=global_token_missing_placeholders_detected,
    )
    if validation_errors:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "canonical column contract synthesis produced an invalid payload",
                "errors": validation_errors,
            },
        )

    return output


@app.post("/contracts/scale-mappings")
def build_scale_mapping_contract(
    req: ScaleMappingResolverRequest,
    _=Depends(require_token),
) -> Dict[str, Any]:
    light_contract_decisions = _coerce_json_input(req.light_contract_decisions, "light_contract_decisions")
    family_worker_json = _coerce_json_input(req.family_worker_json, "family_worker_json", allow_list=True)
    extractor_json = (
        _coerce_json_input(req.scale_mapping_extractor_json, "scale_mapping_extractor_json")
        if req.scale_mapping_extractor_json not in (None, "")
        else {}
    )

    return _build_scale_mapping_contract(
        run_id=req.run_id,
        light_contract_decisions=light_contract_decisions,
        family_worker_json=family_worker_json,
        scale_mapping_extractor_json=extractor_json,
    )


@app.post("/contracts/canonical-columns")
def build_canonical_column_contract(
    req: CanonicalColumnContractRequest,
    _=Depends(require_token),
) -> Dict[str, Any]:
    light_contract_decisions = _coerce_json_input(req.light_contract_decisions, "light_contract_decisions")
    semantic_context_json = _coerce_json_input(req.semantic_context_json, "semantic_context_json")
    type_transform_worker_json = _coerce_json_input(req.type_transform_worker_json, "type_transform_worker_json")
    missingness_worker_json = _coerce_json_input(req.missingness_worker_json, "missingness_worker_json")
    family_worker_json = _coerce_json_input(req.family_worker_json, "family_worker_json", allow_list=True)
    table_layout_worker_json = _coerce_json_input(req.table_layout_worker_json, "table_layout_worker_json")
    scale_mapping_json = _coerce_json_input(req.scale_mapping_json, "scale_mapping_json") if req.scale_mapping_json is not None else {}

    if not isinstance(light_contract_decisions.get("primary_grain_decision"), dict):
        raise HTTPException(status_code=422, detail="light_contract_decisions.primary_grain_decision is required")
    if not isinstance(table_layout_worker_json.get("column_table_assignments"), list):
        raise HTTPException(status_code=422, detail="table_layout_worker_json.column_table_assignments must be an array")
    if not isinstance(table_layout_worker_json.get("table_suggestions"), list):
        raise HTTPException(status_code=422, detail="table_layout_worker_json.table_suggestions must be an array")

    support = _load_canonical_support_artifacts(req.run_id)

    return _synthesize_canonical_column_contract(
        run_id=req.run_id,
        light_contract_decisions=light_contract_decisions,
        semantic_context_json=semantic_context_json,
        type_transform_worker_json=type_transform_worker_json,
        missingness_worker_json=missingness_worker_json,
        family_worker_json=family_worker_json,
        table_layout_worker_json=table_layout_worker_json,
        scale_mapping_json=scale_mapping_json,
        support=support,
    )


if os.getenv("RUN_PRUNING_SMOKE_TESTS", "").strip().lower() in {"1", "true", "yes"}:
    _run_pruning_smoke_checks()
