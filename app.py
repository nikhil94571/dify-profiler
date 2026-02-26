from typing import Any, Dict, List, Optional, Tuple
import io
import os
import hashlib
import time
import json
import uuid
import logging
import re
from collections import defaultdict, deque, Counter
from itertools import combinations
from datetime import timedelta
from uuid import uuid4
from manifest_export import upload_and_sign_text

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from google.cloud import storage
import google.auth
from google.auth import impersonated_credentials


from xlsx_export import build_xlsx_bytes


try:
    from dateutil import parser as dateparser  # optional fallback
except Exception:
    dateparser = None

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
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail", "/evidence_associations", "/export/light-contract-xlsx", "/full-bundle") and request.method.upper() == "POST":
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
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail", "/evidence_associations", "/export/light-contract-xlsx", "/full-bundle") and request.method.upper() == "POST":
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

                # Missingness penalty: light penalty until 20%, then steeper.
                mp = missing_pct / 100.0
                if mp <= 0.20:
                    miss_penalty = 1.0 - 0.25 * (mp / 0.20)  # down to 0.75 at 20%
                else:
                    miss_penalty = 0.75 - 0.75 * ((mp - 0.20) / 0.80)  # down to ~0 at 100%
                miss_penalty = _clamp(miss_penalty, 0.0, 1.0)

                # Lift the floor so clean low-card categoricals exceed 0.90
                cat_conf = _clamp(0.25 + 0.75 * uniq_score * miss_penalty, 0.0, 0.99)

                add_candidate(cat_conf, {
                    "type": "categorical",
                    "parse": "levels",
                    "evidence": {
                        "unique_count": unique_count,
                        "max_categorical_cardinality": max_cat,
                        "multi_token_pct": multi_token_pct,
                        "missing_pct": missing_pct,
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
@app.post("/profile")
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
@app.post("/profile_summary")
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
@app.post("/profile_column_detail")
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

@app.post("/evidence_associations")
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


@app.post("/export/light-contract-xlsx")
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

    def _sanitize_filename(name: str) -> str:
        name = (name or "").strip().replace("\n", "").replace("\r", "")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        if not name.lower().endswith(".xlsx"):
            name += ".xlsx"
        return name or "light_contract_template.xlsx"

    filename = _sanitize_filename(req.filename or "light_contract_template.xlsx")

    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    ttl_minutes = int(os.getenv("EXPORT_SIGNED_URL_TTL_MINUTES", "30"))

    # (Decision) Unique object name per request.
    # Why it matters: avoids collisions and makes requests idempotent-ish for debugging.
    object_name = f"exports/{uuid4().hex}_{filename}"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        blob.upload_from_string(
            xlsx_bytes,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # (Decision) Signed URL for temporary access using IAM SignBlob (Cloud Run-safe).
        # Why it matters: Cloud Run default credentials are token-only and cannot sign V4 URLs directly.
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


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS upload/sign failed: {e}")

    return {
        "status": "success",
        "filename": filename,
        "object": object_name,
        "signed_url": signed_url,
        "expires_minutes": ttl_minutes,
    }

from fastapi import HTTPException

@app.post("/export/manifest-txt")
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
    "A3": {"filename": "review_queue.json", "content_type": "application/json"},
    "A4": {"filename": "missingness_catalog.json", "content_type": "application/json"},
    "A5": {"filename": "key_candidates_and_integrity.json", "content_type": "application/json"},
    "A6": {"filename": "grain_tests.json", "content_type": "application/json"},
    "A7": {"filename": "duplicates_report.json", "content_type": "application/json"},
    "A8": {"filename": "repeat_dimension_candidates.json", "content_type": "application/json"},
    "A9": {"filename": "role_scores.json", "content_type": "application/json"},
    "A10": {"filename": "relationships_and_derivations.json", "content_type": "application/json"},
    "A11": {"filename": "glimpses.json", "content_type": "application/json"},
    "A12": {"filename": "table_layout_candidates.json", "content_type": "application/json"},
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
        families.append({
            "family_id": stem,
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

    return {
        "artifact": "A8",
        "purpose": "repeat_family_detection",
        "families": sorted(families, key=lambda f: (-float(f.get("detection_confidence", 0.0)), -int(f.get("columns_count", 0)), str(f.get("family_id", "")))),
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


def _build_review_queue(df: pd.DataFrame, cols_profile: Dict[str, Any], max_cat: int) -> Dict[str, Any]:
    policy = {
        "review_rules_version": "v2",
        "low_confidence_threshold": 0.9,
        "close_gap_threshold": 0.15,
        "dynamic_candidate_preview_enabled": True,
    }
    review_rows: List[Dict[str, Any]] = []
    for col in df.columns:
        p = cols_profile.get(col, {})
        cands = p.get("candidates") or []
        top = cands[0] if cands else {}
        second = cands[1] if len(cands) > 1 else {}
        third = cands[2] if len(cands) > 2 else {}
        top_conf = float(top.get("confidence", 0.0) or 0.0)
        second_conf = float(second.get("confidence", 0.0) or 0.0)
        third_conf = float(third.get("confidence", 0.0) or 0.0)
        conf_gap = float(top_conf - second_conf)
        conf_gap_1_3 = float(top_conf - third_conf) if third else None
        unique_count = int(p.get("unique_count", 0) or 0)
        raw_inferred = str(p.get("inferred_type", "text"))
        top_type = str(top.get("type") or "")
        second_type = str(second.get("type") or "")
        delimiter_stats = (p.get("patterns", {}).get("multi_value") or {})
        delimiter_options = delimiter_stats.get("top_separators") or []
        delimiter_ambiguous = len(delimiter_options) > 1

        reasons: List[str] = []
        reason_detail: List[str] = []
        if top_conf < policy["low_confidence_threshold"]:
            reasons.append("low_confidence")
            reason_detail.append(f"low_confidence:top_conf={round(top_conf, 6)}")
        if conf_gap < policy["close_gap_threshold"]:
            reasons.append("ambiguous_top_candidates")
            reason_detail.append(f"ambiguous_top_candidates:gap_1_2={round(conf_gap, 6)}")

        fam1 = _candidate_family(top_type)
        fam2 = _candidate_family(second_type)
        if fam1 != fam2 and second_type:
            reasons.append("dtype_candidate_conflict")
            reason_detail.append(f"dtype_candidate_conflict:{top_type}_vs_{second_type}")

        raw_dtype_family = "numeric" if raw_inferred in {"integer", "float"} else ("temporal" if raw_inferred == "datetime" else "textual")
        if top_type and _candidate_family(top_type) != raw_dtype_family:
            reasons.append("dtype_candidate_conflict")
            reason_detail.append(f"dtype_candidate_conflict:raw_{raw_inferred}_vs_{top_type}")

        if delimiter_ambiguous:
            reasons.append("delimiter_ambiguity")
            reason_detail.append(f"delimiter_ambiguity:options={','.join(delimiter_options[:4])}")

        review = p.get("review") or {}
        review_reason = str(review.get("reason") or "")
        if review.get("recommended"):
            if any(x in review_reason for x in ["type_requires_transform", "parse_requires_transform", "op_requires_transform"]):
                reasons.append("transform_required")
                reason_detail.append(f"transform_required:{review_reason}")
            else:
                reasons.append("ambiguous_top_candidates")

        preview_n = 2
        if top_conf < 0.75 or conf_gap < 0.10:
            preview_n = 3
        if conf_gap_1_3 is not None and conf_gap_1_3 < 0.10 and len(cands) > 3:
            preview_n = 4
        preview_n = min(preview_n, len(cands))
        preview = [
            {
                "rank": i + 1,
                "type": c.get("type"),
                "confidence": round(float(c.get("confidence", 0.0) or 0.0), 6),
                "parse": c.get("parse"),
                "op": c.get("op"),
            }
            for i, c in enumerate(cands[:preview_n])
        ]

        risk_score = 0.0
        if "transform_required" in reasons:
            risk_score += 0.45
        if "ambiguous_top_candidates" in reasons:
            risk_score += 0.25
        if "low_confidence" in reasons:
            risk_score += 0.20
        if "delimiter_ambiguity" in reasons:
            risk_score += 0.05
        if "dtype_candidate_conflict" in reasons:
            risk_score += 0.05
        risk_score = min(0.99, risk_score + max(0.0, 0.1 - max(conf_gap, 0.0)))

        if reasons:
            review_rows.append({
                "column": col,
                "review_priority": "high" if risk_score >= 0.75 else ("medium" if risk_score >= 0.45 else "low"),
                "risk_score": round(risk_score, 6),
                "reason_codes": sorted(set(reasons)),
                "reason_detail": sorted(set(reason_detail)),
                "raw_inferred_type": raw_inferred,
                "candidate_count_total": len(cands),
                "candidate_preview": preview,
                "omitted_candidate_count": max(0, len(cands) - len(preview)),
                "candidate_gaps": {
                    "gap_1_2": round(conf_gap, 6),
                    "gap_1_3": round(conf_gap_1_3, 6) if conf_gap_1_3 is not None else None,
                },
                "top_confidence": round(top_conf, 6),
                "second_confidence": round(second_conf, 6),
                "confidence_gap": round(conf_gap, 6),
                "unique_count": unique_count,
                "delimiter_options": delimiter_options,
                "pattern_multi_token_pct": delimiter_stats.get("multi_token_pct", 0.0),
                "evidence_snippets": {
                    "parse_failure_examples": (p.get("patterns", {}).get("date_like", {}).get("parse_failure_examples", []) or [])[:3],
                    "multi_value_examples": (p.get("patterns", {}).get("multi_value_examples", {}) or {}).get("examples", [])[:3],
                },
            })
    review_rows.sort(key=lambda r: (0 if "transform_required" in r["reason_codes"] else 1, -r["risk_score"], r["top_confidence"], r["confidence_gap"], r["column"]))
    return {"policy": policy, "columns_requiring_review": review_rows, "count": len(review_rows)}


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
            if ra["primary_role"] in {"time_index", "repeat_index_embedded_in_name"}
            and c not in pk and ra["primary_score"] >= 0.5
        ][:3]
        entity_attrs = [c for c, ra in role_assignments.items() if ra["primary_role"] in {"invariant_attr", "coded_categorical"} and c not in pk][:30]
        event_measures = [c for c, ra in role_assignments.items() if ra["primary_role"] in {"measure_numeric", "measure_item"} and c not in pk and c not in repeat_dims][:40]

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
            fam_measures = [c for c in fam_cols if role_assignments.get(c, {}).get("primary_role") in {"measure_numeric", "measure_item", "coded_categorical"}]
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
            "raw_inferred_type": p.get("inferred_type", "text"),
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
        "A3": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A4": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A5": {"uses": ["dataset", "cols_profile", "column_signal_map"]},
        "A6": {"uses": ["A5", "A8", "A9", "dataset", "column_signal_map"]},
        "A7": {"uses": ["dataset", "A6"]},
        "A8": {"uses": ["dataset_columns", "tokenized_column_names"]},
        "A9": {"uses": ["A2", "A5", "A8", "column_signal_map", "cols_profile"]},
        "A10": {"uses": ["association_result", "dataset"]},
        "A11": {"uses": ["A5", "A8", "A9", "A10", "dataset"]},
        "A12": {"uses": ["A5", "A6", "A8", "A9", "A10", "dataset"]},
        "B1": {"uses": ["A8", "A2", "A6", "A11"]},
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

        raw_inferred_type = sig.get("raw_inferred_type", p.get("inferred_type", "text"))
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
            "raw_inferred_type": raw_inferred_type,
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
            "missing_pct": missing_pct,
            "unique_count": p.get("unique_count", 0),
            "unique_ratio": round(unique_ratio, 6),
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

    review_queue = _build_review_queue(df, cols_profile, max_categorical_cardinality)
    review_queue["artifact"] = "A3"
    review_queue["purpose"] = "review_queue"
    review_queue["inputs"] = artifact_inputs["A3"]

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

    seed_cols = list(dict.fromkeys((key_integrity.get("handoff_to_A6", {}).get("recommended_seed_columns") or []) + key_candidate_cols[:8]))[:10]
    repeat_candidates_pool = list(dict.fromkeys(repeat_index_cols + repeat_name_candidates + [c for c in df.columns if any(k in c.lower() for k in ["wave", "visit", "session", "trial", "row", "item", "tp", "timepoint"])][:10]))[:12]
    time_like = list(dict.fromkeys(role_time_candidates))[:8]

    pool = list(dict.fromkeys(seed_cols + repeat_candidates_pool + time_like))
    candidate_sets = []
    for seed in seed_cols[:8]:
        candidate_sets.append([seed])
        for rcol in repeat_candidates_pool[:8]:
            if rcol != seed:
                candidate_sets.append([seed, rcol])
        for tcol in time_like[:6]:
            if tcol != seed:
                candidate_sets.append([seed, tcol])
        for rcol in repeat_candidates_pool[:6]:
            for tcol in time_like[:4]:
                if len({seed, rcol, tcol}) == 3:
                    candidate_sets.append([seed, rcol, tcol])

    # deterministic de-dupe + cap
    seen = set()
    dedup_sets = []
    for ss in candidate_sets:
        t = tuple(ss)
        if t in seen:
            continue
        seen.add(t)
        dedup_sets.append(ss)
    candidate_sets = dedup_sets[:120]

    grain_tests = []
    for idx, keys in enumerate(candidate_sets, start=1):
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
        repeat_dimension_present = any(k in repeat_candidates_pool for k in keys)
        temporal_dimension_present = any(k in time_like for k in keys)
        pivot_safety = "safe" if dup_rows == 0 else ("conditional" if conflict_group_pct <= 0.2 else "unsafe")
        decision = "best_available_composite_key" if len(keys) > 1 else ("candidate_single_key" if dup_rows == 0 else "failed_single_key")

        grain_tests.append({
            "test_id": f"GT{idx}",
            "keys_tested": keys,
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

    grain_tests = sorted(grain_tests, key=lambda g: (g.get("collision_severity_score", 1.0), -g.get("uniqueness_rate", 0.0), g.get("dup_row_count", 10**9), len(g.get("keys_tested", []))))
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
            "seed_columns": seed_cols,
            "repeat_dimension_candidates": repeat_candidates_pool,
            "time_like_candidates": time_like,
            "family_repeat_dimensions": family_repeat_dims,
            "pool_size": len(pool),
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
                "raw_inferred_type", "unique_count", "unique_ratio", "missing_pct",
                "parsed_as_numeric_pct", "parsed_as_datetime_pct",
                "top_candidate_type", "top_candidate_confidence", "top_candidate_parse", "top_candidate_op",
            ],
        },
        "columns": [],
    }
    family_member_cols = {c for fam in families for c in fam.get("columns", [])}
    family_repeat_names = {f.get("recommended_repeat_dimension_name") for f in families if f.get("recommended_repeat_dimension_name")}
    for row in column_dictionary_rows:
        uniq = float(row["unique_ratio"])
        inferred = row["raw_inferred_type"]
        top_candidate = row.get("top_candidate") or {}
        top_type = str(top_candidate.get("type") or "")
        top_conf = float(top_candidate.get("confidence", 0.0) or 0.0)
        parsed_as_numeric_pct = float((row.get("numeric_profile") or {}).get("parseable_pct", 0.0) or 0.0)
        parsed_as_datetime_pct = float((row.get("datetime_profile") or {}).get("parseable_pct", 0.0) or 0.0)
        unique_count = int(row.get("unique_count", 0) or 0)
        sample_values = ((row.get("a2_samples") or {}).get("random") or [])[:5]
        repeat_parse = _parse_repeat_structure_name(row["column"])

        measure_numeric = max(parsed_as_numeric_pct / 100.0, 1.0 if top_type in {"numeric", "numeric_with_unit", "percent", "numeric_range"} else 0.0)
        likert_hint = bool(measure_numeric >= 0.8 and unique_count <= 11 and unique_count > 0)
        family_member_hint = row["column"] in family_member_cols
        coded_cat_hint = bool(measure_numeric >= 0.8 and unique_count <= max_categorical_cardinality)
        identifier_numeric_hint = bool(measure_numeric >= 0.8 and uniq >= 0.98 and unique_count > max_categorical_cardinality)

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
                "role": "measure_numeric",
                "score": round(min(1.0, measure_numeric * (0.8 if not identifier_numeric_hint else 0.4)), 6),
                "evidence": {"parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6), "top_candidate_type": top_type, "identifier_numeric_hint": identifier_numeric_hint},
            },
            {
                "role": "measure_item",
                "score": round(0.9 if (likert_hint and family_member_hint) else (0.85 if likert_hint else (0.65 if family_member_hint else 0.2)), 6),
                "evidence": {"parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6), "unique_count": unique_count, "value_range_small_integer": likert_hint, "family_member_hint": family_member_hint},
            },
            {
                "role": "coded_categorical",
                "score": round(0.8 if coded_cat_hint else 0.2, 6),
                "evidence": {"parsed_as_numeric_pct": round(parsed_as_numeric_pct, 6), "unique_count": unique_count, "max_categorical_cardinality": int(max_categorical_cardinality)},
            },
            {
                "role": "repeat_index_embedded_in_name",
                "score": round(0.82 if repeat_parse else 0.1, 6),
                "evidence": {"repeat_name_parse": repeat_parse},
            },
            {
                "role": "invariant_attr",
                "score": round(min(1.0, max(0.0, 1.0 - uniq)), 6),
                "evidence": {"unique_ratio": round(uniq, 6)},
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
                "family_repeat_dimensions": sorted([x for x in family_repeat_names if x]),
                "missing_pct": row.get("missing_pct", 0.0),
                "numeric_profile_parseable_pct": parsed_as_numeric_pct,
                "datetime_profile_parseable_pct": parsed_as_datetime_pct,
            },
            "evidence_not_used": [],
            "features": {"unique_ratio": uniq, "raw_inferred_type": inferred, "top_candidate_type": top_type},
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
    repeat_cols = [c for c in df.columns if (role_by_col.get(c, {}).get("role_scores", {}).get("repeat_index_embedded_in_name", 0.0) >= 0.5)][:3]
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
                "confidence": role_by_col.get(c, {}).get("role_scores", {}).get("time_index") if c in time_cols else role_by_col.get(c, {}).get("role_scores", {}).get("repeat_index_embedded_in_name"),
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

    measure_candidates = [c for c in df.columns if max(role_by_col.get(c, {}).get("role_scores", {}).get("measure_numeric", 0.0), role_by_col.get(c, {}).get("role_scores", {}).get("measure_item", 0.0)) >= 0.5 and c not in grain_cols and not _is_low_signal_col(c)]
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
                "confidence": max(role_by_col.get(c, {}).get("role_scores", {}).get("measure_numeric", 0.0), role_by_col.get(c, {}).get("role_scores", {}).get("measure_item", 0.0)) if c not in key_cols[:1] else None,
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
    for fam in families[:3]:
        fam_cols = [c for c in fam.get("columns", []) if c in df.columns and not _is_low_signal_col(c)]
        cols = list(dict.fromkeys(key_cols[:1] + repeat_cols[:1] + fam_cols[:8]))
        family_panels.append({
            "panel_id": f"family_{fam.get('family_id')}_preview",
            "purpose": "Question block / repeated measures preview",
            "columns": cols,
            "selection_trace": [
                {
                    "column": c,
                    "reason": "family membership",
                    "source_artifact": "A8",
                    "confidence": None,
                    "rank": None,
                    "fallback_rule": None,
                }
                for c in cols
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
            cols = list(dict.fromkeys(key_cols[:1] + sorted(cols_raw)[:8]))
            family_panels.append({
                "panel_id": f"pseudo_family_{stem}_preview",
                "purpose": "Fallback patterned column cluster when family detection is weak",
                "columns": cols,
                "selection_trace": [
                    {
                        "column": c,
                        "reason": "patterned stem cluster fallback",
                        "source_artifact": "A11_fallback",
                        "confidence": None,
                        "rank": None,
                        "fallback_rule": "no_family_detected_pattern_cluster",
                    }
                    for c in cols
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
            "selection_trace": p["selection_trace"],
            "rows": rows,
            "family_reference": p.get("family_reference"),
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
    for fam in families:
        fam_cols = fam["columns"]
        family_packets.append({
            "inputs": artifact_inputs["B1"],
            "family_id": fam["family_id"],
            "columns": fam_cols,
            "detected_pattern_index_summary": {
                "patterns": fam["patterns"],
                "index": fam["extracted_index_set"],
                "index_type_candidate": fam["index_type_candidate"],
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

    run_id = uuid4().hex
    base_url = str(request.base_url).rstrip("/")
    bucket_name = os.getenv("EXPORT_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Missing EXPORT_BUCKET env var")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    payloads: Dict[str, bytes] = {
        "A2": _jsonl_bytes(column_dictionary_rows),
        "A3": _json_bytes(review_queue),
        "A4": _json_bytes(missing_catalog),
        "A5": _json_bytes(key_integrity),
        "A6": _json_bytes(grain_validation),
        "A7": _json_bytes(duplicates_report),
        "A8": _json_bytes(repeat_candidates),
        "A9": _json_bytes(role_scores),
        "A10": _json_bytes(rel),
        "A11": _json_bytes(glimpses),
        "A12": _json_bytes(table_layout_candidates),
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

    # Add A1 record into manifest and persist once more under strict single schema    a1_entry = {
        "artifact_id": "A1",
        "filename": ARTIFACT_SPECS["A1"]["filename"],
        "object_path": manifest_object_path,
        "bucket": bucket_name,
        "content_type": manifest_meta["content_type"],
        "sha256": manifest_meta["sha256"],
        "size_bytes": manifest_meta["size_bytes"],
        "download_url": _build_artifact_url(base_url, "A1", run_id, "download"),
        "meta_url": _build_artifact_url(base_url, "A1", run_id, "meta"),
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
        aid = item["artifact_id"]
        entries.append({
            **item,
            "download_url": _build_artifact_url(base_url, aid, run_id, "download"),
            "meta_url": _build_artifact_url(base_url, aid, run_id, "meta"),
        })

    return {
        "run_id": run_id,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "artifacts": entries,
    }


@app.get("/artifacts/{artifact_id}/meta")
def artifact_meta(
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
    for item in manifest.get("artifact_registry", []):
        if item.get("artifact_id") == artifact_id:
            return item
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
