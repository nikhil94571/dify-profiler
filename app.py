from typing import Any, Dict, List, Optional, Tuple
import io
import os
import hashlib
import time
import json
import uuid
import logging
import re
from collections import defaultdict, deque
from datetime import timedelta
from uuid import uuid4
from manifest_export import upload_and_sign_text

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response
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
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail", "/evidence_associations", "/export/light-contract-xlsx") and request.method.upper() == "POST":
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
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail", "/evidence_associations", "/export/light-contract-xlsx") and request.method.upper() == "POST":
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