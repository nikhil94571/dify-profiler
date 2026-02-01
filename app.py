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

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

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

# Limits for examples emitted per pattern category
EXAMPLES_PER_PATTERN = int(os.getenv("EXAMPLES_PER_PATTERN", "5"))
RARE_EXAMPLES_MAX = int(os.getenv("RARE_EXAMPLES_MAX", "10"))
EXTREMES_N = int(os.getenv("EXTREMES_N", "5"))

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
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail") and request.method.upper() == "POST":
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
    if request.url.path in ("/profile", "/profile_summary", "/profile_column_detail") and request.method.upper() == "POST":
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


def _cap_scan(s: pd.Series, max_rows: int) -> pd.Series:
    if len(s) <= max_rows:
        return s
    # Deterministic: take head of non-null series
    return s.iloc[:max_rows]


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
    "lb", "lbs", "pound", "pounds",

    # height / length
    "mm", "cm", "m",
    "ft", "feet", "in", "inch", "inches",
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
            "pattern_scan_max_rows": PATTERN_SCAN_MAX_ROWS,
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
            # Step 1 — Value sampling per column
            # -----------------------------
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

            # String length stats (on non-null stringified original values, before truncation)
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
            # Use capped scan for performance deterministically
            scan_series = _cap_scan(s_stripped, PATTERN_SCAN_MAX_ROWS)
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
                        if u in ("kgs", "kilogram", "kilograms"):
                            return "kg"
                        if u in ("lbs", "pound", "pounds"):
                            return "lb"
                        if u in ("feet",):
                            return "ft"
                        if u in ("inch", "inches"):
                            return "in"
                        return u

                    for u in raw_units:
                        cu = _canon_unit(str(u))
                        units_detected[cu] = units_detected.get(cu, 0) + 1

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


                # Date-like patterns and parse attempt
                iso_mask = scan_series.str.match(RE_ISO_DATE, na=False)
                month_mask = scan_series.str.contains(RE_MONTH_NAME, na=False)

                parsed_dt, success_dt, policy = _try_parse_date_for_profile(scan_series)
                parse_success_pct = _pct(int(success_dt.sum()), scan_n)

                # Failure examples
                failures = scan_series[~success_dt]
                parse_failure_examples = _take_examples(failures, EXAMPLES_PER_PATTERN)

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

                        # Build vocab top-k from multi-valued rows only
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
                            best_vocab = [{"token": _truncate(str(k), MAX_STRING_LEN), "count": int(v)} for k, v in tok_vc.items()]
                        else:
                            best_vocab = []

                multi_value = {
                    "delimiter": best_spec["name"] if best_spec is not None else None,
                    "delimiter_pattern": best_spec["pattern"] if best_spec is not None else None,
                    "multi_token_pct": round(best_multi_pct, 6) if best_multi_pct >= 0 else 0.0,
                    "avg_tokens": round(best_tokens_stats[0], 6),
                    "max_tokens": best_tokens_stats[1],
                    "token_vocab_topk": best_vocab,
                }


                # Range-like patterns
                year_range_mask = scan_series.str.match(RE_YEAR_RANGE, na=False)
                num_range_mask = scan_series.str.match(RE_NUM_RANGE, na=False)
                range_examples = _take_examples(scan_series[year_range_mask | num_range_mask], EXAMPLES_PER_PATTERN)

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
                }
                UNIT_KIND_BY_UNIT = {
                    "kg": "weight", "lb": "weight",
                    "mm": "length", "cm": "length", "m": "length", "ft": "length", "in": "length",
                }
                UNIT_CONVERSIONS = {
                    # weight
                    ("lb", "kg"): {"factor": 0.45359237, "formula": "kg = lb * 0.45359237"},
                    ("kg", "lb"): {"factor": 1.0 / 0.45359237, "formula": "lb = kg / 0.45359237"},
                    # length (optional but included for completeness)
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

            # Numeric candidates
            if strict_pct >= 80.0:

                # If a small suffix-multiplier minority exists (e.g., "1.6K"), record that explicitly.
                # This prevents downstream systems from assuming "strict-only".
                parse_mode = "strict"
                if suffix_multiplier_pct > 0.0:
                    parse_mode = "strict_with_suffix_minority"

                add_candidate(strict_pct / 100.0, {
                    "type": "numeric",
                    "parse": parse_mode,
                    "evidence": {
                        "strict_pct": strict_pct,
                        "suffix_multiplier_pct": suffix_multiplier_pct,
                        "suffix_pct": suffix_pct,
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
            if multi_token_pct >= 20.0 and multi_like.get("delimiter_pattern") is not None:
                add_candidate(min(0.9, multi_token_pct / 100.0 + 0.1), {
                    "type": "categorical_multi",
                    "op": f"split_regex(pattern='{multi_like.get('delimiter_pattern')}')",
                    "evidence": {
                        "multi_token_pct": multi_token_pct,
                        "delimiter": multi_like.get("delimiter"),
                        "delimiter_pattern": multi_like.get("delimiter_pattern"),
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

            if is_low_card and (not is_multiish) and (not is_rangeish) and (not is_dateish) and (not is_numericish):
                # Confidence increases as uniques approach 2-5 and missingness is low
                # Map uniques: 2->1.0, 5->0.8, max_cat->0.3 (monotonic decreasing)
                if unique_count <= 1:
                    uniq_score = 0.2
                elif unique_count <= 2:
                    uniq_score = 1.0
                elif unique_count <= 5:
                    uniq_score = 0.8
                else:
                    # linear falloff to 0.3 at max_cat
                    uniq_score = 0.8 - (0.5 * ((unique_count - 5) / max(1, (max_cat - 5))))
                    uniq_score = _clamp(uniq_score, 0.3, 0.8)

                miss_penalty = _clamp(1.0 - (missing_pct / 100.0), 0.0, 1.0)
                cat_conf = _clamp(0.15 + 0.85 * uniq_score * miss_penalty, 0.0, 0.99)

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

            # -----------------------------
            # Step 5 — Outlier visibility & parseability (numeric/date)
            # -----------------------------
            # Numeric parseability (value-level) for ANY column using string scan (use scan_series)
            if scan_n > 0:
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
                "evidence": top.get("evidence"),
            },
            # optional but useful for routing to cleaner without extra calls
            "unit_plan": info.get("unit_plan"),

        })

    return {
        "dataset_id": full["dataset_id"],
        "dataset_sha256": full["dataset_sha256"],
        "n_rows": full["n_rows"],
        "n_columns": full["n_columns"],
        "summary": full["summary"],
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



@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

