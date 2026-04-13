import json
import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "testdata" / "canonical_reviewer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"Valid config keys have changed in V2:.*",
    category=UserWarning,
)

if "google" not in sys.modules:
    google_mod = types.ModuleType("google")
    cloud_mod = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    auth_mod = types.ModuleType("google.auth")
    auth_transport_mod = types.ModuleType("google.auth.transport")
    auth_requests_mod = types.ModuleType("google.auth.transport.requests")
    impersonated_mod = types.ModuleType("google.auth.impersonated_credentials")
    class _StorageClient:
        pass
    class Bucket:
        pass
    class Blob:
        pass
    storage_mod.Client = _StorageClient
    storage_mod.Bucket = Bucket
    storage_mod.Blob = Blob
    auth_mod.default = lambda *args, **kwargs: (object(), None)
    auth_requests_mod.Request = object
    auth_mod.impersonated_credentials = impersonated_mod
    google_mod.cloud = cloud_mod
    google_mod.auth = auth_mod
    cloud_mod.storage = storage_mod
    sys.modules["google"] = google_mod
    sys.modules["google.cloud"] = cloud_mod
    sys.modules["google.cloud.storage"] = storage_mod
    sys.modules["google.auth"] = auth_mod
    sys.modules["google.auth.transport"] = auth_transport_mod
    sys.modules["google.auth.transport.requests"] = auth_requests_mod
    sys.modules["google.auth.impersonated_credentials"] = impersonated_mod

if "pandas" not in sys.modules:
    pandas_mod = types.ModuleType("pandas")
    class Series:
        pass
    class DataFrame:
        pass
    pandas_mod.Series = Series
    pandas_mod.DataFrame = DataFrame
    sys.modules["pandas"] = pandas_mod

if "fastapi" not in sys.modules:
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_security_mod = types.ModuleType("fastapi.security")
    fastapi_responses_mod = types.ModuleType("fastapi.responses")

    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def middleware(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def get(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def post(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:
        headers = {}
        client = None
        state = types.SimpleNamespace()

    class UploadFile:
        pass

    def Depends(value=None):
        return value

    def File(*args, **kwargs):
        return None

    def Form(*args, **kwargs):
        return None

    class HTTPBearer:
        def __init__(self, *args, **kwargs):
            pass

    class HTTPAuthorizationCredentials:
        credentials = ""

    class JSONResponse:
        def __init__(self, *args, **kwargs):
            pass

    class Response:
        def __init__(self, *args, **kwargs):
            pass

    class StreamingResponse:
        def __init__(self, *args, **kwargs):
            pass

    fastapi_mod.FastAPI = FastAPI
    fastapi_mod.UploadFile = UploadFile
    fastapi_mod.File = File
    fastapi_mod.Form = Form
    fastapi_mod.Depends = Depends
    fastapi_mod.HTTPException = HTTPException
    fastapi_mod.Request = Request
    fastapi_security_mod.HTTPBearer = HTTPBearer
    fastapi_security_mod.HTTPAuthorizationCredentials = HTTPAuthorizationCredentials
    fastapi_responses_mod.JSONResponse = JSONResponse
    fastapi_responses_mod.Response = Response
    fastapi_responses_mod.StreamingResponse = StreamingResponse
    sys.modules["fastapi"] = fastapi_mod
    sys.modules["fastapi.security"] = fastapi_security_mod
    sys.modules["fastapi.responses"] = fastapi_responses_mod

if "pydantic" not in sys.modules:
    pydantic_mod = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, **kwargs):
        return default

    pydantic_mod.BaseModel = BaseModel
    pydantic_mod.Field = Field
    sys.modules["pydantic"] = pydantic_mod

if "openpyxl" not in sys.modules:
    openpyxl_mod = types.ModuleType("openpyxl")
    openpyxl_styles_mod = types.ModuleType("openpyxl.styles")
    openpyxl_utils_mod = types.ModuleType("openpyxl.utils")
    openpyxl_worksheet_mod = types.ModuleType("openpyxl.worksheet")
    openpyxl_validation_mod = types.ModuleType("openpyxl.worksheet.datavalidation")

    class Workbook:
        pass

    def load_workbook(*args, **kwargs):
        return None

    class Alignment:
        def __init__(self, *args, **kwargs):
            pass

    class Font:
        def __init__(self, *args, **kwargs):
            pass

    class PatternFill:
        def __init__(self, *args, **kwargs):
            pass

    def get_column_letter(value):
        return str(value)

    class DataValidation:
        def __init__(self, *args, **kwargs):
            pass

    openpyxl_mod.Workbook = Workbook
    openpyxl_mod.load_workbook = load_workbook
    openpyxl_styles_mod.Alignment = Alignment
    openpyxl_styles_mod.Font = Font
    openpyxl_styles_mod.PatternFill = PatternFill
    openpyxl_utils_mod.get_column_letter = get_column_letter
    openpyxl_validation_mod.DataValidation = DataValidation
    sys.modules["openpyxl"] = openpyxl_mod
    sys.modules["openpyxl.styles"] = openpyxl_styles_mod
    sys.modules["openpyxl.utils"] = openpyxl_utils_mod
    sys.modules["openpyxl.worksheet"] = openpyxl_worksheet_mod
    sys.modules["openpyxl.worksheet.datavalidation"] = openpyxl_validation_mod

from app import (
    _build_baseline_column_resolution_artifact,
    _synthesize_canonical_column_contract,
    _validate_canonical_column_contract_output,
)


def _load_validator_main():
    def _validator(payload_json, expected_source_columns_json):
        payload = json.loads(payload_json)
        expected_source_columns = json.loads(expected_source_columns_json)
        errors = _validate_canonical_column_contract_output(payload, expected_source_columns=expected_source_columns)
        return {
            "validation_ok": "true" if not errors else "false",
            "validation_error": "" if not errors else errors[0],
            "validation_errors_json": json.dumps(errors, separators=(",", ":")),
        }

    return _validator


def _load_standalone_validator_main():
    namespace = {}
    code = Path("JSON validators/canonical_column_contract_validator.json").read_text()
    exec(code, namespace)
    return namespace["main"]


def _load_fixture_json(name):
    return json.loads((FIXTURES / name).read_text())


def _support():
    a2_rows = [
        {
            "column": "Unnamed: 0",
            "top_candidate": {"type": "numeric", "confidence": 0.99},
            "confidence": 0.99,
            "unique_count": 3,
            "unique_ratio": 1.0,
            "missing_pct": 0.0,
            "missing_tokens_observed": {},
            "top_levels": ["1", "2", "3"],
            "a2_samples": {"random": ["1", "2", "3"]},
            "numeric_profile": {"parseable_pct": 100.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": False,
        },
        {
            "column": "RespondentId",
            "top_candidate": {"type": "numeric", "confidence": 0.96},
            "confidence": 0.96,
            "unique_count": 3,
            "unique_ratio": 1.0,
            "missing_pct": 0.0,
            "missing_tokens_observed": {},
            "top_levels": ["001", "002", "003"],
            "a2_samples": {"random": ["001", "002", "003"]},
            "numeric_profile": {"parseable_pct": 100.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": False,
        },
        {
            "column": "StatusCode",
            "top_candidate": {"type": "numeric", "confidence": 0.72},
            "confidence": 0.72,
            "unique_count": 2,
            "unique_ratio": 0.2,
            "missing_pct": 0.0,
            "missing_tokens_observed": {},
            "top_levels": ["1", "2"],
            "a2_samples": {"random": ["1", "2"]},
            "numeric_profile": {"parseable_pct": 100.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": False,
        },
        {
            "column": "Q12",
            "top_candidate": {"type": "categorical", "confidence": 0.91},
            "confidence": 0.91,
            "unique_count": 2,
            "unique_ratio": 0.1,
            "missing_pct": 0.0,
            "missing_tokens_observed": {},
            "top_levels": ["Yes", "No"],
            "a2_samples": {"random": ["Yes", "No"]},
            "numeric_profile": {"parseable_pct": 0.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": False,
        },
        {
            "column": "Q13_Reason",
            "top_candidate": {"type": "text", "confidence": 0.88},
            "confidence": 0.88,
            "unique_count": 2,
            "unique_ratio": 0.7,
            "missing_pct": 70.0,
            "missing_tokens_observed": {},
            "top_levels": ["Too expensive", "Too hard"],
            "a2_samples": {"random": ["Too expensive", "Too hard"]},
            "numeric_profile": {"parseable_pct": 0.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": True,
        },
        {
            "column": "FreeTextComment",
            "top_candidate": {"type": "text", "confidence": 0.83},
            "confidence": 0.83,
            "unique_count": 3,
            "unique_ratio": 1.0,
            "missing_pct": 0.0,
            "missing_tokens_observed": {},
            "top_levels": ["Great", "Bad", "Okay"],
            "a2_samples": {"random": ["Great service", "Could be better", "Okay overall"]},
            "numeric_profile": {"parseable_pct": 0.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": False,
        },
        {
            "column": "Q1FreeText",
            "top_candidate": {"type": "numeric", "confidence": 0.78},
            "confidence": 0.78,
            "unique_count": 3,
            "unique_ratio": 0.9,
            "missing_pct": 0.0,
            "missing_tokens_observed": {},
            "top_levels": ["1", "2", "3"],
            "a2_samples": {"random": ["Loved it overall", "Needs more support", "Neutral response"]},
            "numeric_profile": {"parseable_pct": 100.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": False,
        },
        {
            "column": "Q6Main_cell_groupRow1",
            "top_candidate": {"type": "categorical", "confidence": 0.79},
            "confidence": 0.79,
            "unique_count": 5,
            "unique_ratio": 0.5,
            "missing_pct": 0.0,
            "missing_tokens_observed": {},
            "top_levels": ["Strongly agree", "Agree", "Neutral"],
            "a2_samples": {"random": ["Strongly agree", "Agree", "Neutral"]},
            "numeric_profile": {"parseable_pct": 0.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": False,
        },
        {
            "column": "Q18Main_cell_groupRow1",
            "top_candidate": {"type": "numeric", "confidence": 0.8},
            "confidence": 0.8,
            "unique_count": 5,
            "unique_ratio": 0.4,
            "missing_pct": 52.0,
            "missing_tokens_observed": {},
            "top_levels": ["1", "2", "3", "4", "5"],
            "a2_samples": {"random": ["1", "2", "3", "4", "5"]},
            "numeric_profile": {"parseable_pct": 100.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": True,
        },
        {
            "column": "Q18Main_cell_groupRow2",
            "top_candidate": {"type": "numeric", "confidence": 0.8},
            "confidence": 0.8,
            "unique_count": 5,
            "unique_ratio": 0.4,
            "missing_pct": 55.0,
            "missing_tokens_observed": {},
            "top_levels": ["1", "2", "3", "4", "5"],
            "a2_samples": {"random": ["1", "2", "3", "4", "5"]},
            "numeric_profile": {"parseable_pct": 100.0},
            "datetime_profile": {"parseable_pct": 0.0},
            "high_missingness": True,
        },
    ]
    a9_rows = [
        {"column": "RespondentId", "primary_role": "id_key", "encoding_type": "nominal"},
        {"column": "StatusCode", "primary_role": "coded_categorical", "encoding_type": "nominal"},
        {"column": "Q12", "primary_role": "invariant_attr", "encoding_type": "nominal"},
        {"column": "Q13_Reason", "primary_role": "invariant_attr", "encoding_type": "nominal"},
        {"column": "FreeTextComment", "primary_role": "invariant_attr", "encoding_type": "nominal"},
        {"column": "Q1FreeText", "primary_role": "measure", "encoding_type": "nominal"},
        {"column": "Q6Main_cell_groupRow1", "primary_role": "measure_item", "encoding_type": "ordinal"},
        {"column": "Q18Main_cell_groupRow1", "primary_role": "measure_item", "encoding_type": "ordinal"},
        {"column": "Q18Main_cell_groupRow2", "primary_role": "measure_item", "encoding_type": "ordinal"},
    ]
    a13_rows = [
        {"column": "RespondentId", "detected_anchors": []},
        {"column": "StatusCode", "detected_anchors": []},
        {"column": "Q12", "detected_anchors": []},
        {"column": "Q13_Reason", "detected_anchors": []},
        {"column": "FreeTextComment", "detected_anchors": [{"anchor": "EMAIL"}]},
        {"column": "Q1FreeText", "detected_anchors": []},
        {"column": "Q6Main_cell_groupRow1", "detected_anchors": []},
        {"column": "Q18Main_cell_groupRow1", "detected_anchors": []},
        {"column": "Q18Main_cell_groupRow2", "detected_anchors": []},
    ]
    a14_rows = [
        {"column": "RespondentId", "global_quality_score": 0.99, "drift_detected": False},
        {"column": "StatusCode", "global_quality_score": 0.98, "drift_detected": False},
        {"column": "Q12", "global_quality_score": 0.97, "drift_detected": False},
        {"column": "Q13_Reason", "global_quality_score": 0.89, "drift_detected": False},
        {"column": "FreeTextComment", "global_quality_score": 0.95, "drift_detected": False},
        {"column": "Q1FreeText", "global_quality_score": 0.94, "drift_detected": False},
        {"column": "Q6Main_cell_groupRow1", "global_quality_score": 0.96, "drift_detected": False},
        {"column": "Q18Main_cell_groupRow1", "global_quality_score": 0.90, "drift_detected": False},
        {"column": "Q18Main_cell_groupRow2", "global_quality_score": 0.89, "drift_detected": False},
    ]
    a17 = _build_baseline_column_resolution_artifact(
        a2_rows=a2_rows,
        a3t_payload={"items": []},
        a3v_payload={"items": []},
        a4_payload={
            "per_column": [
                {"column": "Q13_Reason", "missing_pct": 70.0, "token_breakdown": {}},
                {"column": "Q18Main_cell_groupRow1", "missing_pct": 52.0, "token_breakdown": {}},
                {"column": "Q18Main_cell_groupRow2", "missing_pct": 55.0, "token_breakdown": {}},
            ]
        },
        a9_payload={"columns": a9_rows},
        a13_payload={"columns": a13_rows},
        a14_payload={"columns": a14_rows},
        a16_payload={
            "detected_skip_logic": [
                {
                    "trigger_column": "Q12",
                    "missing_explained_pct": 97.0,
                    "directionality": "forward",
                    "sample_affected_columns": ["Q18Main_cell_groupRow1", "Q18Main_cell_groupRow2"],
                }
            ],
            "master_switch_candidates": [],
        },
    )
    return {
        "a2_rows": a2_rows,
        "a2_by_col": {row["column"]: row for row in a2_rows},
        "a2_order": [row["column"] for row in a2_rows],
        "a17_by_col": {row["column"]: row for row in a17["columns"]},
        "family_by_column": {
            "Q6Main_cell_groupRow1": "q_6_main_cell_group",
            "Q18Main_cell_groupRow1": "q_18_main_cell_group",
            "Q18Main_cell_groupRow2": "q_18_main_cell_group",
        },
        "missing_artifacts": [],
        "a17_backfilled": False,
    }


def _inputs():
    return {
        "light_contract_decisions": {
            "primary_grain_decision": {"status": "accept", "keys": ["RespondentId"], "comments": ""},
            "reference_decisions": [],
            "family_decisions": [
                {
                    "family_id": "q_6_main_cell_group",
                    "status": "accept",
                    "table_name": "q6_main_cell_group_rows",
                    "repeat_index_name": "row",
                    "parent_key": "RespondentId",
                    "comments": "",
                },
                {
                    "family_id": "q_18_main_cell_group",
                    "status": "accept",
                    "table_name": "q18_main_cell_group_rows",
                    "repeat_index_name": "row",
                    "parent_key": "RespondentId",
                    "comments": "",
                }
            ],
        },
        "semantic_context_json": {
            "worker": "semantic_context_interpreter",
            "summary": {"overview": "Smoke test semantic context", "key_points": ["Q12 is a gate", "StatusCode has labels"]},
            "dataset_context": {
                "dataset_purpose": "",
                "row_meaning_notes": "",
                "collection_change_notes": [],
                "known_optional_or_conditioned_sections": []
            },
            "important_variables": [
                {
                    "column_or_family": "Q12",
                    "kind": "condition_column",
                    "meaning": "Master gating variable for conditional follow-up questions.",
                    "downstream_importance": "Protect downstream gated fields from generic null penalties.",
                    "confidence": 0.97
                }
            ],
            "codebook_hints": [
                {
                    "column": "StatusCode",
                    "codes_or_labels_note": "1 = active, 2 = paused",
                    "meaning": "Operational status code for the respondent record.",
                    "confidence": 0.96
                }
            ],
            "review_flags": [],
            "assumptions": []
        },
        "type_transform_worker_json": {
            "worker": "type_value_specialist",
            "summary": {"overview": "Smoke test type decisions", "key_patterns": []},
            "column_decisions": [
                {
                    "column": "RespondentId",
                    "recommended_logical_type": "identifier",
                    "recommended_storage_type": "string",
                    "transform_actions": ["trim_whitespace", "cast_to_string"],
                    "structural_transform_hints": [],
                    "interpretation_hints": ["identifier_not_measure"],
                    "normalization_notes": "Preserve respondent IDs as strings.",
                    "confidence": 0.98,
                    "reasoning": "Reviewed ID semantics.",
                    "skip_logic_protected": False,
                    "needs_human_review": False
                },
                {
                    "column": "Q6Main_cell_groupRow1",
                    "recommended_logical_type": "ordinal_category",
                    "recommended_storage_type": "string",
                    "transform_actions": ["trim_whitespace", "normalize_category_tokens"],
                    "structural_transform_hints": ["requires_child_table_review"],
                    "interpretation_hints": ["repeat_context_do_not_use_as_base_key"],
                    "normalization_notes": "Reviewed as ordinal question response.",
                    "confidence": 0.9,
                    "reasoning": "Family member with ordinal labels.",
                    "skip_logic_protected": False,
                    "needs_human_review": False
                },
                {
                    "column": "Q18Main_cell_groupRow1",
                    "recommended_logical_type": "ordinal_category",
                    "recommended_storage_type": "string",
                    "transform_actions": ["trim_whitespace", "normalize_category_tokens"],
                    "structural_transform_hints": [],
                    "interpretation_hints": ["skip_logic_protected"],
                    "normalization_notes": "Reviewed as gated ordinal question response.",
                    "confidence": 0.91,
                    "reasoning": "Explicit reviewed override should outrank family defaults.",
                    "skip_logic_protected": True,
                    "needs_human_review": False
                },
                {
                    "column": "GhostCol",
                    "recommended_logical_type": "mixed_or_ambiguous",
                    "recommended_storage_type": "string",
                    "transform_actions": [],
                    "structural_transform_hints": [],
                    "interpretation_hints": ["mixed_content_high_risk"],
                    "normalization_notes": "",
                    "confidence": 0.7,
                    "reasoning": "Reviewed-only reference not in A2.",
                    "skip_logic_protected": False,
                    "needs_human_review": True
                }
            ],
            "global_transform_rules": [
                {
                    "rule_name": "normalize_missing_tokens",
                    "applies_when": "Stable missing tokens recur across reviewed columns.",
                    "rule_description": "Map stable placeholder tokens to canonical missing values."
                }
            ],
            "review_flags": [],
            "assumptions": []
        },
        "missingness_worker_json": {
            "worker": "missingness_structural_validity_specialist",
            "summary": {"overview": "Smoke test missingness decisions", "key_patterns": []},
            "column_decisions": [
                {
                    "column": "Q13_Reason",
                    "missingness_disposition": "structurally_valid_missingness",
                    "structural_validity": "confirmed_structural",
                    "recommended_handling": "protect_from_null_penalty",
                    "trigger_columns": ["Q12"],
                    "normalization_notes": "Missingness is gated by Q12.",
                    "reasoning": "Reviewed skip-logic evidence.",
                    "confidence": 0.95,
                    "skip_logic_protected": False,
                    "needs_human_review": False
                },
                {
                    "column": "Q18Main_cell_groupRow1",
                    "missingness_disposition": "structurally_valid_missingness",
                    "structural_validity": "confirmed_structural",
                    "recommended_handling": "protect_from_null_penalty",
                    "trigger_columns": ["Q12"],
                    "normalization_notes": "Q18 family is gated by Q12.",
                    "reasoning": "Reviewed skip-logic evidence for the family.",
                    "confidence": 0.94,
                    "skip_logic_protected": True,
                    "needs_human_review": False
                }
            ],
            "global_contract": {
                "token_missing_placeholders_detected": False,
                "notes": "No dataset-wide token placeholder pattern is supported in the smoke fixture."
            },
            "global_findings": [],
            "review_flags": [],
            "assumptions": []
        },
        "family_worker_json": {
            "family_results": [
                {
                    "family_id": "q_6_main_cell_group",
                    "recommended_table_name": "q6_main_cell_group_rows",
                    "recommended_parent_key": "RespondentId",
                    "recommended_repeat_index_name": "row",
                    "recommended_family_role": "repeated_survey_block",
                    "recommended_handling": "retain_as_child_table",
                    "member_semantics_notes": "Repeated survey block with row-level ordinal responses.",
                    "confidence": 0.93,
                    "reasoning": "Smoke test family result.",
                    "needs_human_review": False
                },
                {
                    "family_id": "q_18_main_cell_group",
                    "recommended_table_name": "q18_main_cell_group_rows",
                    "recommended_parent_key": "RespondentId",
                    "recommended_repeat_index_name": "row",
                    "recommended_family_role": "repeated_survey_block",
                    "recommended_handling": "retain_as_child_table",
                    "member_semantics_notes": "Repeated survey block with row-level Likert responses gated by Q12.",
                    "confidence": 0.92,
                    "reasoning": "Smoke test family result for Q18.",
                    "needs_human_review": False,
                    "member_defaults": {
                        "recommended_logical_type": "ordinal_category",
                        "recommended_storage_type": "string",
                        "transform_actions": ["trim_whitespace", "normalize_category_tokens"],
                        "interpretation_hints": ["skip_logic_protected"],
                        "missingness_disposition": "structurally_valid_missingness",
                        "missingness_handling": "protect_from_null_penalty",
                        "skip_logic_protected": True,
                        "normalization_notes": "Family-level defaults for Q12-gated ordinal responses.",
                        "confidence": 0.88,
                        "needs_human_review": False
                    }
                }
            ]
        },
        "table_layout_worker_json": {
            "worker": "table_layout_specialist",
            "summary": {"overview": "Smoke test layout", "recommended_model_shape": "base_plus_children", "key_layout_principles": ["Base plus repeat child"]},
            "table_suggestions": [
                {
                    "table_name": "base_respondents",
                    "kind": "base_entity",
                    "source_basis": {"kind": "primary_grain"},
                    "parent_table_name": "",
                    "parent_key": [],
                    "grain_columns": ["RespondentId"],
                    "repeat_index_name": "",
                    "build_strategy": "direct_select",
                    "included_columns": ["RespondentId", "StatusCode", "Q12", "Q13_Reason", "FreeTextComment", "Q1FreeText"],
                    "source_family_id": "",
                    "included_column_count": 6,
                    "included_columns_preview": ["RespondentId", "StatusCode", "Q12"],
                    "excluded_columns": ["Unnamed: 0"],
                    "confidence": 0.98,
                    "reasoning": "Smoke test base table.",
                    "needs_human_review": False
                },
                {
                    "table_name": "q6_main_cell_group_rows",
                    "kind": "child_repeat",
                    "source_basis": {"kind": "accepted_family"},
                    "parent_table_name": "base_respondents",
                    "parent_key": ["RespondentId"],
                    "grain_columns": ["RespondentId", "row"],
                    "repeat_index_name": "row",
                    "build_strategy": "wide_to_long_family",
                    "source_family_id": "q_6_main_cell_group",
                    "included_column_count": 1,
                    "included_columns_preview": ["Q6Main_cell_groupRow1"],
                    "excluded_columns": [],
                    "confidence": 0.92,
                    "reasoning": "Smoke test family table.",
                    "needs_human_review": False
                },
                {
                    "table_name": "q18_main_cell_group_rows",
                    "kind": "child_repeat",
                    "source_basis": {"kind": "accepted_family"},
                    "parent_table_name": "base_respondents",
                    "parent_key": ["RespondentId"],
                    "grain_columns": ["RespondentId", "row"],
                    "repeat_index_name": "row",
                    "build_strategy": "wide_to_long_family",
                    "source_family_id": "q_18_main_cell_group",
                    "included_column_count": 2,
                    "included_columns_preview": ["Q18Main_cell_groupRow1", "Q18Main_cell_groupRow2"],
                    "excluded_columns": [],
                    "confidence": 0.91,
                    "reasoning": "Smoke test family table for Q18.",
                    "needs_human_review": False
                }
            ],
            "column_table_assignments": [
                {"column": "Unnamed: 0", "assigned_table": "", "assignment_role": "exclude_from_outputs", "source_family_id": "", "why": "Export-only index"},
                {"column": "RespondentId", "assigned_table": "base_respondents", "assignment_role": "base_key", "source_family_id": "", "why": "Primary grain"},
                {"column": "StatusCode", "assigned_table": "base_respondents", "assignment_role": "base_attribute", "source_family_id": "", "why": "Status code"},
                {"column": "Q12", "assigned_table": "base_respondents", "assignment_role": "base_attribute", "source_family_id": "", "why": "Gate variable"},
                {"column": "Q13_Reason", "assigned_table": "base_respondents", "assignment_role": "base_attribute", "source_family_id": "", "why": "Conditioned free text"},
                {"column": "FreeTextComment", "assigned_table": "base_respondents", "assignment_role": "base_attribute", "source_family_id": "", "why": "Free text comment"},
                {"column": "Q1FreeText", "assigned_table": "base_respondents", "assignment_role": "base_attribute", "source_family_id": "", "why": "Open response"},
                {"column": "Q6Main_cell_groupRow1", "assigned_table": "q6_main_cell_group_rows", "assignment_role": "melt_member", "source_family_id": "q_6_main_cell_group", "why": "Family member"},
                {"column": "Q18Main_cell_groupRow1", "assigned_table": "q18_main_cell_group_rows", "assignment_role": "melt_member", "source_family_id": "q_18_main_cell_group", "why": "Family member"},
                {"column": "Q18Main_cell_groupRow2", "assigned_table": "q18_main_cell_group_rows", "assignment_role": "melt_member", "source_family_id": "q_18_main_cell_group", "why": "Family member"}
            ],
            "global_layout_findings": [],
            "review_flags": [],
            "assumptions": []
        }
    }


def _row_by_column(rows, column):
    return next(row for row in rows if row["column"] == column)


def main():
    support = _support()
    inputs = _inputs()
    result = _synthesize_canonical_column_contract(
        run_id="run_smoke",
        light_contract_decisions=inputs["light_contract_decisions"],
        semantic_context_json=inputs["semantic_context_json"],
        type_transform_worker_json=inputs["type_transform_worker_json"],
        missingness_worker_json=inputs["missingness_worker_json"],
        family_worker_json=inputs["family_worker_json"],
        table_layout_worker_json=inputs["table_layout_worker_json"],
        support=support,
    )

    app_validator = _load_validator_main()
    standalone_validator = _load_standalone_validator_main()

    validation = app_validator(
        json.dumps(result),
        expected_source_columns_json=json.dumps(support["a2_order"]),
    )
    if validation["validation_ok"] != "true":
        raise AssertionError(f"Validator failed: {validation}")
    standalone_validation = standalone_validator(
        json.dumps(result),
        json.dumps(support["a2_order"]),
        json.dumps(inputs["missingness_worker_json"]),
    )
    if standalone_validation["validation_ok"] != "true":
        raise AssertionError(f"Standalone validator failed: {standalone_validation}")

    skipped_result = _synthesize_canonical_column_contract(
        run_id="run_smoke",
        light_contract_decisions=inputs["light_contract_decisions"],
        semantic_context_json={"status": "skipped", "reason": "light_contract_accepted"},
        type_transform_worker_json=inputs["type_transform_worker_json"],
        missingness_worker_json=inputs["missingness_worker_json"],
        family_worker_json=inputs["family_worker_json"],
        table_layout_worker_json=inputs["table_layout_worker_json"],
        support=support,
    )
    skipped_validation = app_validator(
        json.dumps(skipped_result),
        expected_source_columns_json=json.dumps(support["a2_order"]),
    )
    if skipped_validation["validation_ok"] != "true":
        raise AssertionError(f"Skipped semantic validator failed: {skipped_validation}")

    respondent = _row_by_column(result["column_contracts"], "RespondentId")
    assert respondent["type_decision_source"] == "reviewed_type_worker"
    assert respondent["recommended_logical_type"] == "identifier"
    assert "type_transform_worker" in respondent["applied_sources"]

    status_code = _row_by_column(result["column_contracts"], "StatusCode")
    assert status_code["recommended_logical_type"] == "categorical_code"
    assert status_code["semantic_decision_source"] == "semantic_context_worker"
    assert status_code["codebook_note"] == "1 = active, 2 = paused"

    unnamed = _row_by_column(result["column_contracts"], "Unnamed: 0")
    assert unnamed["canonical_modeling_status"] == "excluded_from_outputs"
    assert unnamed["canonical_table_name"] == ""

    child = _row_by_column(result["column_contracts"], "Q6Main_cell_groupRow1")
    assert child["canonical_modeling_status"] == "child_repeat_member"
    assert child["type_decision_source"] == "reviewed_type_worker"
    assert "requires_child_table_review" not in child["structural_transform_hints"]

    invalid_child_hints = json.loads(json.dumps(result))
    invalid_child_row = _row_by_column(invalid_child_hints["column_contracts"], "Q6Main_cell_groupRow1")
    invalid_child_row["structural_transform_hints"] = [
        "requires_child_table_review",
        "requires_wide_to_long_review",
    ]
    invalid_child_hint_errors = _validate_canonical_column_contract_output(
        invalid_child_hints,
        expected_source_columns=support["a2_order"],
    )
    assert any(
        "structural_transform_hints must not contain requires_child_table_review when canonical_modeling_status is child_repeat_member"
        in err
        for err in invalid_child_hint_errors
    )
    assert any(
        "structural_transform_hints must not contain requires_wide_to_long_review when canonical_modeling_status is child_repeat_member"
        in err
        for err in invalid_child_hint_errors
    )

    protected = _row_by_column(result["column_contracts"], "Q13_Reason")
    assert protected["skip_logic_protected"] is True
    assert "skip_logic_protected" in protected["interpretation_hints"]
    assert protected["missingness_decision_source"] == "reviewed_missingness_worker"
    assert "missingness_worker" in protected["applied_sources"]

    baseline = _row_by_column(result["column_contracts"], "FreeTextComment")
    assert baseline["type_decision_source"] == "a17_baseline"
    assert baseline["semantic_meaning"] == ""
    assert baseline["semantic_decision_source"] == "unknown"
    assert "A17" in baseline["applied_sources"]

    q1_free_text = _row_by_column(result["column_contracts"], "Q1FreeText")
    assert q1_free_text["type_decision_source"] == "a17_baseline"
    assert q1_free_text["recommended_logical_type"] == "free_text"
    assert q1_free_text["recommended_storage_type"] == "string"

    q18_row1 = _row_by_column(result["column_contracts"], "Q18Main_cell_groupRow1")
    assert q18_row1["type_decision_source"] == "reviewed_type_worker"
    assert q18_row1["recommended_logical_type"] == "ordinal_category"
    assert q18_row1["missingness_decision_source"] == "reviewed_missingness_worker"
    assert q18_row1["skip_logic_protected"] is True

    q18_row2 = _row_by_column(result["column_contracts"], "Q18Main_cell_groupRow2")
    assert q18_row2["recommended_logical_type"] == "ordinal_category"
    assert q18_row2["type_decision_source"] == "family_default"
    assert q18_row2["missingness_decision_source"] == "family_default"
    assert q18_row2["skip_logic_protected"] is True
    assert "family_worker.member_defaults" in q18_row2["applied_sources"]

    ghost = _row_by_column(result["column_contracts"], "GhostCol")
    assert ghost["canonical_modeling_status"] == "unresolved"
    assert ghost["type_decision_source"] == "reviewed_type_worker"
    assert ghost["missingness_decision_source"] == "unresolved_no_a2_evidence"
    assert any(flag["item"] == "GhostCol" and flag["issue"] == "reviewed_column_not_in_a2" for flag in result["review_flags"])
    assert any(flag["issue"] == "high_deterministic_baseline_share" for flag in result["review_flags"])

    assert result["summary"]["reviewed_override_count"] == 5
    assert result["summary"]["family_default_count"] == 1
    assert result["summary"]["deterministic_baseline_count"] == 5

    invalid_skip = json.loads(json.dumps(result))
    invalid_skip_row = _row_by_column(invalid_skip["column_contracts"], "Q13_Reason")
    invalid_skip_row["skip_logic_protected"] = False
    invalid_skip_errors = _validate_canonical_column_contract_output(
        invalid_skip,
        expected_source_columns=support["a2_order"],
    )
    assert any(
        "skip_logic_protected must be true when missingness_handling is protect_from_null_penalty" in err
        for err in invalid_skip_errors
    )

    invalid_disposition = json.loads(json.dumps(result))
    invalid_disposition_row = _row_by_column(invalid_disposition["column_contracts"], "Q13_Reason")
    invalid_disposition_row["missingness_disposition"] = "no_material_missingness"
    invalid_disposition_errors = _validate_canonical_column_contract_output(
        invalid_disposition,
        expected_source_columns=support["a2_order"],
    )
    assert any(
        "missingness_handling must be one of ['no_action_needed'] when missingness_disposition is no_material_missingness" in err
        for err in invalid_disposition_errors
    )

    family_default_leak_inputs = json.loads(json.dumps(inputs))
    family_default_leak_inputs["family_worker_json"]["family_results"][1]["member_defaults"]["missingness_disposition"] = "token_missingness_present"
    family_default_leak_inputs["family_worker_json"]["family_results"][1]["member_defaults"]["missingness_handling"] = "retain_with_caution"
    family_default_leak_inputs["family_worker_json"]["family_results"][1]["member_defaults"]["skip_logic_protected"] = False
    family_default_leak_result = _synthesize_canonical_column_contract(
        run_id="run_smoke_family_default_leak",
        light_contract_decisions=family_default_leak_inputs["light_contract_decisions"],
        semantic_context_json=family_default_leak_inputs["semantic_context_json"],
        type_transform_worker_json=family_default_leak_inputs["type_transform_worker_json"],
        missingness_worker_json=family_default_leak_inputs["missingness_worker_json"],
        family_worker_json=family_default_leak_inputs["family_worker_json"],
        table_layout_worker_json=family_default_leak_inputs["table_layout_worker_json"],
        support=support,
    )
    q18_family_leak_row = _row_by_column(family_default_leak_result["column_contracts"], "Q18Main_cell_groupRow2")
    assert q18_family_leak_row["missingness_decision_source"] != "family_default"
    assert q18_family_leak_row["missingness_disposition"] != "token_missingness_present"

    parity_cases = [
        (
            "canonical_contract_invalid_missingness_pair.json",
            "missingness_handling must be one of ['no_action_needed'] when missingness_disposition is no_material_missingness",
        ),
        (
            "canonical_contract_invalid_family_default_non_structural.json",
            "missingness_decision_source may not be family_default for non-structural missingness_disposition token_missingness_present",
        ),
        (
            "canonical_contract_invalid_child_hint.json",
            "structural_transform_hints must not contain requires_child_table_review when canonical_modeling_status is child_repeat_member",
        ),
    ]
    for fixture_name, expected_substring in parity_cases:
        fixture = _load_fixture_json(fixture_name)
        expected_columns = json.dumps([row["column"] for row in fixture["column_contracts"]])
        app_errors = _validate_canonical_column_contract_output(
            fixture,
            expected_source_columns=[row["column"] for row in fixture["column_contracts"]],
        )
        standalone_result = standalone_validator(json.dumps(fixture), expected_columns)
        assert any(expected_substring in err for err in app_errors), (fixture_name, app_errors)
        assert standalone_result["validation_ok"] == "false", standalone_result
        standalone_errors = json.loads(standalone_result["validation_errors_json"])
        assert any(expected_substring in err for err in standalone_errors), (fixture_name, standalone_errors)

    skipped_anchor = _row_by_column(skipped_result["column_contracts"], "FreeTextComment")
    assert skipped_anchor["semantic_meaning"] == ""
    assert skipped_anchor["codebook_note"] == ""
    assert skipped_anchor["semantic_decision_source"] == "unknown"
    assert any(item["assumption"] == "semantic_context_skipped" for item in skipped_result["assumptions"])

    print("PASS: canonical column contract smoke checks")


if __name__ == "__main__":
    main()
