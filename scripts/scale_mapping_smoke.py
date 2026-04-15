#!/usr/bin/env python3
import sys
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    impersonated_mod.Credentials = object
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

    class Timestamp:
        @staticmethod
        def utcnow():
            class _Now:
                def isoformat(self):
                    return "2026-04-14T00:00:00Z"

            return _Now()

    class Series:
        pass

    class DataFrame:
        pass

    pandas_mod.Timestamp = Timestamp
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

    class UploadFile:
        pass

    def File(*args, **kwargs):
        return None

    def Form(*args, **kwargs):
        return None

    def Depends(*args, **kwargs):
        return None

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:
        headers = {}
        client = types.SimpleNamespace(host="127.0.0.1")

    class HTTPBearer:
        def __init__(self, *args, **kwargs):
            pass

    class HTTPAuthorizationCredentials:
        credentials = ""

    class JSONResponse:
        def __init__(self, content=None, headers=None):
            self.content = content
            self.headers = headers or {}

    class Response:
        headers = {}

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

    def Field(*args, **kwargs):
        return kwargs.get("default")

    pydantic_mod.BaseModel = BaseModel
    pydantic_mod.Field = Field
    sys.modules["pydantic"] = pydantic_mod

from xlsx_export import build_light_contract_xlsx_bytes, parse_light_contract_xlsx_bytes
import app


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def test_light_contract_scale_mapping_sheet():
    payload = {
        "run_id": "run_smoke",
        "generated_at": "2026-04-14T00:00:00Z",
        "source_endpoint": "/light-contracts/xlsx",
        "column_guide_rows": [{"column_index": 1, "column_name": "Q9Row1", "family_id": "q_9_main_cell_group", "notes": ""}],
        "grain_summary_rows": [],
        "primary_grain_rows": [{"item": "base", "recommended_key_1": "RespondentId", "recommended_key_2": "", "recommended_key_3": "", "your_key_1": "", "your_key_2": "", "your_key_3": "", "status": "accept", "comments": ""}],
        "reference_rows": [],
        "repeat_family_rows": [],
        "structural_gate_rows": [],
        "scale_mapping_rows": [
            {
                "target_kind": "family",
                "target_id": "q_9_main_cell_group",
                "response_scale_kind": "familiarity_scale",
                "ordered_labels_low_to_high": "Never Heard of It 0|Somewhat Familiar 3|Very Familiar 6",
                "numeric_scores_low_to_high": "",
                "notes": "Confirmed from user notes.",
            }
        ],
        "override_rows": [],
    }
    xlsx_bytes = build_light_contract_xlsx_bytes(payload)
    parsed = parse_light_contract_xlsx_bytes(xlsx_bytes)
    handoff = app._build_parsed_light_contract_handoff("run_smoke", parsed)
    scale_input = handoff.get("scale_mapping_input") or []
    _assert(len(scale_input) == 1, "expected one normalized scale mapping input row")
    _assert(scale_input[0]["target_kind"] == "family", "expected family target kind")
    _assert(scale_input[0]["ordered_labels"][0] == "Never Heard of It 0", "expected ordered labels parsed from sheet")


def test_scale_mapping_resolver_and_canonical_integration():
    support = {
        "a2_rows": [
            {
                "column": "Q9Row1",
                "top_levels": ["Never Heard of It 0", "Somewhat Familiar 3", "Very Familiar 6"],
                "a2_samples": {"random": []},
                "unique_count": 3,
                "top_candidate": {"type": "categorical"},
            },
            {
                "column": "OverallSatisfaction",
                "top_levels": ["Strongly disagree 1", "Disagree 2", "Neutral 3", "Agree 4", "Strongly agree 5"],
                "a2_samples": {"random": []},
                "unique_count": 5,
                "top_candidate": {"type": "categorical"},
            },
        ],
        "a2_by_col": {},
        "a2_order": ["Q9Row1", "OverallSatisfaction"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {
            "Q9Row1": {
                "column": "Q9Row1",
                "a9_primary_role": "measure_item",
                "recommended_logical_type": "numeric_measure",
                "recommended_storage_type": "integer",
                "transform_actions": ["trim_whitespace", "strip_numeric_formatting", "cast_to_integer"],
                "structural_transform_hints": ["requires_codebook_or_label_mapping_review"],
                "interpretation_hints": [],
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "type_normalization_notes": "validate numeric-label mapping before casting to numeric",
                "missingness_normalization_notes": "",
                "quality_score": 0.9,
                "drift_detected": False,
                "type_decision_source": "a17_baseline",
                "missingness_decision_source": "a17_baseline",
                "type_confidence": 0.61,
                "missingness_confidence": 0.85,
                "type_review_required": False,
                "missingness_review_required": False,
                "confidence": 0.71,
                "applied_sources": ["A17"],
            },
            "OverallSatisfaction": {
                "column": "OverallSatisfaction",
                "a9_primary_role": "measure_item",
                "recommended_logical_type": "numeric_measure",
                "recommended_storage_type": "integer",
                "transform_actions": ["trim_whitespace", "strip_numeric_formatting", "cast_to_integer"],
                "structural_transform_hints": ["requires_codebook_or_label_mapping_review"],
                "interpretation_hints": [],
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "type_normalization_notes": "validate numeric-label mapping before casting to numeric",
                "missingness_normalization_notes": "",
                "quality_score": 0.9,
                "drift_detected": False,
                "type_decision_source": "a17_baseline",
                "missingness_decision_source": "a17_baseline",
                "type_confidence": 0.61,
                "missingness_confidence": 0.85,
                "type_review_required": False,
                "missingness_review_required": False,
                "confidence": 0.71,
                "applied_sources": ["A17"],
            },
        },
        "family_by_column": {"Q9Row1": "q_9_main_cell_group"},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    support["a2_by_col"] = {row["column"]: row for row in support["a2_rows"]}

    app._load_canonical_support_artifacts = lambda run_id: support

    light_contract_decisions = {
        "run_id": "run_smoke",
        "light_contract_status": "accepted",
        "primary_grain_decision": {"status": "accept", "keys": ["RespondentId"], "comments": ""},
        "reference_decisions": [],
        "dimension_decisions": [],
        "family_decisions": [
            {
                "family_id": "q_9_main_cell_group",
                "status": "accept",
                "table_name": "q9_long",
                "repeat_index_name": "row",
                "parent_key": "RespondentId",
                "comments": "",
            }
        ],
        "override_notes": {},
        "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
        "scale_mapping_input": [
            {
                "target_kind": "family",
                "target_id": "q_9_main_cell_group",
                "response_scale_kind": "familiarity_scale",
                "ordered_labels": ["Never Heard of It 0", "Somewhat Familiar 3", "Very Familiar 6"],
                "numeric_scores": [],
                "notes": "Confirmed family-level familiarity ladder.",
            }
        ],
    }
    family_worker_json = {
        "family_results": [
            {
                "family_result": {
                    "family_id": "q_9_main_cell_group",
                    "recommended_family_role": "repeated_survey_block",
                    "member_semantics_notes": "15-item familiarity rating matrix",
                }
            }
        ]
    }

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        family_worker_json=family_worker_json,
        scale_mapping_extractor_json={},
    )
    by_target = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}
    _assert(by_target[("family", "q_9_main_cell_group")]["mapping_status"] == "human_confirmed", "expected human-confirmed family mapping")
    _assert(by_target[("column", "OverallSatisfaction")]["mapping_status"] == "deterministic_inferred", "expected deterministic standalone mapping")

    canon = app._synthesize_canonical_column_contract(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        semantic_context_json={"status": "skipped", "reason": "light_contract_accepted"},
        type_transform_worker_json={"column_decisions": [], "global_transform_rules": []},
        missingness_worker_json={"column_decisions": [], "global_contract": {"token_missing_placeholders_detected": False, "notes": ""}, "global_findings": []},
        family_worker_json=family_worker_json,
        table_layout_worker_json={
            "column_table_assignments": [
                {"column": "Q9Row1", "assigned_table": "q9_long", "assignment_role": "melt_member", "source_family_id": "q_9_main_cell_group"},
                {"column": "OverallSatisfaction", "assigned_table": "base_table", "assignment_role": "base_attribute", "source_family_id": ""},
            ],
            "table_suggestions": [
                {"table_name": "q9_long", "kind": "child_repeat"},
                {"table_name": "base_table", "kind": "base_table"},
            ],
        },
        scale_mapping_json=scale_mapping_json,
        support=support,
    )

    rows = {row["column"]: row for row in canon["column_contracts"]}
    q9 = rows["Q9Row1"]
    _assert(q9["recommended_logical_type"] == "ordinal_category", "expected ordinal category after scale mapping")
    _assert(q9["recommended_storage_type"] == "string", "expected string storage after scale mapping")
    _assert(q9["type_decision_source"] == "scale_mapping_resolver", "expected scale mapping to own the type refinement")
    _assert("requires_codebook_or_label_mapping_review" not in q9["structural_transform_hints"], "expected confirmed mapping to clear codebook review hint")
    _assert("Confirmed scale mapping" in q9["codebook_note"], "expected confirmed codebook note")


if __name__ == "__main__":
    test_light_contract_scale_mapping_sheet()
    test_scale_mapping_resolver_and_canonical_integration()
    print("scale mapping smoke passed")
