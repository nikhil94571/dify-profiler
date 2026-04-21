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


def _baseline_row(
    column,
    *,
    role,
    logical_type,
    storage_type,
    transform_actions,
    structural_transform_hints=None,
    interpretation_hints=None,
    normalization_notes="",
):
    return {
        "column": column,
        "a9_primary_role": role,
        "recommended_logical_type": logical_type,
        "recommended_storage_type": storage_type,
        "transform_actions": list(transform_actions),
        "structural_transform_hints": list(structural_transform_hints or []),
        "interpretation_hints": list(interpretation_hints or []),
        "missingness_disposition": "no_material_missingness",
        "missingness_handling": "no_action_needed",
        "skip_logic_protected": False,
        "type_normalization_notes": normalization_notes,
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
    }


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
    overall = rows["OverallSatisfaction"]
    _assert(q9["recommended_logical_type"] == "ordinal_category", "expected ordinal category after scale mapping")
    _assert(q9["recommended_storage_type"] == "string", "expected string storage after scale mapping")
    _assert(q9["type_decision_source"] == "scale_mapping_resolver", "expected scale mapping to own the type refinement")
    _assert("requires_codebook_or_label_mapping_review" not in q9["structural_transform_hints"], "expected confirmed mapping to clear codebook review hint")
    _assert("Confirmed scale mapping" in q9["codebook_note"], "expected confirmed codebook note")
    _assert(overall["recommended_logical_type"] == "numeric_measure", "expected deterministic standalone mapping to stay metadata-only in canon")
    _assert(overall["recommended_storage_type"] == "integer", "expected deterministic standalone mapping to preserve baseline storage")
    _assert(overall["type_decision_source"] == "a17_baseline", "expected canon to ignore deterministic standalone mappings for type refinement")


def test_scale_mapping_resolver_skips_numeric_standalone_columns():
    support = {
        "a2_rows": [
            {"column": "ID", "top_levels": ["1", "2", "3", "4"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "categorical"}},
            {"column": "Age", "top_levels": ["18", "19", "20", "21"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "numeric"}},
            {"column": "GPA", "top_levels": ["2.7", "3.0", "3.3", "4.0"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "numeric"}},
            {"column": "Final_Grade", "top_levels": ["60", "70", "80", "90"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "numeric"}},
            {"column": "ANXATT", "top_levels": ["1", "2", "3"], "a2_samples": {"random": []}, "unique_count": 3, "top_candidate": {"type": "categorical"}},
            {"column": "MATHATT", "top_levels": ["1", "2", "3"], "a2_samples": {"random": []}, "unique_count": 3, "top_candidate": {"type": "categorical"}},
            {"column": "OverallSatisfaction", "top_levels": ["Strongly disagree 1", "Disagree 2", "Neutral 3", "Agree 4", "Strongly agree 5"], "a2_samples": {"random": []}, "unique_count": 5, "top_candidate": {"type": "categorical"}},
        ],
        "a2_by_col": {},
        "a2_order": ["ID", "Age", "GPA", "Final_Grade", "ANXATT", "MATHATT", "OverallSatisfaction"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {
            "ID": _baseline_row("ID", role="id_key", logical_type="identifier", storage_type="string", transform_actions=["trim_whitespace", "cast_to_string"], interpretation_hints=["identifier_not_measure"]),
            "Age": _baseline_row("Age", role="measure", logical_type="numeric_measure", storage_type="decimal", transform_actions=["trim_whitespace", "cast_to_decimal"]),
            "GPA": _baseline_row("GPA", role="measure", logical_type="numeric_measure", storage_type="decimal", transform_actions=["trim_whitespace", "cast_to_decimal"]),
            "Final_Grade": _baseline_row("Final_Grade", role="measure", logical_type="numeric_measure", storage_type="decimal", transform_actions=["trim_whitespace", "cast_to_decimal"]),
            "ANXATT": _baseline_row("ANXATT", role="measure", logical_type="categorical_code", storage_type="string", transform_actions=["trim_whitespace", "cast_to_string"]),
            "MATHATT": _baseline_row("MATHATT", role="measure", logical_type="categorical_code", storage_type="string", transform_actions=["trim_whitespace", "cast_to_string"]),
            "OverallSatisfaction": _baseline_row("OverallSatisfaction", role="measure_item", logical_type="numeric_measure", storage_type="integer", transform_actions=["trim_whitespace", "strip_numeric_formatting", "cast_to_integer"]),
        },
        "family_by_column": {},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    support["a2_by_col"] = {row["column"]: row for row in support["a2_rows"]}
    app._load_canonical_support_artifacts = lambda run_id: support

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions={
            "run_id": "run_smoke",
            "light_contract_status": "accepted",
            "primary_grain_decision": {"status": "accept", "keys": ["ID"], "comments": ""},
            "reference_decisions": [],
            "dimension_decisions": [],
            "family_decisions": [],
            "override_notes": {},
            "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
            "scale_mapping_input": [],
        },
        family_worker_json={},
        scale_mapping_extractor_json={},
    )
    by_target = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}
    for column in ["ID", "Age", "GPA", "Final_Grade", "ANXATT", "MATHATT"]:
        _assert(("column", column) not in by_target, f"expected resolver to skip numeric standalone scale inference for {column}")
    _assert(by_target[("column", "OverallSatisfaction")]["mapping_status"] == "deterministic_inferred", "expected textual ladder standalone mapping to remain available as resolver metadata")


def test_scale_mapping_resolver_drops_placeholder_human_family_rows():
    support = {
        "a2_rows": [
            {"column": "A1Row1", "top_levels": ["A", "B"], "a2_samples": {"random": []}, "unique_count": 2, "top_candidate": {"type": "categorical"}},
            {"column": "A2Row1", "top_levels": ["A", "B"], "a2_samples": {"random": []}, "unique_count": 2, "top_candidate": {"type": "categorical"}},
            {"column": "M1_Q1", "top_levels": ["A", "B"], "a2_samples": {"random": []}, "unique_count": 2, "top_candidate": {"type": "categorical"}},
            {"column": "M2_Q1", "top_levels": ["A", "B"], "a2_samples": {"random": []}, "unique_count": 2, "top_candidate": {"type": "categorical"}},
            {"column": "Q1", "top_levels": ["A", "B"], "a2_samples": {"random": []}, "unique_count": 2, "top_candidate": {"type": "categorical"}},
        ],
        "a2_by_col": {},
        "a2_order": ["A1Row1", "A2Row1", "M1_Q1", "M2_Q1", "Q1"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {
            "A1Row1": "a_1",
            "A2Row1": "a_2",
            "M1_Q1": "m_1",
            "M2_Q1": "m_2",
            "Q1": "q",
        },
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
            {"family_id": "a_1", "status": "accept", "table_name": "a1_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
            {"family_id": "a_2", "status": "accept", "table_name": "a2_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
            {"family_id": "m_1", "status": "accept", "table_name": "m1_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
            {"family_id": "m_2", "status": "accept", "table_name": "m2_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
            {"family_id": "q", "status": "accept", "table_name": "q_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
        ],
        "override_notes": {},
        "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
        "scale_mapping_input": [
            {"target_kind": "family", "target_id": "a_1"},
            {"target_kind": "family", "target_id": "a_2"},
            {"target_kind": "family", "target_id": "m_1"},
            {"target_kind": "family", "target_id": "m_2"},
            {"target_kind": "family", "target_id": "q"},
        ],
    }
    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        family_worker_json={},
        scale_mapping_extractor_json={},
    )
    by_target = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}
    for family_id in ["a_1", "a_2", "m_1", "m_2", "q"]:
        _assert(("family", family_id) not in by_target, f"expected placeholder light-contract row for {family_id} to be dropped instead of confirmed")


def test_canonical_scale_mapping_applies_only_confirmed_label_complete_mappings():
    support = {
        "a2_rows": [
            {"column": "A1Row1", "top_levels": ["1", "2", "3", "4", "5"], "a2_samples": {"random": []}, "unique_count": 5, "top_candidate": {"type": "categorical"}},
            {"column": "A2Row1", "top_levels": ["1", "2", "3", "4", "5"], "a2_samples": {"random": []}, "unique_count": 5, "top_candidate": {"type": "categorical"}},
            {"column": "M1_Q1", "top_levels": ["A", "B", "C", "D"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "categorical"}},
            {"column": "M2_Q1", "top_levels": ["A", "B", "C", "D"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "categorical"}},
            {"column": "Q1", "top_levels": ["A", "B", "C", "D"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "categorical"}},
            {"column": "ID", "top_levels": ["1", "2", "3", "4"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "categorical"}},
            {"column": "Age", "top_levels": ["18", "19", "20", "21"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "numeric"}},
            {"column": "GPA", "top_levels": ["2.7", "3.0", "3.3", "4.0"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "numeric"}},
            {"column": "Final_Grade", "top_levels": ["60", "70", "80", "90"], "a2_samples": {"random": []}, "unique_count": 4, "top_candidate": {"type": "numeric"}},
            {"column": "ANXATT", "top_levels": ["1", "2", "3"], "a2_samples": {"random": []}, "unique_count": 3, "top_candidate": {"type": "categorical"}},
            {"column": "MATHATT", "top_levels": ["1", "2", "3"], "a2_samples": {"random": []}, "unique_count": 3, "top_candidate": {"type": "categorical"}},
        ],
        "a2_by_col": {},
        "a2_order": ["A1Row1", "A2Row1", "M1_Q1", "M2_Q1", "Q1", "ID", "Age", "GPA", "Final_Grade", "ANXATT", "MATHATT"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {
            "A1Row1": _baseline_row("A1Row1", role="measure_item", logical_type="numeric_measure", storage_type="integer", transform_actions=["trim_whitespace", "strip_numeric_formatting", "cast_to_integer"], structural_transform_hints=["requires_codebook_or_label_mapping_review"], normalization_notes="validate numeric-label mapping before casting to numeric"),
            "A2Row1": _baseline_row("A2Row1", role="measure_item", logical_type="numeric_measure", storage_type="integer", transform_actions=["trim_whitespace", "strip_numeric_formatting", "cast_to_integer"], structural_transform_hints=["requires_codebook_or_label_mapping_review"], normalization_notes="validate numeric-label mapping before casting to numeric"),
            "M1_Q1": _baseline_row("M1_Q1", role="measure_item", logical_type="nominal_category", storage_type="string", transform_actions=["trim_whitespace", "uppercase_values"]),
            "M2_Q1": _baseline_row("M2_Q1", role="measure_item", logical_type="nominal_category", storage_type="string", transform_actions=["trim_whitespace", "uppercase_values"]),
            "Q1": _baseline_row("Q1", role="invariant_attr", logical_type="nominal_category", storage_type="string", transform_actions=["trim_whitespace", "uppercase_values"]),
            "ID": _baseline_row("ID", role="id_key", logical_type="identifier", storage_type="string", transform_actions=["trim_whitespace", "cast_to_string"], interpretation_hints=["identifier_not_measure"]),
            "Age": _baseline_row("Age", role="measure", logical_type="numeric_measure", storage_type="decimal", transform_actions=["trim_whitespace", "cast_to_decimal"]),
            "GPA": _baseline_row("GPA", role="measure", logical_type="numeric_measure", storage_type="decimal", transform_actions=["trim_whitespace", "cast_to_decimal"]),
            "Final_Grade": _baseline_row("Final_Grade", role="measure", logical_type="numeric_measure", storage_type="decimal", transform_actions=["trim_whitespace", "cast_to_decimal"]),
            "ANXATT": _baseline_row("ANXATT", role="measure", logical_type="categorical_code", storage_type="string", transform_actions=["trim_whitespace", "cast_to_string"]),
            "MATHATT": _baseline_row("MATHATT", role="measure", logical_type="categorical_code", storage_type="string", transform_actions=["trim_whitespace", "cast_to_string"]),
        },
        "family_by_column": {
            "A1Row1": "a_1",
            "A2Row1": "a_2",
            "M1_Q1": "m_1",
            "M2_Q1": "m_2",
            "Q1": "q",
        },
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    support["a2_by_col"] = {row["column"]: row for row in support["a2_rows"]}
    app._load_canonical_support_artifacts = lambda run_id: support

    light_contract_decisions = {
        "run_id": "run_smoke",
        "light_contract_status": "accepted",
        "primary_grain_decision": {"status": "accept", "keys": ["ID"], "comments": ""},
        "reference_decisions": [],
        "dimension_decisions": [],
        "family_decisions": [
            {"family_id": "a_1", "status": "accept", "table_name": "a1_long", "repeat_index_name": "row", "parent_key": "ID", "comments": ""},
            {"family_id": "a_2", "status": "accept", "table_name": "a2_long", "repeat_index_name": "row", "parent_key": "ID", "comments": ""},
            {"family_id": "m_1", "status": "accept", "table_name": "m1_long", "repeat_index_name": "row", "parent_key": "ID", "comments": ""},
            {"family_id": "m_2", "status": "accept", "table_name": "m2_long", "repeat_index_name": "row", "parent_key": "ID", "comments": ""},
            {"family_id": "q", "status": "accept", "table_name": "q_long", "repeat_index_name": "row", "parent_key": "ID", "comments": ""},
        ],
        "override_notes": {},
        "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
        "scale_mapping_input": [
            {"target_kind": "family", "target_id": "a_1", "response_scale_kind": "ordinal_scale", "ordered_labels": ["1", "2", "3", "4", "5"], "numeric_scores": [1, 2, 3, 4, 5], "notes": "Confirmed A1 ladder."},
            {"target_kind": "family", "target_id": "a_2", "response_scale_kind": "ordinal_scale", "ordered_labels": ["1", "2", "3", "4", "5"], "numeric_scores": [1, 2, 3, 4, 5], "notes": "Confirmed A2 ladder."},
            {"target_kind": "family", "target_id": "m_1"},
            {"target_kind": "family", "target_id": "m_2"},
            {"target_kind": "family", "target_id": "q"},
        ],
    }
    family_worker_json = {
        "family_results": [
            {"family_result": {"family_id": "a_1", "recommended_family_role": "repeated_survey_block", "member_semantics_notes": "Start-of-term anxiety ladder"}},
            {"family_result": {"family_id": "a_2", "recommended_family_role": "repeated_survey_block", "member_semantics_notes": "End-of-term anxiety ladder"}},
            {"family_result": {"family_id": "m_1", "recommended_family_role": "repeated_assessment", "member_semantics_notes": "Math item responses"}},
            {"family_result": {"family_id": "m_2", "recommended_family_role": "repeated_assessment", "member_semantics_notes": "Math item responses"}},
            {"family_result": {"family_id": "q", "recommended_family_role": "reference_lookup", "member_semantics_notes": "Answer key codes"}},
        ]
    }

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        family_worker_json=family_worker_json,
        scale_mapping_extractor_json={},
    )
    by_target = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}
    _assert(by_target[("family", "a_1")]["mapping_status"] == "human_confirmed", "expected A1 confirmed family mapping")
    _assert(by_target[("family", "a_2")]["mapping_status"] == "human_confirmed", "expected A2 confirmed family mapping")
    for family_id in ["m_1", "m_2", "q"]:
        _assert(("family", family_id) not in by_target, f"expected placeholder row for {family_id} to be dropped before canon")

    canon = app._synthesize_canonical_column_contract(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        semantic_context_json={"status": "skipped", "reason": "light_contract_accepted"},
        type_transform_worker_json={"column_decisions": [], "global_transform_rules": []},
        missingness_worker_json={"column_decisions": [], "global_contract": {"token_missing_placeholders_detected": False, "notes": ""}, "global_findings": []},
        family_worker_json=family_worker_json,
        table_layout_worker_json={
            "column_table_assignments": [
                {"column": "A1Row1", "assigned_table": "a1_long", "assignment_role": "melt_member", "source_family_id": "a_1"},
                {"column": "A2Row1", "assigned_table": "a2_long", "assignment_role": "melt_member", "source_family_id": "a_2"},
                {"column": "M1_Q1", "assigned_table": "m1_long", "assignment_role": "melt_member", "source_family_id": "m_1"},
                {"column": "M2_Q1", "assigned_table": "m2_long", "assignment_role": "melt_member", "source_family_id": "m_2"},
                {"column": "Q1", "assigned_table": "q_long", "assignment_role": "melt_member", "source_family_id": "q"},
                {"column": "ID", "assigned_table": "students", "assignment_role": "base_key", "source_family_id": ""},
                {"column": "Age", "assigned_table": "students", "assignment_role": "base_attribute", "source_family_id": ""},
                {"column": "GPA", "assigned_table": "students", "assignment_role": "base_attribute", "source_family_id": ""},
                {"column": "Final_Grade", "assigned_table": "students", "assignment_role": "base_attribute", "source_family_id": ""},
                {"column": "ANXATT", "assigned_table": "students", "assignment_role": "base_attribute", "source_family_id": ""},
                {"column": "MATHATT", "assigned_table": "students", "assignment_role": "base_attribute", "source_family_id": ""},
            ],
            "table_suggestions": [
                {"table_name": "a1_long", "kind": "child_repeat"},
                {"table_name": "a2_long", "kind": "child_repeat"},
                {"table_name": "m1_long", "kind": "child_repeat"},
                {"table_name": "m2_long", "kind": "child_repeat"},
                {"table_name": "q_long", "kind": "child_repeat"},
                {"table_name": "students", "kind": "base_table"},
            ],
        },
        scale_mapping_json=scale_mapping_json,
        support=support,
    )

    rows = {row["column"]: row for row in canon["column_contracts"]}
    for column in ["A1Row1", "A2Row1"]:
        _assert(rows[column]["recommended_logical_type"] == "ordinal_category", f"expected confirmed family mapping to refine {column}")
        _assert(rows[column]["type_decision_source"] == "scale_mapping_resolver", f"expected scale mapping to own the type refinement for {column}")
    _assert(rows["M1_Q1"]["recommended_logical_type"] == "nominal_category", "expected M1 categorical answers to stay categorical")
    _assert(rows["M1_Q1"]["type_decision_source"] == "a17_baseline", "expected M1 to preserve baseline type source")
    _assert(rows["M2_Q1"]["recommended_logical_type"] == "nominal_category", "expected M2 categorical answers to stay categorical")
    _assert(rows["M2_Q1"]["type_decision_source"] == "a17_baseline", "expected M2 to preserve baseline type source")
    _assert(rows["Q1"]["recommended_logical_type"] == "nominal_category", "expected Q answer-key codes to stay categorical")
    _assert(rows["Q1"]["type_decision_source"] == "a17_baseline", "expected Q answer-key codes to preserve baseline type source")
    _assert(rows["ID"]["recommended_logical_type"] == "identifier", "expected ID to remain an identifier")
    _assert(rows["ID"]["type_decision_source"] == "a17_baseline", "expected ID to preserve baseline type source")
    for column in ["Age", "GPA", "Final_Grade"]:
        _assert(rows[column]["recommended_logical_type"] == "numeric_measure", f"expected {column} to remain numeric")
        _assert(rows[column]["type_decision_source"] == "a17_baseline", f"expected {column} to preserve baseline type source")
    for column in ["ANXATT", "MATHATT"]:
        _assert(rows[column]["recommended_logical_type"] == "categorical_code", f"expected {column} to remain a coded gating field")
        _assert(rows[column]["type_decision_source"] == "a17_baseline", f"expected {column} to preserve baseline type source")


def test_scale_mapping_resolver_reconciles_codebook_labels_to_numeric_source_tokens():
    support = {
        "a2_rows": [
            {
                "column": "A1Row1",
                "top_levels": ["1", "2", "3", "4", "5"],
                "a2_samples": {"random": []},
                "unique_count": 5,
                "top_candidate": {"type": "categorical"},
            },
            {
                "column": "A2Row1",
                "top_levels": ["1", "2", "3", "4", "5"],
                "a2_samples": {"random": []},
                "unique_count": 5,
                "top_candidate": {"type": "categorical"},
            },
        ],
        "a2_by_col": {},
        "a2_order": ["A1Row1", "A2Row1"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {
            "A1Row1": "a_1",
            "A2Row1": "a_2",
        },
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
            {"family_id": "a_1", "status": "accept", "table_name": "a1_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
            {"family_id": "a_2", "status": "accept", "table_name": "a2_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
        ],
        "override_notes": {},
        "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
        "scale_mapping_input": [],
    }
    family_worker_json = {
        "family_results": [
            {"family_result": {"family_id": "a_1", "recommended_family_role": "repeated_survey_block", "member_semantics_notes": "STARS anxiety start-of-term"}},
            {"family_result": {"family_id": "a_2", "recommended_family_role": "repeated_survey_block", "member_semantics_notes": "STARS anxiety end-of-term"}},
        ]
    }
    extractor_json = {
        "worker": "scale_mapping_extractor",
        "summary": {"overview": "", "key_points": []},
        "mappings": [
            {
                "target_kind": "family",
                "target_id": "a_1",
                "mapping_status": "codebook_confirmed",
                "response_scale_kind": "anxiety_likert_1_to_5",
                "ordered_labels": ["No anxiety 1", "2", "3", "4", "Strong anxiety 5"],
                "label_to_ordinal_position": {
                    "No anxiety 1": 1,
                    "2": 2,
                    "3": 3,
                    "4": 4,
                    "Strong anxiety 5": 5,
                },
                "label_to_numeric_score": {
                    "No anxiety 1": 1,
                    "2": 2,
                    "3": 3,
                    "4": 4,
                    "Strong anxiety 5": 5,
                },
                "numeric_score_semantics_confirmed": True,
                "source": "codebook_pdf",
                "notes": "Codebook says 1 means No anxiety and 5 means Strong anxiety.",
                "confidence": 0.92,
            }
        ],
        "review_flags": [],
        "assumptions": [],
    }

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        family_worker_json=family_worker_json,
        scale_mapping_extractor_json=extractor_json,
    )
    by_target = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}
    a1 = by_target[("family", "a_1")]
    _assert(a1["ordered_labels"] == ["1", "2", "3", "4", "5"], "expected resolver to rewrite polluted labels to observed numeric source tokens")
    _assert(a1["label_to_numeric_score"] == {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}, "expected reconciled numeric mapping to use raw numeric tokens")
    _assert("Codebook meaning: 1 = No anxiety; 5 = Strong anxiety." in a1["notes"], "expected semantic explanation to be preserved in notes")

    support["a17_by_col"] = {
        "A1Row1": {
            "column": "A1Row1",
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
        }
    }
    canon = app._synthesize_canonical_column_contract(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        semantic_context_json={"status": "skipped", "reason": "light_contract_accepted"},
        type_transform_worker_json={"column_decisions": [], "global_transform_rules": []},
        missingness_worker_json={"column_decisions": [], "global_contract": {"token_missing_placeholders_detected": False, "notes": ""}, "global_findings": []},
        family_worker_json=family_worker_json,
        table_layout_worker_json={
            "column_table_assignments": [
                {"column": "A1Row1", "assigned_table": "a1_long", "assignment_role": "melt_member", "source_family_id": "a_1"},
            ],
            "table_suggestions": [
                {"table_name": "a1_long", "kind": "child_repeat"},
            ],
        },
        scale_mapping_json=scale_mapping_json,
        support=support,
    )
    canon_row = {row["column"]: row for row in canon["column_contracts"]}["A1Row1"]
    _assert("Ordered labels: 1 | 2 | 3 | 4 | 5." in canon_row["codebook_note"], "expected canon note to use reconciled raw source labels")
    _assert("No anxiety 1" not in canon_row["codebook_note"], "expected canon note to avoid polluted hybrid labels")


def test_scale_mapping_resolver_preserves_textual_source_labels():
    support = {
        "a2_rows": [
            {
                "column": "AnxietyOverall",
                "top_levels": ["No anxiety", "Low", "Moderate", "High", "Strong anxiety"],
                "a2_samples": {"random": []},
                "unique_count": 5,
                "top_candidate": {"type": "categorical"},
            }
        ],
        "a2_by_col": {},
        "a2_order": ["AnxietyOverall"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    support["a2_by_col"] = {row["column"]: row for row in support["a2_rows"]}
    app._load_canonical_support_artifacts = lambda run_id: support

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions={
            "run_id": "run_smoke",
            "light_contract_status": "accepted",
            "primary_grain_decision": {"status": "accept", "keys": ["RespondentId"], "comments": ""},
            "reference_decisions": [],
            "dimension_decisions": [],
            "family_decisions": [],
            "override_notes": {},
            "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
            "scale_mapping_input": [],
        },
        family_worker_json={},
        scale_mapping_extractor_json={
            "worker": "scale_mapping_extractor",
            "summary": {"overview": "", "key_points": []},
            "mappings": [
                {
                    "target_kind": "column",
                    "target_id": "AnxietyOverall",
                    "mapping_status": "codebook_confirmed",
                    "response_scale_kind": "anxiety_likert_1_to_5",
                    "ordered_labels": ["No anxiety", "Low", "Moderate", "High", "Strong anxiety"],
                    "label_to_ordinal_position": {
                        "No anxiety": 1,
                        "Low": 2,
                        "Moderate": 3,
                        "High": 4,
                        "Strong anxiety": 5,
                    },
                    "label_to_numeric_score": {},
                    "numeric_score_semantics_confirmed": False,
                    "source": "codebook_pdf",
                    "notes": "Observed raw values are already textual.",
                    "confidence": 0.92,
                }
            ],
            "review_flags": [],
            "assumptions": [],
        },
    )
    by_target = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}
    textual = by_target[("column", "AnxietyOverall")]
    _assert(textual["ordered_labels"] == ["No anxiety", "Low", "Moderate", "High", "Strong anxiety"], "expected textual raw labels to remain unchanged")


def test_scale_mapping_resolver_reconciles_numeric_tokens_with_whitespace():
    support = {
        "a2_rows": [
            {
                "column": "AnxietyNumeric",
                "top_levels": [" 1 ", "2", " 3", "4 ", "5"],
                "a2_samples": {"random": []},
                "unique_count": 5,
                "top_candidate": {"type": "categorical"},
            }
        ],
        "a2_by_col": {},
        "a2_order": ["AnxietyNumeric"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    support["a2_by_col"] = {row["column"]: row for row in support["a2_rows"]}
    app._load_canonical_support_artifacts = lambda run_id: support

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions={
            "run_id": "run_smoke",
            "light_contract_status": "accepted",
            "primary_grain_decision": {"status": "accept", "keys": ["RespondentId"], "comments": ""},
            "reference_decisions": [],
            "dimension_decisions": [],
            "family_decisions": [],
            "override_notes": {},
            "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
            "scale_mapping_input": [],
        },
        family_worker_json={},
        scale_mapping_extractor_json={
            "worker": "scale_mapping_extractor",
            "summary": {"overview": "", "key_points": []},
            "mappings": [
                {
                    "target_kind": "column",
                    "target_id": "AnxietyNumeric",
                    "mapping_status": "codebook_confirmed",
                    "response_scale_kind": "anxiety_likert_1_to_5",
                    "ordered_labels": ["No anxiety 1", "2", "3", "4", "Strong anxiety 5"],
                    "label_to_ordinal_position": {"No anxiety 1": 1, "2": 2, "3": 3, "4": 4, "Strong anxiety 5": 5},
                    "label_to_numeric_score": {"No anxiety 1": 1, "2": 2, "3": 3, "4": 4, "Strong anxiety 5": 5},
                    "numeric_score_semantics_confirmed": True,
                    "source": "codebook_pdf",
                    "notes": "Codebook says 1 means No anxiety and 5 means Strong anxiety.",
                    "confidence": 0.92,
                }
            ],
            "review_flags": [],
            "assumptions": [],
        },
    )
    mapping = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}[("column", "AnxietyNumeric")]
    _assert(mapping["ordered_labels"] == ["1", "2", "3", "4", "5"], "expected whitespace-padded numeric raw tokens to reconcile cleanly")


def test_scale_mapping_resolver_flags_ambiguous_source_value_mismatch():
    support = {
        "a2_rows": [
            {
                "column": "AnxietyMismatch",
                "top_levels": ["1", "2", "4", "5"],
                "a2_samples": {"random": []},
                "unique_count": 4,
                "top_candidate": {"type": "categorical"},
            }
        ],
        "a2_by_col": {},
        "a2_order": ["AnxietyMismatch"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    support["a2_by_col"] = {row["column"]: row for row in support["a2_rows"]}
    app._load_canonical_support_artifacts = lambda run_id: support

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions={
            "run_id": "run_smoke",
            "light_contract_status": "accepted",
            "primary_grain_decision": {"status": "accept", "keys": ["RespondentId"], "comments": ""},
            "reference_decisions": [],
            "dimension_decisions": [],
            "family_decisions": [],
            "override_notes": {},
            "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
            "scale_mapping_input": [],
        },
        family_worker_json={},
        scale_mapping_extractor_json={
            "worker": "scale_mapping_extractor",
            "summary": {"overview": "", "key_points": []},
            "mappings": [
                {
                    "target_kind": "column",
                    "target_id": "AnxietyMismatch",
                    "mapping_status": "codebook_confirmed",
                    "response_scale_kind": "anxiety_likert_1_to_5",
                    "ordered_labels": ["No anxiety 1", "2", "3", "Strong anxiety 5"],
                    "label_to_ordinal_position": {"No anxiety 1": 1, "2": 2, "3": 3, "Strong anxiety 5": 4},
                    "label_to_numeric_score": {"No anxiety 1": 1, "2": 2, "3": 3, "Strong anxiety 5": 5},
                    "numeric_score_semantics_confirmed": True,
                    "source": "codebook_pdf",
                    "notes": "Extractor proposed a 1-5 ladder.",
                    "confidence": 0.92,
                }
            ],
            "review_flags": [],
            "assumptions": [],
        },
    )
    mapping = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}[("column", "AnxietyMismatch")]
    _assert(mapping["ordered_labels"] == ["No anxiety 1", "2", "3", "Strong anxiety 5"], "expected ambiguous mapping to remain unchanged when reconciliation is unsafe")
    review_flags = scale_mapping_json.get("review_flags") or []
    _assert(any(flag.get("issue") == "scale_mapping_source_value_mismatch" and flag.get("item") == "column:AnxietyMismatch" for flag in review_flags), "expected ambiguous mismatch to raise a resolver review flag")


def test_scale_mapping_resolver_preserves_human_override_precedence():
    support = {
        "a2_rows": [
            {
                "column": "A1Row1",
                "top_levels": ["1", "2", "3", "4", "5"],
                "a2_samples": {"random": []},
                "unique_count": 5,
                "top_candidate": {"type": "categorical"},
            }
        ],
        "a2_by_col": {},
        "a2_order": ["A1Row1"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {"A1Row1": "a_1"},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    support["a2_by_col"] = {row["column"]: row for row in support["a2_rows"]}
    app._load_canonical_support_artifacts = lambda run_id: support

    scale_mapping_json = app._build_scale_mapping_contract(
        run_id="run_smoke",
        light_contract_decisions={
            "run_id": "run_smoke",
            "light_contract_status": "accepted",
            "primary_grain_decision": {"status": "accept", "keys": ["RespondentId"], "comments": ""},
            "reference_decisions": [],
            "dimension_decisions": [],
            "family_decisions": [
                {"family_id": "a_1", "status": "accept", "table_name": "a1_long", "repeat_index_name": "row", "parent_key": "RespondentId", "comments": ""},
            ],
            "override_notes": {},
            "semantic_context_input": {"dataset_context_and_collection_notes": "", "semantic_codebook_and_important_variables": ""},
            "scale_mapping_input": [
                {
                    "target_kind": "family",
                    "target_id": "a_1",
                    "response_scale_kind": "anxiety_likert_1_to_5",
                    "ordered_labels": ["1", "2", "3", "4", "5"],
                    "numeric_scores": [1, 2, 3, 4, 5],
                    "notes": "Human confirmed numeric ladder.",
                }
            ],
        },
        family_worker_json={},
        scale_mapping_extractor_json={
            "worker": "scale_mapping_extractor",
            "summary": {"overview": "", "key_points": []},
            "mappings": [
                {
                    "target_kind": "family",
                    "target_id": "a_1",
                    "mapping_status": "codebook_confirmed",
                    "response_scale_kind": "anxiety_likert_1_to_5",
                    "ordered_labels": ["No anxiety 1", "2", "3", "4", "Strong anxiety 5"],
                    "label_to_ordinal_position": {"No anxiety 1": 1, "2": 2, "3": 3, "4": 4, "Strong anxiety 5": 5},
                    "label_to_numeric_score": {"No anxiety 1": 1, "2": 2, "3": 3, "4": 4, "Strong anxiety 5": 5},
                    "numeric_score_semantics_confirmed": True,
                    "source": "codebook_pdf",
                    "notes": "Extractor tried to restate the codebook anchors.",
                    "confidence": 0.92,
                }
            ],
            "review_flags": [],
            "assumptions": [],
        },
    )
    mapping = {(row["target_kind"], row["target_id"]): row for row in scale_mapping_json["mappings"]}[("family", "a_1")]
    _assert(mapping["mapping_status"] == "human_confirmed", "expected structured light-contract mapping to keep precedence over extractor output")
    _assert(mapping["ordered_labels"] == ["1", "2", "3", "4", "5"], "expected human-confirmed raw labels to remain untouched")


def test_scale_mapping_bundle_no_codebook_render_request():
    support = {
        "a2_rows": [],
        "a2_by_col": {},
        "a2_order": [],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    app._load_canonical_support_artifacts = lambda run_id: support
    app._load_optional_json_from_run_object = lambda run_id, object_name: None

    bundle = app._build_scale_mapping_bundle(
        run_id="run_smoke",
        light_contract_decisions={
            "semantic_context_input": {
                "dataset_context_and_collection_notes": "",
                "semantic_codebook_and_important_variables": "",
            }
        },
        family_worker_json={},
        include_rendered_codebook_pages=True,
    )

    codebook_context = bundle["codebook_context"]
    _assert(codebook_context["present"] is False, "expected codebook to be absent")
    _assert(codebook_context["rendered_page_images"] == [], "expected no rendered page images")
    _assert(codebook_context["rendered_page_selection"]["requested"] is False, "expected render request to stay false without codebook")
    _assert(codebook_context["combined_rendered_pages"]["present"] is False, "expected no combined rendered page artifact")


def test_scale_mapping_bundle_rendered_pages_and_cache():
    support = {
        "a2_rows": [
            {
                "column": "Q9Row1",
                "top_levels": ["Never Heard of It 0", "Very Familiar 6"],
                "a2_samples": {"random": []},
                "unique_count": 2,
                "top_candidate": {"type": "categorical"},
            }
        ],
        "a2_by_col": {
            "Q9Row1": {
                "column": "Q9Row1",
                "top_levels": ["Never Heard of It 0", "Very Familiar 6"],
                "a2_samples": {"random": []},
                "unique_count": 2,
                "top_candidate": {"type": "categorical"},
            }
        },
        "a2_order": ["Q9Row1"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {"Q9Row1": "q_9_main_cell_group"},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    app._load_canonical_support_artifacts = lambda run_id: support

    manifest_state = {"manifest": None, "render_calls": 0, "contact_sheet_calls": 0}
    codebook_document = {
        "filename": "codebook.pdf",
        "object_path": "runs/run_smoke/codebook.pdf",
        "page_count": 5,
        "signed_url": "https://signed/codebook.pdf",
    }
    codebook_pages = {
        "pages": [
            {"page_number": 1, "text": "introduction page", "char_count": 150},
            {"page_number": 2, "text": "Q9Main_cell_group Never Heard of It Very Familiar", "char_count": 120},
            {"page_number": 3, "text": "", "char_count": 0},
            {"page_number": 4, "text": "", "char_count": 0},
            {"page_number": 5, "text": "", "char_count": 0},
        ]
    }

    def fake_load_optional_json(run_id, object_name):
        if object_name == "codebook_document.json":
            return codebook_document
        if object_name == "codebook_pages.json":
            return codebook_pages
        if object_name == app.CODEBOOK_RENDER_MANIFEST_OBJECT_NAME:
            return manifest_state["manifest"]
        return None

    def fake_upload_json(run_id, object_name, payload):
        if object_name == app.CODEBOOK_RENDER_MANIFEST_OBJECT_NAME:
            manifest_state["manifest"] = payload
        return f"runs/{run_id}/{object_name}"

    def fake_render(pdf_bytes, page_numbers, scale=0):
        manifest_state["render_calls"] += 1
        return {page_number: f"png-{page_number}".encode("utf-8") for page_number in page_numbers}

    def fake_contact_sheet(rendered_bytes, page_numbers):
        manifest_state["contact_sheet_calls"] += 1
        return b"combined-png", "grid_2col", len(page_numbers)

    app._load_optional_json_from_run_object = fake_load_optional_json
    app._load_bytes_from_run_object = lambda run_id, object_name: b"%PDF-render-test"
    app._upload_json_to_run_object = fake_upload_json
    app._upload_bytes_to_run_object = lambda run_id, object_name, payload, content_type: f"runs/{run_id}/{object_name}"
    app._try_sign_run_object_download = lambda run_id, object_name, filename, content_type: (f"https://signed/{object_name}", "")
    app._run_object_exists = lambda run_id, object_name: True
    app._render_codebook_pages_to_pngs = fake_render
    app._build_codebook_contact_sheet = fake_contact_sheet

    light_contract_decisions = {
        "semantic_context_input": {
            "dataset_context_and_collection_notes": "",
            "semantic_codebook_and_important_variables": "",
        }
    }
    family_worker_json = {
        "family_results": [
            {
                "family_result": {
                    "family_id": "q_9_main_cell_group",
                    "recommended_family_role": "repeated_survey_block",
                    "member_semantics_notes": "Familiarity scale",
                }
            }
        ]
    }

    first_bundle = app._build_scale_mapping_bundle(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        family_worker_json=family_worker_json,
        include_rendered_codebook_pages=True,
    )
    first_context = first_bundle["codebook_context"]
    _assert(first_context["rendered_page_selection"]["requested"] is True, "expected render request to be enabled")
    _assert(first_context["rendered_page_selection"]["selected_page_numbers"] == [1, 2, 3, 4, 5], "expected matched, adjacent, and low-text pages to be selected")
    _assert([row["page_number"] for row in first_context["rendered_page_images"]] == [1, 2, 3, 4, 5], "expected rendered page entries in page order")
    _assert(first_context["combined_rendered_pages"]["present"] is True, "expected combined rendered pages artifact")
    _assert(first_context["combined_rendered_pages"]["page_numbers"] == [1, 2, 3, 4, 5], "expected combined artifact page numbers to match selection")
    _assert(first_context["combined_rendered_pages"]["layout"] == "grid_2col", "expected 5-page contact sheet to use 2-column grid")
    _assert(first_context["combined_rendered_pages"]["tile_count"] == 5, "expected combined artifact tile count to match selection")
    _assert(manifest_state["render_calls"] == 1, "expected first render pass to render once")
    _assert(manifest_state["contact_sheet_calls"] == 1, "expected first render pass to build one contact sheet")

    second_bundle = app._build_scale_mapping_bundle(
        run_id="run_smoke",
        light_contract_decisions=light_contract_decisions,
        family_worker_json=family_worker_json,
        include_rendered_codebook_pages=True,
    )
    second_context = second_bundle["codebook_context"]
    _assert([row["image_signed_url"] for row in second_context["rendered_page_images"]], "expected signed URLs on cached render reuse")
    _assert(second_context["combined_rendered_pages"]["image_signed_url"], "expected signed URL on cached combined artifact reuse")
    _assert(manifest_state["render_calls"] == 1, "expected cached render manifest reuse without rerendering")
    _assert(manifest_state["contact_sheet_calls"] == 1, "expected cached combined artifact reuse without recomposing")


def test_scale_mapping_bundle_rendered_pages_vertical_layout():
    support = {
        "a2_rows": [
            {
                "column": "Q9Row1",
                "top_levels": ["Never Heard of It 0", "Very Familiar 6"],
                "a2_samples": {"random": []},
                "unique_count": 2,
                "top_candidate": {"type": "categorical"},
            }
        ],
        "a2_by_col": {
            "Q9Row1": {
                "column": "Q9Row1",
                "top_levels": ["Never Heard of It 0", "Very Familiar 6"],
                "a2_samples": {"random": []},
                "unique_count": 2,
                "top_candidate": {"type": "categorical"},
            }
        },
        "a2_order": ["Q9Row1"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {"Q9Row1": "q_9_main_cell_group"},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    app._load_canonical_support_artifacts = lambda run_id: support

    manifest_state = {"manifest": None}
    codebook_document = {
        "filename": "codebook.pdf",
        "object_path": "runs/run_smoke/codebook.pdf",
        "page_count": 2,
        "signed_url": "https://signed/codebook.pdf",
    }
    codebook_pages = {
        "pages": [
            {"page_number": 1, "text": "Q9Main_cell_group Never Heard of It Very Familiar", "char_count": 120},
            {"page_number": 2, "text": "", "char_count": 0},
        ]
    }

    def fake_load_optional_json(run_id, object_name):
        if object_name == "codebook_document.json":
            return codebook_document
        if object_name == "codebook_pages.json":
            return codebook_pages
        if object_name == app.CODEBOOK_RENDER_MANIFEST_OBJECT_NAME:
            return manifest_state["manifest"]
        return None

    app._load_optional_json_from_run_object = fake_load_optional_json
    app._load_bytes_from_run_object = lambda run_id, object_name: b"%PDF-vertical"
    app._upload_json_to_run_object = lambda run_id, object_name, payload: manifest_state.__setitem__("manifest", payload) or f"runs/{run_id}/{object_name}"
    app._upload_bytes_to_run_object = lambda run_id, object_name, payload, content_type: f"runs/{run_id}/{object_name}"
    app._try_sign_run_object_download = lambda run_id, object_name, filename, content_type: (f"https://signed/{object_name}", "")
    app._run_object_exists = lambda run_id, object_name: True
    app._render_codebook_pages_to_pngs = lambda pdf_bytes, page_numbers, scale=0: {
        page_number: f"png-{page_number}".encode("utf-8") for page_number in page_numbers
    }
    app._build_codebook_contact_sheet = lambda rendered_bytes, page_numbers: (b"combined-png", "vertical", len(page_numbers))

    bundle = app._build_scale_mapping_bundle(
        run_id="run_smoke",
        light_contract_decisions={
            "semantic_context_input": {
                "dataset_context_and_collection_notes": "",
                "semantic_codebook_and_important_variables": "",
            }
        },
        family_worker_json={
            "family_results": [
                {
                    "family_result": {
                        "family_id": "q_9_main_cell_group",
                        "recommended_family_role": "repeated_survey_block",
                        "member_semantics_notes": "Familiarity scale",
                    }
                }
            ]
        },
        include_rendered_codebook_pages=True,
    )

    combined = bundle["codebook_context"]["combined_rendered_pages"]
    _assert(combined["layout"] == "vertical", "expected 2-page contact sheet to use vertical layout")
    _assert(combined["tile_count"] == 2, "expected vertical contact sheet to include both pages")


def test_scale_mapping_bundle_rendered_pages_fallback():
    support = {
        "a2_rows": [],
        "a2_by_col": {},
        "a2_order": [],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    app._load_canonical_support_artifacts = lambda run_id: support

    manifest_state = {"manifest": None}
    codebook_document = {
        "filename": "codebook.pdf",
        "object_path": "runs/run_smoke/codebook.pdf",
        "page_count": 5,
        "signed_url": "https://signed/codebook.pdf",
    }
    codebook_pages = {
        "pages": [
            {"page_number": 1, "text": "overview", "char_count": 120},
            {"page_number": 2, "text": "methods", "char_count": 120},
            {"page_number": 3, "text": "appendix", "char_count": 120},
            {"page_number": 4, "text": "notes", "char_count": 120},
            {"page_number": 5, "text": "closing", "char_count": 120},
        ]
    }

    def fake_load_optional_json(run_id, object_name):
        if object_name == "codebook_document.json":
            return codebook_document
        if object_name == "codebook_pages.json":
            return codebook_pages
        if object_name == app.CODEBOOK_RENDER_MANIFEST_OBJECT_NAME:
            return manifest_state["manifest"]
        return None

    app._load_optional_json_from_run_object = fake_load_optional_json
    app._load_bytes_from_run_object = lambda run_id, object_name: b"%PDF-fallback"
    app._upload_json_to_run_object = lambda run_id, object_name, payload: manifest_state.__setitem__("manifest", payload) or f"runs/{run_id}/{object_name}"
    app._upload_bytes_to_run_object = lambda run_id, object_name, payload, content_type: f"runs/{run_id}/{object_name}"
    app._try_sign_run_object_download = lambda run_id, object_name, filename, content_type: (f"https://signed/{object_name}", "")
    app._run_object_exists = lambda run_id, object_name: True
    app._render_codebook_pages_to_pngs = lambda pdf_bytes, page_numbers, scale=0: {
        page_number: f"png-{page_number}".encode("utf-8") for page_number in page_numbers
    }
    app._build_codebook_contact_sheet = lambda rendered_bytes, page_numbers: (b"combined-png", "grid_2col", len(page_numbers))

    bundle = app._build_scale_mapping_bundle(
        run_id="run_smoke",
        light_contract_decisions={
            "semantic_context_input": {
                "dataset_context_and_collection_notes": "",
                "semantic_codebook_and_important_variables": "",
            }
        },
        family_worker_json={},
        include_rendered_codebook_pages=True,
    )

    selection = bundle["codebook_context"]["rendered_page_selection"]
    _assert(selection["used_fallback"] is True, "expected fallback mode when no matched pages exist")
    _assert(selection["selected_page_numbers"] == [1, 2, 3, 4], "expected fallback to render the first four pages")
    combined = bundle["codebook_context"]["combined_rendered_pages"]
    _assert(combined["page_numbers"] == [1, 2, 3, 4], "expected combined artifact to use fallback-selected pages")


def test_scale_mapping_bundle_combined_rendered_pages_signing_failure():
    support = {
        "a2_rows": [
            {
                "column": "Q9Row1",
                "top_levels": ["Never Heard of It 0", "Very Familiar 6"],
                "a2_samples": {"random": []},
                "unique_count": 2,
                "top_candidate": {"type": "categorical"},
            }
        ],
        "a2_by_col": {
            "Q9Row1": {
                "column": "Q9Row1",
                "top_levels": ["Never Heard of It 0", "Very Familiar 6"],
                "a2_samples": {"random": []},
                "unique_count": 2,
                "top_candidate": {"type": "categorical"},
            }
        },
        "a2_order": ["Q9Row1"],
        "a9_by_col": {},
        "a13_by_col": {},
        "a14_by_col": {},
        "a16_by_col": {},
        "a17_by_col": {},
        "family_by_column": {"Q9Row1": "q_9_main_cell_group"},
        "missing_artifacts": [],
        "a17_backfilled": False,
    }
    app._load_canonical_support_artifacts = lambda run_id: support

    manifest_state = {"manifest": None}
    codebook_document = {
        "filename": "codebook.pdf",
        "object_path": "runs/run_smoke/codebook.pdf",
        "page_count": 2,
        "signed_url": "https://signed/codebook.pdf",
    }
    codebook_pages = {
        "pages": [
            {"page_number": 1, "text": "Q9Main_cell_group Never Heard of It Very Familiar", "char_count": 120},
            {"page_number": 2, "text": "", "char_count": 0},
        ]
    }

    def fake_load_optional_json(run_id, object_name):
        if object_name == "codebook_document.json":
            return codebook_document
        if object_name == "codebook_pages.json":
            return codebook_pages
        if object_name == app.CODEBOOK_RENDER_MANIFEST_OBJECT_NAME:
            return manifest_state["manifest"]
        return None

    def fake_sign(run_id, object_name, filename, content_type):
        if object_name == app.CODEBOOK_RENDER_CONTACT_SHEET_OBJECT_NAME:
            return "", "combined signing failed"
        return f"https://signed/{object_name}", ""

    app._load_optional_json_from_run_object = fake_load_optional_json
    app._load_bytes_from_run_object = lambda run_id, object_name: b"%PDF-signing"
    app._upload_json_to_run_object = lambda run_id, object_name, payload: manifest_state.__setitem__("manifest", payload) or f"runs/{run_id}/{object_name}"
    app._upload_bytes_to_run_object = lambda run_id, object_name, payload, content_type: f"runs/{run_id}/{object_name}"
    app._try_sign_run_object_download = fake_sign
    app._run_object_exists = lambda run_id, object_name: True
    app._render_codebook_pages_to_pngs = lambda pdf_bytes, page_numbers, scale=0: {
        page_number: f"png-{page_number}".encode("utf-8") for page_number in page_numbers
    }
    app._build_codebook_contact_sheet = lambda rendered_bytes, page_numbers: (b"combined-png", "vertical", len(page_numbers))

    bundle = app._build_scale_mapping_bundle(
        run_id="run_smoke",
        light_contract_decisions={
            "semantic_context_input": {
                "dataset_context_and_collection_notes": "",
                "semantic_codebook_and_important_variables": "",
            }
        },
        family_worker_json={
            "family_results": [
                {
                    "family_result": {
                        "family_id": "q_9_main_cell_group",
                        "recommended_family_role": "repeated_survey_block",
                        "member_semantics_notes": "Familiarity scale",
                    }
                }
            ]
        },
        include_rendered_codebook_pages=True,
    )

    combined = bundle["codebook_context"]["combined_rendered_pages"]
    _assert(combined["present"] is True, "expected combined artifact metadata even when signing fails")
    _assert(combined["image_object_path"], "expected combined object path on signing failure")
    _assert(combined["image_signed_url"] == "", "expected blank combined signed URL on signing failure")
    _assert(combined["signed_url_error"] == "combined signing failed", "expected combined signing failure detail")


if __name__ == "__main__":
    test_light_contract_scale_mapping_sheet()
    test_scale_mapping_resolver_and_canonical_integration()
    test_scale_mapping_resolver_skips_numeric_standalone_columns()
    test_scale_mapping_resolver_drops_placeholder_human_family_rows()
    test_canonical_scale_mapping_applies_only_confirmed_label_complete_mappings()
    test_scale_mapping_resolver_reconciles_codebook_labels_to_numeric_source_tokens()
    test_scale_mapping_resolver_preserves_textual_source_labels()
    test_scale_mapping_resolver_reconciles_numeric_tokens_with_whitespace()
    test_scale_mapping_resolver_flags_ambiguous_source_value_mismatch()
    test_scale_mapping_resolver_preserves_human_override_precedence()
    test_scale_mapping_bundle_no_codebook_render_request()
    test_scale_mapping_bundle_rendered_pages_and_cache()
    test_scale_mapping_bundle_rendered_pages_vertical_layout()
    test_scale_mapping_bundle_rendered_pages_fallback()
    test_scale_mapping_bundle_combined_rendered_pages_signing_failure()
    print("scale mapping smoke passed")
