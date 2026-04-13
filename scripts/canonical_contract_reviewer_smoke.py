import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "testdata" / "canonical_reviewer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_validator_main():
    namespace = {}
    code = Path("JSON validators/canonical_contract_reviewer_validator.json").read_text()
    exec(code, namespace)
    return namespace["main"]


def _load_apply_main():
    namespace = {}
    code = Path("canonical_contract_reviewer_apply_patch_node.py").read_text()
    exec(code, namespace)
    return namespace["main"]


def _load_unwrap_helpers():
    namespace = {"__name__": "unwrap_canonical_reviewer_artifacts"}
    code = Path("scripts/unwrap_canonical_reviewer_artifacts.py").read_text()
    exec(code, namespace)
    return namespace["unwrap_known_artifacts"]


def _load_fixture_json(name):
    return json.loads((FIXTURES / name).read_text())


def _base_contract():
    return {
        "worker": "canonical_column_contract_builder",
        "summary": {
            "overview": "Reviewed baseline contract fixture.",
            "total_source_columns": 3,
            "included_column_count": 3,
            "excluded_column_count": 0,
            "unresolved_column_count": 0,
            "reviewed_override_count": 3,
            "family_default_count": 0,
            "deterministic_baseline_count": 0,
            "reviewed_type_count": 3,
            "fallback_type_count": 0,
            "key_contract_principles": [
                "Control fields are complete for every contract row; semantic enrichment stays blank unless evidence exists.",
                "Reviewed table layout, semantic context, reviewed specialist outputs, and family defaults outrank the A17 deterministic baseline.",
            ],
        },
        "column_contracts": [
            {
                "column": "RespondentId",
                "canonical_modeling_status": "base_field",
                "canonical_table_name": "respondents",
                "canonical_assignment_role": "base_key",
                "source_family_id": "",
                "a9_primary_role": "id_key",
                "recommended_logical_type": "identifier",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "cast_to_string"],
                "structural_transform_hints": [],
                "interpretation_hints": ["identifier_not_measure", "numeric_parse_is_misleading"],
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "Primary respondent identifier.",
                "quality_score": 1.0,
                "drift_detected": False,
                "type_decision_source": "reviewed_type_worker",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "a17_baseline",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "type_transform_worker"],
                "confidence": 0.99,
                "needs_human_review": False,
            },
            {
                "column": "Q2",
                "canonical_modeling_status": "base_field",
                "canonical_table_name": "respondents",
                "canonical_assignment_role": "base_attribute",
                "source_family_id": "",
                "a9_primary_role": "invariant_attr",
                "recommended_logical_type": "ordinal_category",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "normalize_category_tokens"],
                "structural_transform_hints": [],
                "interpretation_hints": [],
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "Demographic gating question.",
                "quality_score": 0.99,
                "drift_detected": False,
                "type_decision_source": "reviewed_type_worker",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "reviewed_missingness_worker",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "type_transform_worker", "missingness_worker"],
                "confidence": 0.9,
                "needs_human_review": False,
            },
            {
                "column": "Q2_Other_Other__please_specify",
                "canonical_modeling_status": "base_field",
                "canonical_table_name": "respondents",
                "canonical_assignment_role": "base_attribute",
                "source_family_id": "",
                "a9_primary_role": "measure",
                "recommended_logical_type": "free_text",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "cast_to_string"],
                "structural_transform_hints": [],
                "interpretation_hints": ["skip_logic_protected"],
                "missingness_disposition": "structurally_valid_missingness",
                "missingness_handling": "protect_from_null_penalty",
                "skip_logic_protected": True,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "Other-specify text expected to be mostly missing when Q2 is not Other.",
                "quality_score": 0.79,
                "drift_detected": False,
                "type_decision_source": "reviewed_type_worker",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "reviewed_missingness_worker",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "type_transform_worker", "missingness_worker"],
                "confidence": 0.75,
                "needs_human_review": True,
            },
        ],
        "global_value_rules": [],
        "review_flags": [],
        "assumptions": [],
    }


def _stale_summary_contract():
    contract = _base_contract()
    contract["summary"]["fallback_type_count"] = 999
    return contract


def _stale_incoherent_contract():
    contract = _base_contract()
    contract["column_contracts"][1]["missingness_handling"] = "protect_from_null_penalty"
    contract["column_contracts"][1]["skip_logic_protected"] = False
    return contract


def _stats_regression_contract():
    return {
        "worker": "canonical_column_contract_builder",
        "summary": {
            "overview": "Statistics reviewer regression fixture.",
            "total_source_columns": 4,
            "included_column_count": 4,
            "excluded_column_count": 0,
            "unresolved_column_count": 0,
            "reviewed_override_count": 1,
            "family_default_count": 1,
            "deterministic_baseline_count": 2,
            "reviewed_type_count": 1,
            "fallback_type_count": 3,
            "key_contract_principles": [
                "Keep skip-logic bookkeeping deterministic and keep reviewer edits on substantive row fields only.",
            ],
        },
        "column_contracts": [
            {
                "column": "Mjr2",
                "canonical_modeling_status": "base_field",
                "canonical_table_name": "base_students",
                "canonical_assignment_role": "base_attribute",
                "source_family_id": "",
                "a9_primary_role": "invariant_attr",
                "recommended_logical_type": "categorical_code",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "titlecase_values", "cast_to_string"],
                "structural_transform_hints": [],
                "interpretation_hints": ["skip_logic_protected"],
                "missingness_disposition": "partially_structural_missingness",
                "missingness_handling": "retain_with_caution",
                "skip_logic_protected": False,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "Blank commonly means no second major.",
                "quality_score": 0.75,
                "drift_detected": False,
                "type_decision_source": "reviewed_type_worker",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "reviewed_missingness_worker",
                "semantic_decision_source": "semantic_context_worker",
                "applied_sources": [
                    "table_layout_worker",
                    "semantic_context_worker",
                    "type_transform_worker",
                    "missingness_worker",
                ],
                "confidence": 0.75,
                "needs_human_review": True,
            },
            {
                "column": "A1_Q2",
                "canonical_modeling_status": "child_repeat_member",
                "canonical_table_name": "a1_question_responses",
                "canonical_assignment_role": "melt_member",
                "source_family_id": "a_1",
                "a9_primary_role": "measure_item",
                "recommended_logical_type": "ordinal_category",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "normalize_category_tokens", "cast_to_string"],
                "structural_transform_hints": [],
                "interpretation_hints": ["skip_logic_protected", "repeat_context_do_not_use_as_base_key"],
                "missingness_disposition": "structurally_valid_missingness",
                "missingness_handling": "protect_from_null_penalty",
                "skip_logic_protected": True,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "ANXATT-gated family member.",
                "quality_score": 0.75,
                "drift_detected": False,
                "type_decision_source": "family_default",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "family_default",
                "semantic_decision_source": "family_worker",
                "applied_sources": [
                    "table_layout_worker",
                    "A17",
                    "family_worker.family_result",
                    "family_worker.member_defaults",
                ],
                "confidence": 0.75,
                "needs_human_review": True,
            },
            {
                "column": "GPA",
                "canonical_modeling_status": "base_field",
                "canonical_table_name": "base_students",
                "canonical_assignment_role": "base_attribute",
                "source_family_id": "",
                "a9_primary_role": "measure",
                "recommended_logical_type": "numeric_measure",
                "recommended_storage_type": "decimal",
                "transform_actions": ["trim_whitespace", "strip_numeric_formatting", "cast_to_decimal"],
                "structural_transform_hints": [],
                "interpretation_hints": [],
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "Core respondent-level measure.",
                "quality_score": 0.84,
                "drift_detected": False,
                "type_decision_source": "a17_baseline",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "a17_baseline",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "A17", "A2", "A9", "A14", "A4"],
                "confidence": 0.84,
                "needs_human_review": False,
            },
            {
                "column": "Final_Grade",
                "canonical_modeling_status": "base_field",
                "canonical_table_name": "base_students",
                "canonical_assignment_role": "base_attribute",
                "source_family_id": "",
                "a9_primary_role": "measure",
                "recommended_logical_type": "numeric_measure",
                "recommended_storage_type": "decimal",
                "transform_actions": ["trim_whitespace", "strip_numeric_formatting", "cast_to_decimal"],
                "structural_transform_hints": [],
                "interpretation_hints": [],
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "Core respondent-level measure.",
                "quality_score": 0.84,
                "drift_detected": False,
                "type_decision_source": "a17_baseline",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "a17_baseline",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "A17", "A2", "A9", "A14", "A4"],
                "confidence": 0.84,
                "needs_human_review": False,
            },
        ],
        "global_value_rules": [],
        "review_flags": [],
        "assumptions": [],
    }


def _build_patch_envelope(change_set, overview):
    return {
        "worker": "canonical_contract_reviewer",
        "review_summary": {
            "overview": overview,
            "review_principles": [
                "Preserve deterministic contract structure and patch only evidence-supported row fields.",
            ],
        },
        "change_set": change_set,
        "review_flags": [],
        "assumptions": [],
    }


def _build_patch(change_id, column, field, after_value, reasoning, justification, confidence=0.9, needs_human_review=False):
    return {
        "change_id": change_id,
        "column": column,
        "field": field,
        "after_value": after_value,
        "reasoning": reasoning,
        "justification": justification,
        "confidence": confidence,
        "needs_human_review": needs_human_review,
    }


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _validate_patch(name, reviewer_output, original_contract, expected_ok, expected_error_substring=""):
    validator = _load_validator_main()
    raw_output = reviewer_output if isinstance(reviewer_output, str) else json.dumps(reviewer_output)
    result = validator(raw_output, json.dumps(original_contract))
    actual_ok = result["validation_ok"] == "true"
    if actual_ok != expected_ok:
        raise AssertionError(f"{name} expected validation_ok={expected_ok}, got {result}")
    error_haystack = f"{result.get('validation_error', '')}\n{result.get('validation_errors_json', '')}"
    if expected_error_substring and expected_error_substring not in error_haystack:
        raise AssertionError(f"{name} expected error containing {expected_error_substring!r}, got {result}")
    print(f"PASS: {name}")
    return result


def _apply_patch(patch_json, original_contract):
    apply_main = _load_apply_main()
    result = apply_main(patch_json, json.dumps(original_contract))
    return json.loads(result["canon_review_json"])


def _assert_apply_fails(name, patch_json, original_contract, expected_error_substring):
    apply_main = _load_apply_main()
    try:
        apply_main(patch_json, json.dumps(original_contract))
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if expected_error_substring not in message:
            raise AssertionError(
                f"{name} expected error containing {expected_error_substring!r}, got {message!r}"
            ) from exc
        print(f"PASS: {name}")
        return
    raise AssertionError(f"{name} expected apply to fail")


def main():
    original = _base_contract()

    unchanged = _build_patch_envelope([], "No evidence-supported row edits were required.")
    unchanged_result = _validate_patch("unchanged patch passes", unchanged, original, expected_ok=True)
    unchanged_final = _apply_patch(unchanged_result["canon_review_patch"], original)
    _assert(unchanged_final["change_log"] == [], unchanged_final)
    _assert(unchanged_final["review_summary"]["change_count"] == 0, unchanged_final)
    _assert(unchanged_final["reviewed_contract"]["summary"]["review_applied"] is True, unchanged_final)
    _assert(unchanged_final["reviewed_contract"]["summary"]["review_change_count"] == 0, unchanged_final)
    _assert(unchanged_final["reviewed_contract"]["summary"]["review_changed_column_count"] == 0, unchanged_final)
    _assert("Post-review canonical column contract" in unchanged_final["reviewed_contract"]["summary"]["overview"], unchanged_final)
    _assert(unchanged_final["reviewer_flags"] == unchanged_final["review_flags"], unchanged_final)
    _assert(unchanged_final["reviewer_assumptions"] == unchanged_final["assumptions"], unchanged_final)
    _assert(unchanged_final["reviewed_contract"]["column_contracts"][2]["needs_human_review"] is True, unchanged_final)

    single_patch = _build_patch_envelope(
        [
            _build_patch(
                "chg_001",
                "Q2_Other_Other__please_specify",
                "needs_human_review",
                False,
                "The row already has direct skip-logic protection and does not need forced manual escalation.",
                "This is a reviewer-only calibration correction; no structural or missingness redesign is being made.",
                confidence=0.78,
            )
        ],
        "Lowered one human-review flag after contract coherence review.",
    )
    single_result = _validate_patch("single-field row patch passes", single_patch, original, expected_ok=True)
    single_final = _apply_patch(single_result["canon_review_patch"], original)
    _assert(single_final["review_summary"]["change_count"] == 1, single_final)
    _assert(single_final["review_summary"]["changed_column_count"] == 1, single_final)
    _assert(single_final["reviewed_contract"]["summary"]["review_change_count"] == 1, single_final)
    _assert(single_final["reviewed_contract"]["summary"]["review_changed_column_count"] == 1, single_final)
    _assert(single_final["reviewed_contract"]["column_contracts"][2]["needs_human_review"] is False, single_final)
    _assert(single_final["change_log"][0]["before_value"] is True, single_final)
    _assert(single_final["change_log"][0]["after_value"] is False, single_final)
    _assert(
        single_final["change_log"][0]["target_path"] == "/reviewed_contract/column_contracts/2/needs_human_review",
        single_final,
    )

    stale_original = _stale_summary_contract()
    stale_result = _validate_patch(
        "empty patch still passes on stale summary draft",
        _build_patch_envelope([], "No reviewer row edits were required."),
        stale_original,
        expected_ok=True,
    )
    stale_final = _apply_patch(stale_result["canon_review_patch"], stale_original)
    _assert(stale_final["review_summary"]["change_count"] == 1, stale_final)
    _assert(stale_final["review_summary"]["changed_column_count"] == 0, stale_final)
    _assert(stale_final["reviewed_contract"]["summary"]["review_change_count"] == 1, stale_final)
    _assert(stale_final["reviewed_contract"]["summary"]["review_changed_column_count"] == 0, stale_final)
    _assert(stale_final["change_log"][0]["target_path"] == "/reviewed_contract/summary/fallback_type_count", stale_final)
    _assert(stale_final["change_log"][0]["column"] == "", stale_final)
    _assert(stale_final["reviewed_contract"]["summary"]["fallback_type_count"] == 0, stale_final)

    duplicate_field_patch = _build_patch_envelope(
        [
            _build_patch(
                "chg_010",
                "Q2",
                "confidence",
                0.81,
                "Invalid fixture.",
                "Invalid fixture.",
            ),
            _build_patch(
                "chg_011",
                "Q2",
                "confidence",
                0.82,
                "Invalid fixture.",
                "Invalid fixture.",
            ),
        ],
        "Invalid duplicate field fixture.",
    )
    _validate_patch(
        "duplicate column field fails",
        duplicate_field_patch,
        original,
        expected_ok=False,
        expected_error_substring="Duplicate change_set column/field pair: Q2.confidence",
    )

    forbidden_field_patch = _build_patch_envelope(
        [
            _build_patch(
                "chg_020",
                "Q2",
                "a9_primary_role",
                "measure",
                "Invalid fixture.",
                "Invalid fixture.",
            )
        ],
        "Invalid forbidden-field fixture.",
    )
    _validate_patch(
        "forbidden field fails",
        forbidden_field_patch,
        original,
        expected_ok=False,
        expected_error_substring="field points to a forbidden field",
    )

    no_op_patch = _load_fixture_json("reviewer_patch_noop.json")
    no_op_result = _validate_patch(
        "no-op patch normalizes away",
        no_op_patch,
        original,
        expected_ok=True,
    )
    _assert(no_op_result["normalization_applied"] == "true", no_op_result)
    _assert(
        any("no_op_change_entry" in item["defaulted_to"] for item in json.loads(no_op_result["normalization_log_json"])),
        no_op_result,
    )
    _assert(json.loads(no_op_result["canon_review_patch"])["change_set"] == [], no_op_result)

    unknown_column_patch = _build_patch_envelope(
        [
            _build_patch(
                "chg_040",
                "GhostCol",
                "confidence",
                0.81,
                "Invalid fixture.",
                "Invalid fixture.",
            )
        ],
        "Invalid unknown column fixture.",
    )
    _validate_patch(
        "unknown column fails",
        unknown_column_patch,
        original,
        expected_ok=False,
        expected_error_substring=".column does not exist in canon_contract_json: GhostCol",
    )

    stale_incoherent = _validate_patch(
        "stale incoherent source contract fails before patch validation",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_041",
                    "Q2",
                    "confidence",
                    0.81,
                    "Valid-looking reviewer patch.",
                    "Valid-looking reviewer patch.",
                )
            ],
            "Should fail because the source contract is stale.",
        ),
        _stale_incoherent_contract(),
        expected_ok=False,
        expected_error_substring="canon_contract_json.column_contracts[1].skip_logic_protected must be true when missingness_handling is protect_from_null_penalty.",
    )

    stats_original = _stats_regression_contract()
    stats_unchanged = _validate_patch(
        "stats regression fixture accepts no-change review",
        _build_patch_envelope([], "No substantive reviewer-owned row edits were required."),
        stats_original,
        expected_ok=True,
    )
    stats_final = _apply_patch(stats_unchanged["canon_review_patch"], stats_original)
    _assert(stats_final["change_log"] == [], stats_final)

    _validate_patch(
        "frozen skip_logic_protected fails",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_001",
                    "Mjr2",
                    "skip_logic_protected",
                    True,
                    "Invalid reviewer-owned bookkeeping change.",
                    "Invalid reviewer-owned bookkeeping change.",
                )
            ],
            "Invalid frozen-field fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="field points to a forbidden field: skip_logic_protected",
    )

    _validate_patch(
        "frozen type provenance fails for GPA",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_002",
                    "GPA",
                    "type_decision_source",
                    "reviewed_type_worker",
                    "Invalid provenance promotion.",
                    "Invalid provenance promotion.",
                )
            ],
            "Invalid frozen-field fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="field points to a forbidden field: type_decision_source",
    )

    _validate_patch(
        "frozen type provenance fails for Final_Grade",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_003",
                    "Final_Grade",
                    "type_decision_source",
                    "reviewed_type_worker",
                    "Invalid provenance promotion.",
                    "Invalid provenance promotion.",
                )
            ],
            "Invalid frozen-field fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="field points to a forbidden field: type_decision_source",
    )

    _validate_patch(
        "frozen missingness provenance fails for A1_Q2",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_004",
                    "A1_Q2",
                    "missingness_decision_source",
                    "reviewed_missingness_worker",
                    "Invalid provenance promotion.",
                    "Invalid provenance promotion.",
                )
            ],
            "Invalid frozen-field fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="field points to a forbidden field: missingness_decision_source",
    )

    _validate_patch(
        "frozen applied_sources fails for A1_Q2",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_005",
                    "A1_Q2",
                    "applied_sources",
                    [
                        "table_layout_worker",
                        "A17",
                        "family_worker.family_result",
                        "family_worker.member_defaults",
                        "missingness_worker",
                    ],
                    "Invalid provenance promotion.",
                    "Invalid provenance promotion.",
                )
            ],
            "Invalid frozen-field fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="field points to a forbidden field: applied_sources",
    )

    _validate_patch(
        "skip hint bookkeeping edit fails",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_006",
                    "GPA",
                    "interpretation_hints",
                    ["skip_logic_protected"],
                    "Invalid bookkeeping leak.",
                    "Invalid bookkeeping leak.",
                )
            ],
            "Invalid bookkeeping fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="cannot add or remove skip_logic_protected in interpretation_hints",
    )

    _validate_patch(
        "missingness pair incoherence fails",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_006b",
                    "Final_Grade",
                    "missingness_handling",
                    "retain_with_caution",
                    "Invalid missingness pair fixture.",
                    "Invalid missingness pair fixture.",
                )
            ],
            "Invalid missingness pair fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="missingness_handling must be one of ['no_action_needed'] when missingness_disposition is no_material_missingness",
    )

    _validate_patch(
        "family default non-structural missingness fails",
        _build_patch_envelope(
            [
                _build_patch(
                    "chg_stats_006c",
                    "A1_Q2",
                    "missingness_disposition",
                    "token_missingness_present",
                    "Invalid family-default missingness fixture.",
                    "Invalid family-default missingness fixture.",
                )
            ],
            "Invalid family-default missingness fixture.",
        ),
        stats_original,
        expected_ok=False,
        expected_error_substring="missingness_decision_source may not be family_default for non-structural missingness_disposition token_missingness_present",
    )

    child_hint_patch = _build_patch_envelope(
        [
            _build_patch(
                "chg_stats_007",
                "A1_Q2",
                "structural_transform_hints",
                ["requires_child_table_review"],
                "Invalid post-placement child hint patch.",
                "Invalid post-placement child hint patch.",
            )
        ],
        "Invalid child-placement hint fixture.",
    )
    child_hint_result = _validate_patch(
        "child placement hint fails on finalized child row",
        child_hint_patch,
        stats_original,
        expected_ok=False,
        expected_error_substring="must not contain requires_child_table_review when canonical_modeling_status is child_repeat_member",
    )
    _assert(
        "preview_contract.column_contracts[1].structural_transform_hints" in child_hint_result["validation_error"],
        child_hint_result,
    )
    _assert_apply_fails(
        "apply rejects finalized child hint if validator is bypassed",
        json.dumps(child_hint_patch),
        stats_original,
        "reviewed_contract.column_contracts[1].structural_transform_hints must not contain requires_child_table_review when canonical_modeling_status is child_repeat_member",
    )

    stale_metadata_fixture = _load_fixture_json("reviewer_patch_stale_metadata_after_deletion.json")
    _assert(stale_metadata_fixture["review_flags"], stale_metadata_fixture)
    _assert(stale_metadata_fixture["assumptions"], stale_metadata_fixture)
    repair_prompt = (ROOT / "prompts" / "REPAIR_canonical_contract_reviewer.md").read_text()
    _assert("remove that stale metadata too" in repair_prompt, repair_prompt)

    unwrap_known_artifacts = _load_unwrap_helpers()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "canon_contract.json").write_text(
            json.dumps(
                {
                    "status_code": 200,
                    "body": json.dumps(original),
                }
            )
        )
        (temp_path / "resolve_patch.json").write_text(
            json.dumps(
                {
                    "canon_review_patch": json.dumps(json.loads(single_result["canon_review_patch"])),
                }
            )
        )
        (temp_path / "reviewed_contract.json").write_text(
            json.dumps(
                {
                    "canon_review_json": json.dumps(single_final),
                }
            )
        )
        written_paths = unwrap_known_artifacts(temp_path)
        _assert(len(written_paths) == 6, written_paths)
        payload_contract = json.loads((temp_path / "canon_contract.payload.json").read_text())
        payload_patch = json.loads((temp_path / "canon_review_patch.payload.json").read_text())
        payload_review = json.loads((temp_path / "canon_review.payload.json").read_text())
        _assert(payload_contract["worker"] == "canonical_column_contract_builder", payload_contract)
        _assert(payload_patch["worker"] == "canonical_contract_reviewer", payload_patch)
        _assert(payload_review["worker"] == "canonical_contract_reviewer", payload_review)
        _assert(payload_review["reviewed_contract"]["summary"]["review_applied"] is True, payload_review)
        _assert(payload_review["reviewer_flags"] == payload_review["review_flags"], payload_review)

    print("PASS: canonical contract reviewer smoke checks")


if __name__ == "__main__":
    main()
