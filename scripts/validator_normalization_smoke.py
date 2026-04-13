#!/usr/bin/env python3

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VALIDATORS = ROOT / "JSON validators"


def _load_validator_main(filename):
    namespace = {}
    code = (VALIDATORS / filename).read_text()
    exec(code, namespace)
    return namespace["main"]


def _load_apply_main():
    namespace = {}
    code = (ROOT / "canonical_contract_reviewer_apply_patch_node.py").read_text()
    exec(code, namespace)
    return namespace["main"]


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _assert_ok(result, output_key):
    _assert(result["validation_ok"] == "true", f"Expected success, got {result}")
    _assert(result[output_key], f"Expected normalized output in {output_key}, got {result}")


def _assert_fail(result, needle):
    _assert(result["validation_ok"] == "false", f"Expected failure, got {result}")
    _assert(needle in result["validation_error"], f"Expected {needle!r} in {result}")


def _run_grain_case():
    main = _load_validator_main("grain_validator.json")
    payload = {
        "recommended_primary_grain": {
            "description": "One row per respondent",
            "grain_type": "single_column",
            "keys": ["RespondentId"],
            "confidence_score": 0.95,
            "justification": "Unique identifier",
        },
        "review_questions": {
            "grain_confirmation": {
                "prompt": "Confirm grain",
                "recommended_answer": "accept",
                "why_it_matters": "Required",
                "answer_type": "enum",
            },
            "family_confirmation": {
                "prompt": "Confirm family",
                "recommended_answer": "accept",
                "why_it_matters": "Required",
                "answer_type": "enum",
            },
            "index_drop_confirmation": {
                "prompt": "Drop index?",
                "recommended_answer": "yes",
                "why_it_matters": "Required",
                "answer_type": "boolean",
            },
        },
        "user_inputs_requested": {
            "global_renaming_instructions": {"input_type": "free_text", "required": False, "purpose": "Optional"},
            "missed_family_information": {"input_type": "free_text", "required": False, "purpose": "Optional"},
            "free_text_override_instructions": {"input_type": "free_text", "required": False, "purpose": "Optional"},
        },
        "diagnostics": {
            "encoding_hint": "unknown",
            "encoding_justification": "Smoke test",
            "recommended_next_action": "Review",
        },
        "reasoning": "Smoke test reasoning",
    }
    result = main(json.dumps(payload))
    _assert_ok(result, "grain_worker_json")
    normalized = json.loads(result["grain_worker_json"])
    _assert(normalized["candidate_reference_tables"] == [], result)
    _assert(normalized["review_questions"]["grain_confirmation"]["allowed_answers"] == [], result)


def _run_semantic_case():
    main = _load_validator_main("semantic_context_validator.json")
    payload = {
        "worker": "semantic_context_interpreter",
        "summary": {"overview": "Smoke test"},
        "dataset_context": {
            "dataset_purpose": "Test",
            "row_meaning_notes": "One row per respondent",
        },
        "important_variables": [],
    }
    result = main(json.dumps(payload))
    _assert_ok(result, "semantic_context_json")
    normalized = json.loads(result["semantic_context_json"])
    _assert(normalized["summary"]["key_points"] == [], result)
    _assert(normalized["codebook_hints"] == [], result)


def _run_type_cases():
    main = _load_validator_main("type_validator.json")
    good_payload = {
        "worker": "type_value_specialist",
        "summary": {"overview": "Smoke test"},
        "column_decisions": [
            {
                "column": "RespondentId",
                "recommended_logical_type": "identifier",
                "recommended_storage_type": "string",
                "normalization_notes": "Keep as string identifier",
                "confidence": 0.95,
                "reasoning": "Primary key",
                "skip_logic_protected": False,
                "needs_human_review": False,
            }
        ],
    }
    result = main(json.dumps(good_payload), json.dumps(["RespondentId"]))
    _assert_ok(result, "type_transform_worker_json")
    normalized = json.loads(result["type_transform_worker_json"])
    row = normalized["column_decisions"][0]
    _assert(row["transform_actions"] == [], result)
    _assert(row["structural_transform_hints"] == [], result)
    _assert(row["interpretation_hints"] == [], result)

    fail_result = main(json.dumps(good_payload), json.dumps(["RespondentId", "OtherCol"]))
    _assert_fail(fail_result, "column_decisions is missing reviewed columns")


def _run_missingness_cases():
    main = _load_validator_main("missingness_validator.json")
    good_payload = {
        "worker": "missingness_structural_validity_specialist",
        "summary": {"overview": "Smoke test"},
        "column_decisions": [
            {
                "column": "A1_Q1",
                "missingness_disposition": "structurally_valid_missingness",
                "structural_validity": "confirmed_structural",
                "recommended_handling": "protect_from_null_penalty",
                "normalization_notes": "Protected by skip logic",
                "reasoning": "Direct trigger evidence",
                "confidence": 0.95,
                "skip_logic_protected": True,
                "needs_human_review": False,
            }
        ],
        "global_contract": {
            "token_missing_placeholders_detected": False,
            "notes": "No dataset-wide token placeholder pattern detected in the fixture.",
        },
    }
    result = main(json.dumps(good_payload))
    _assert_ok(result, "missingness_worker_json")
    normalized = json.loads(result["missingness_worker_json"])
    _assert(normalized["summary"]["key_patterns"] == [], result)
    _assert(normalized["column_decisions"][0]["trigger_columns"] == [], result)

    bad_payload = json.loads(json.dumps(good_payload))
    bad_payload["column_decisions"].append(json.loads(json.dumps(bad_payload["column_decisions"][0])))
    fail_result = main(json.dumps(bad_payload))
    _assert_fail(fail_result, "Duplicate column_decisions entry for column: A1_Q1")

    missing_global_contract = json.loads(json.dumps(good_payload))
    del missing_global_contract["global_contract"]
    fail_result = main(json.dumps(missing_global_contract))
    _assert_fail(fail_result, "Missing required top-level key: global_contract")


def _run_family_case():
    main = _load_validator_main("family_validator.json")
    payload = {
        "worker": "family_specialist",
        "family_id": "f1",
        "family_result": {
            "family_id": "f1",
            "recommended_table_name": "f1_rows",
            "recommended_parent_key": "RespondentId",
            "recommended_repeat_index_name": "q",
            "recommended_family_role": "repeated_survey_block",
            "recommended_handling": "retain_as_child_table",
            "member_semantics_notes": "Survey block",
            "confidence": 0.9,
            "reasoning": "Wide repeated family",
            "needs_human_review": False,
        },
        "member_defaults": {
            "recommended_logical_type": "ordinal_category",
        },
    }
    result = main(json.dumps(payload))
    _assert_ok(result, "family_worker_item_json")
    normalized = json.loads(result["family_worker_item_json"])
    _assert(normalized["member_defaults"]["transform_actions"] == [], result)
    _assert(normalized["review_flags"] == [], result)

    invalid_missingness_payload = json.loads(json.dumps(payload))
    invalid_missingness_payload["member_defaults"] = {
        "missingness_disposition": "token_missingness_present",
        "missingness_handling": "retain_with_caution",
        "skip_logic_protected": False,
    }
    fail_result = main(json.dumps(invalid_missingness_payload))
    _assert_fail(fail_result, "member_defaults.missingness_disposition must be one of")


def _run_table_layout_case():
    main = _load_validator_main("table_layout_validator.json")
    payload = {
        "worker": "table_layout_specialist",
        "summary": {
            "overview": "Smoke test",
            "recommended_model_shape": "single_base_table",
        },
        "table_suggestions": [
            {
                "table_name": "base_rows",
                "kind": "base_entity",
                "source_basis": {"kind": "primary_grain"},
                "grain_columns": ["RespondentId"],
                "build_strategy": "direct_select",
                "confidence": 0.95,
                "reasoning": "Base table",
                "needs_human_review": False,
            }
        ],
        "column_table_assignments": [
            {
                "column": "RespondentId",
                "assigned_table": "base_rows",
                "assignment_role": "base_key",
                "why": "Primary key",
            }
        ],
    }
    result = main(json.dumps(payload), json.dumps(["RespondentId"]))
    _assert_ok(result, "table_layout_worker_json")
    normalized = json.loads(result["table_layout_worker_json"])
    table = normalized["table_suggestions"][0]
    _assert(table["excluded_columns"] == [], result)
    _assert(table["parent_key"] == [], result)
    _assert(normalized["column_table_assignments"][0]["source_family_id"] == "", result)


def _run_analysis_layout_case():
    main = _load_validator_main("analysis_layout_validator.json")
    payload = {
        "worker": "analysis_layout_specialist",
        "summary": {"overview": "Smoke test"},
        "analysis_table_suggestions": [
            {
                "table_name": "scores",
                "kind": "score_table",
                "source_canonical_tables": ["quiz_rows"],
                "grain_columns": ["RespondentId"],
                "build_strategy": "aggregate_scores",
                "value_columns": ["score"],
                "confidence": 0.9,
                "reasoning": "Aggregate scores",
                "needs_human_review": False,
            }
        ],
    }
    result = main(json.dumps(payload))
    _assert_ok(result, "analysis_layout_worker_json")
    normalized = json.loads(result["analysis_layout_worker_json"])
    _assert(normalized["summary"]["analysis_layout_principles"] == [], result)
    _assert(normalized["derivations"] == [], result)
    _assert(normalized["analysis_table_suggestions"][0]["included_family_ids"] == [], result)
    _assert(normalized["analysis_table_suggestions"][0]["wave_column"] == "", result)


def _canonical_contract_payload():
    return {
        "worker": "canonical_column_contract_builder",
        "summary": {
            "overview": "Smoke test",
            "key_contract_principles": ["Keep stable structure"],
            "total_source_columns": 1,
            "included_column_count": 1,
            "excluded_column_count": 0,
            "unresolved_column_count": 0,
            "reviewed_override_count": 0,
            "family_default_count": 0,
            "deterministic_baseline_count": 1,
            "reviewed_type_count": 0,
            "fallback_type_count": 1,
        },
        "column_contracts": [
            {
                "column": "RespondentId",
                "canonical_modeling_status": "base_field",
                "canonical_table_name": "base_rows",
                "canonical_assignment_role": "base_key",
                "recommended_logical_type": "identifier",
                "recommended_storage_type": "string",
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "confidence": 0.95,
                "needs_human_review": False,
                "type_decision_source": "a17_baseline",
                "structure_decision_source": "light_contract_fallback",
                "missingness_decision_source": "a17_baseline",
                "semantic_decision_source": "unknown",
            }
        ],
        "global_value_rules": [],
        "review_flags": [],
        "assumptions": [],
    }


def _run_canonical_contract_case():
    if not (VALIDATORS / "canonical_column_contract_validator.json").exists():
        print("SKIP: canonical_column_contract_validator.json not present")
        return
    main = _load_validator_main("canonical_column_contract_validator.json")
    payload = _canonical_contract_payload()
    result = main(
        json.dumps(payload),
        json.dumps(["RespondentId"]),
        json.dumps(
            {
                "worker": "missingness_structural_validity_specialist",
                "summary": {"overview": "Fixture", "key_patterns": []},
                "column_decisions": [],
                "global_contract": {
                    "token_missing_placeholders_detected": False,
                    "notes": "No dataset-wide token placeholder pattern detected in the fixture.",
                },
                "global_findings": [],
                "review_flags": [],
                "assumptions": [],
            }
        ),
    )
    _assert_ok(result, "canonical_column_contract_json")
    normalized = json.loads(result["canonical_column_contract_json"])
    row = normalized["column_contracts"][0]
    _assert(row["transform_actions"] == [], result)
    _assert(row["source_family_id"] == "", result)
    _assert(normalized["summary"]["total_source_columns"] == 1, result)


def _run_reviewer_case():
    main = _load_validator_main("canonical_contract_reviewer_validator.json")
    apply_main = _load_apply_main()
    original_contract = _canonical_contract_payload()
    original_contract["column_contracts"][0].update({
        "source_family_id": "",
        "a9_primary_role": "",
        "transform_actions": [],
        "structural_transform_hints": [],
        "interpretation_hints": [],
        "applied_sources": [],
    })
    reviewer_payload = {
        "worker": "canonical_contract_reviewer",
        "review_summary": {
            "overview": "No changes",
        },
        "change_set": [],
    }
    result = main(json.dumps(reviewer_payload), json.dumps(original_contract))
    _assert_ok(result, "canon_review_patch")
    normalized_patch = json.loads(result["canon_review_patch"])
    _assert(normalized_patch["review_summary"]["review_principles"] == [], result)

    final_result = apply_main(result["canon_review_patch"], json.dumps(original_contract))
    final_envelope = json.loads(final_result["canon_review_json"])
    _assert(final_envelope["review_summary"]["change_count"] == 0, final_result)
    _assert(final_envelope["change_log"] == [], final_result)
    _assert(final_envelope["reviewed_contract"]["summary"]["review_applied"] is True, final_result)
    _assert(final_envelope["reviewer_flags"] == final_envelope["review_flags"], final_result)
    _assert(final_envelope["reviewer_assumptions"] == final_envelope["assumptions"], final_result)


def main():
    _run_grain_case()
    _run_semantic_case()
    _run_type_cases()
    _run_missingness_cases()
    _run_family_case()
    _run_table_layout_case()
    _run_analysis_layout_case()
    _run_canonical_contract_case()
    _run_reviewer_case()
    print("PASS: validator normalization smoke")


if __name__ == "__main__":
    main()
