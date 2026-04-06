import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_validator_main():
    namespace = {}
    code = Path("JSON validators/canonical_contract_reviewer_validator.json").read_text()
    exec(code, namespace)
    return namespace["main"]


def _base_contract():
    return {
        "worker": "canonical_column_contract_builder",
        "summary": {
            "overview": "Reviewed baseline contract fixture.",
            "total_source_columns": 6,
            "included_column_count": 5,
            "excluded_column_count": 1,
            "unresolved_column_count": 0,
            "reviewed_override_count": 4,
            "family_default_count": 2,
            "deterministic_baseline_count": 0,
            "reviewed_type_count": 4,
            "fallback_type_count": 2,
            "key_contract_principles": [
                "Control fields are complete for every contract row; semantic enrichment stays blank unless evidence exists.",
                "Reviewed table layout, semantic context, reviewed specialist outputs, and family defaults outrank the A17 deterministic baseline.",
                "Raw prose is not allowed to directly override structured contract fields in v1.",
            ],
        },
        "column_contracts": [
            {
                "column": "Unnamed: 0",
                "canonical_modeling_status": "excluded_from_outputs",
                "canonical_table_name": "",
                "canonical_assignment_role": "exclude_from_outputs",
                "source_family_id": "",
                "a9_primary_role": "id_key",
                "recommended_logical_type": "mixed_or_ambiguous",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "cast_to_string"],
                "structural_transform_hints": [],
                "interpretation_hints": ["light_contract_override_applied"],
                "missingness_disposition": "no_material_missingness",
                "missingness_handling": "no_action_needed",
                "skip_logic_protected": False,
                "semantic_meaning": "",
                "codebook_note": "",
                "normalization_notes": "Export/index-style numeric column retained only for traceability.",
                "quality_score": 1.0,
                "drift_detected": False,
                "type_decision_source": "reviewed_type_worker",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "a17_baseline",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "A17", "type_transform_worker", "A2", "A9", "A14", "A4"],
                "confidence": 0.76,
                "needs_human_review": False,
            },
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
                "applied_sources": ["table_layout_worker", "A17", "type_transform_worker", "A2", "A9", "A14", "A4"],
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
                "quality_score": 0.993548,
                "drift_detected": False,
                "type_decision_source": "reviewed_type_worker",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "reviewed_missingness_worker",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "A17", "type_transform_worker", "missingness_worker"],
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
                "quality_score": 0.793506,
                "drift_detected": False,
                "type_decision_source": "reviewed_type_worker",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "reviewed_missingness_worker",
                "semantic_decision_source": "unknown",
                "applied_sources": ["table_layout_worker", "A17", "type_transform_worker", "missingness_worker"],
                "confidence": 0.75,
                "needs_human_review": True,
            },
            {
                "column": "Q6Main_cell_groupRow1",
                "canonical_modeling_status": "child_repeat_member",
                "canonical_table_name": "q6_main_cell_group_rows",
                "canonical_assignment_role": "melt_member",
                "source_family_id": "q_6_main_cell_group",
                "a9_primary_role": "measure",
                "recommended_logical_type": "ordinal_category",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "normalize_category_tokens"],
                "structural_transform_hints": [],
                "interpretation_hints": ["skip_logic_protected"],
                "missingness_disposition": "structurally_valid_missingness",
                "missingness_handling": "protect_from_null_penalty",
                "skip_logic_protected": True,
                "semantic_meaning": "Matrix of repeated survey items measuring familiarity.",
                "codebook_note": "",
                "normalization_notes": "Family default propagated across matrix rows.",
                "quality_score": 1.0,
                "drift_detected": False,
                "type_decision_source": "family_default",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "family_default",
                "semantic_decision_source": "family_worker",
                "applied_sources": ["table_layout_worker", "A17", "family_worker.family_result", "family_worker.member_defaults"],
                "confidence": 0.88,
                "needs_human_review": False,
            },
            {
                "column": "Q6Main_cell_groupRow2",
                "canonical_modeling_status": "child_repeat_member",
                "canonical_table_name": "q6_main_cell_group_rows",
                "canonical_assignment_role": "melt_member",
                "source_family_id": "q_6_main_cell_group",
                "a9_primary_role": "measure",
                "recommended_logical_type": "ordinal_category",
                "recommended_storage_type": "string",
                "transform_actions": ["trim_whitespace", "normalize_category_tokens"],
                "structural_transform_hints": [],
                "interpretation_hints": ["skip_logic_protected"],
                "missingness_disposition": "structurally_valid_missingness",
                "missingness_handling": "protect_from_null_penalty",
                "skip_logic_protected": True,
                "semantic_meaning": "Matrix of repeated survey items measuring familiarity.",
                "codebook_note": "",
                "normalization_notes": "Family default propagated across matrix rows.",
                "quality_score": 0.993548,
                "drift_detected": False,
                "type_decision_source": "family_default",
                "structure_decision_source": "table_layout_worker",
                "missingness_decision_source": "family_default",
                "semantic_decision_source": "family_worker",
                "applied_sources": ["table_layout_worker", "A17", "family_worker.family_result", "family_worker.member_defaults"],
                "confidence": 0.88,
                "needs_human_review": False,
            },
        ],
        "global_value_rules": [],
        "review_flags": [],
        "assumptions": [],
    }


def _stale_summary_contract():
    contract = _base_contract()
    contract["summary"]["overview"] = "Synthesized a deterministic canonical column contract for 6 columns by merging structure decisions, semantic context, reviewed overrides, family defaults, and the A17 baseline layer."
    contract["summary"]["fallback_type_count"] = 5
    return contract


def _recompute_summary(contract):
    rows = contract["column_contracts"]
    contract["summary"]["total_source_columns"] = len(rows)
    contract["summary"]["included_column_count"] = sum(
        1 for row in rows if row["canonical_modeling_status"] not in {"excluded_from_outputs", "unresolved"}
    )
    contract["summary"]["excluded_column_count"] = sum(
        1 for row in rows if row["canonical_modeling_status"] == "excluded_from_outputs"
    )
    contract["summary"]["unresolved_column_count"] = sum(
        1 for row in rows if row["canonical_modeling_status"] == "unresolved"
    )
    contract["summary"]["reviewed_override_count"] = sum(
        1
        for row in rows
        if row["type_decision_source"] == "reviewed_type_worker"
        or row["missingness_decision_source"] == "reviewed_missingness_worker"
    )
    contract["summary"]["family_default_count"] = sum(
        1
        for row in rows
        if row["type_decision_source"] == "family_default"
        or row["missingness_decision_source"] == "family_default"
    )
    contract["summary"]["deterministic_baseline_count"] = sum(
        1
        for row in rows
        if row["type_decision_source"] == "a17_baseline"
        and row["missingness_decision_source"] == "a17_baseline"
    )
    contract["summary"]["reviewed_type_count"] = sum(
        1 for row in rows if row["type_decision_source"] == "reviewed_type_worker"
    )
    contract["summary"]["fallback_type_count"] = sum(
        1 for row in rows if row["type_decision_source"] != "reviewed_type_worker"
    )


def _decode_pointer_segment(value):
    return str(value).replace("~1", "/").replace("~0", "~")


def _resolve_pointer(doc, pointer):
    current = doc
    if pointer == "":
        return current
    for raw_part in pointer.split("/")[1:]:
        part = _decode_pointer_segment(raw_part)
        if isinstance(current, dict):
            current = current[part]
        else:
            current = current[int(part)]
    return current


def _set_pointer(doc, pointer, value):
    parts = pointer.split("/")[1:]
    current = doc
    for raw_part in parts[:-1]:
        part = _decode_pointer_segment(raw_part)
        if isinstance(current, dict):
            current = current[part]
        else:
            current = current[int(part)]
    last = _decode_pointer_segment(parts[-1])
    if isinstance(current, dict):
        current[last] = value
    else:
        current[int(last)] = value


def _build_change(original_contract, reviewed_contract, target_path, change_id, column, reasoning, justification, confidence=0.9, needs_human_review=False):
    contract_pointer = target_path[len("/reviewed_contract"):] or ""
    before_value = _resolve_pointer(original_contract, contract_pointer) if contract_pointer else original_contract
    after_value = _resolve_pointer({"reviewed_contract": reviewed_contract}, target_path)
    return {
        "change_id": change_id,
        "column": column,
        "target_path": target_path,
        "before_value": before_value,
        "after_value": after_value,
        "reasoning": reasoning,
        "justification": justification,
        "confidence": confidence,
        "needs_human_review": needs_human_review,
    }


def _build_envelope(reviewed_contract, change_log, overview):
    changed_columns = {
        str(item.get("column") or "").strip()
        for item in change_log
        if str(item.get("column") or "").strip() != ""
    }
    return {
        "worker": "canonical_contract_reviewer",
        "review_summary": {
            "overview": overview,
            "change_count": len(change_log),
            "changed_column_count": len(changed_columns),
            "review_principles": [
                "Preserve deterministic contract structure and only edit evidence-supported fields.",
                "Every substantive edit must be ledgered against the reviewed full contract.",
            ],
        },
        "reviewed_contract": reviewed_contract,
        "change_log": change_log,
        "review_flags": [],
        "assumptions": [],
    }


def _assert_validation(name, reviewer_output, original_contract, expected_ok, expected_error_substring=""):
    validator = _load_validator_main()
    raw_output = reviewer_output if isinstance(reviewer_output, str) else json.dumps(reviewer_output)
    result = validator(raw_output, json.dumps(original_contract))
    actual_ok = result["validation_ok"] == "true"
    if actual_ok != expected_ok:
        raise AssertionError(f"{name} expected validation_ok={expected_ok}, got {result}")
    if expected_error_substring and expected_error_substring not in result["validation_error"]:
        raise AssertionError(f"{name} expected error containing {expected_error_substring!r}, got {result}")
    print(f"PASS: {name}")


def main():
    original = _base_contract()

    unchanged_reviewed = deepcopy(original)
    _assert_validation(
        "unchanged contract passes",
        _build_envelope(unchanged_reviewed, [], "No evidence-supported edits were required."),
        original,
        expected_ok=True,
    )

    single_reviewed = deepcopy(original)
    _set_pointer(single_reviewed, "/column_contracts/3/needs_human_review", False)
    _recompute_summary(single_reviewed)
    single_change = _build_change(
        original,
        single_reviewed,
        "/reviewed_contract/column_contracts/3/needs_human_review",
        "chg_001",
        "Q2_Other_Other__please_specify",
        "The reviewed row already has explicit skip-logic evidence and no remaining contradiction that requires manual escalation.",
        "This is a reviewer-only calibration change; the row stays structurally protected but no longer needs forced human review.",
        confidence=0.78,
    )
    _assert_validation(
        "single-row field correction passes",
        _build_envelope(single_reviewed, [single_change], "Lowered one human-review flag after contract coherence review."),
        original,
        expected_ok=True,
    )

    stale_original = _stale_summary_contract()
    summary_reviewed = deepcopy(stale_original)
    _recompute_summary(summary_reviewed)
    summary_change = _build_change(
        stale_original,
        summary_reviewed,
        "/reviewed_contract/summary/fallback_type_count",
        "chg_002",
        "",
        "The draft metric was stale after prior contract edits.",
        "The reviewed contract must carry recomputed summary counts derived from the final row set.",
        confidence=0.99,
    )
    _assert_validation(
        "summary-only correction passes",
        _build_envelope(summary_reviewed, [summary_change], "Recomputed stale summary metrics from the unchanged reviewed row set."),
        stale_original,
        expected_ok=True,
    )

    multi_reviewed = deepcopy(original)
    for row_index in [4, 5]:
        _set_pointer(multi_reviewed, f"/column_contracts/{row_index}/interpretation_hints", [])
        _set_pointer(multi_reviewed, f"/column_contracts/{row_index}/missingness_disposition", "no_material_missingness")
        _set_pointer(multi_reviewed, f"/column_contracts/{row_index}/missingness_handling", "no_action_needed")
        _set_pointer(multi_reviewed, f"/column_contracts/{row_index}/skip_logic_protected", False)
    _recompute_summary(multi_reviewed)
    multi_changes = []
    for row_index, column in [(4, "Q6Main_cell_groupRow1"), (5, "Q6Main_cell_groupRow2")]:
        for change_id_suffix, field in [
            ("a", "interpretation_hints"),
            ("b", "missingness_disposition"),
            ("c", "missingness_handling"),
            ("d", "skip_logic_protected"),
        ]:
            multi_changes.append(
                _build_change(
                    original,
                    multi_reviewed,
                    f"/reviewed_contract/column_contracts/{row_index}/{field}",
                    f"chg_00{row_index}{change_id_suffix}",
                    column,
                    "Family-level missingness protection was broader than the reviewed evidence supported for this row.",
                    "The row keeps its family typing but no longer carries inherited skip-logic protection without explicit row-level evidence.",
                    confidence=0.82,
                    needs_human_review=(field == "skip_logic_protected"),
                )
            )
    _assert_validation(
        "multi-field row correction passes",
        _build_envelope(multi_reviewed, multi_changes, "Narrowed over-propagated family missingness defaults on two matrix rows."),
        original,
        expected_ok=True,
    )

    missing_change_log = _build_envelope(deepcopy(original), [], "Invalid fixture missing change_log.")
    del missing_change_log["change_log"]
    _assert_validation(
        "missing change_log fails",
        missing_change_log,
        original,
        expected_ok=False,
        expected_error_substring="Missing required top-level key: change_log",
    )

    unlogged_diff_reviewed = deepcopy(original)
    _set_pointer(unlogged_diff_reviewed, "/column_contracts/0/recommended_logical_type", "identifier")
    _recompute_summary(unlogged_diff_reviewed)
    _assert_validation(
        "unlogged substantive diff fails",
        _build_envelope(unlogged_diff_reviewed, [], "Invalid fixture with unlogged edit."),
        original,
        expected_ok=False,
        expected_error_substring="Unlogged substantive diff",
    )

    reordered_reviewed = deepcopy(original)
    reordered_reviewed["column_contracts"][4], reordered_reviewed["column_contracts"][5] = (
        reordered_reviewed["column_contracts"][5],
        reordered_reviewed["column_contracts"][4],
    )
    _recompute_summary(reordered_reviewed)
    reorder_change = {
        "change_id": "chg_010",
        "column": "",
        "target_path": "/reviewed_contract/column_contracts",
        "before_value": original["column_contracts"],
        "after_value": reordered_reviewed["column_contracts"],
        "reasoning": "Invalid fixture.",
        "justification": "Invalid fixture.",
        "confidence": 0.5,
        "needs_human_review": True,
    }
    _assert_validation(
        "reordered column_contracts fails",
        _build_envelope(reordered_reviewed, [reorder_change], "Invalid row reorder."),
        original,
        expected_ok=False,
        expected_error_substring="reviewed_contract.column_contracts order changed",
    )

    stale_before_reviewed = deepcopy(original)
    _set_pointer(stale_before_reviewed, "/summary/fallback_type_count", 1)
    _recompute_summary(stale_before_reviewed)
    stale_before_change = _build_change(
        original,
        stale_before_reviewed,
        "/reviewed_contract/summary/fallback_type_count",
        "chg_011",
        "",
        "Invalid fixture.",
        "Invalid fixture.",
    )
    stale_before_change["before_value"] = 999
    _assert_validation(
        "stale before_value fails",
        _build_envelope(stale_before_reviewed, [stale_before_change], "Invalid stale before_value."),
        original,
        expected_ok=False,
        expected_error_substring="before_value must match the original contract value",
    )

    after_mismatch_reviewed = deepcopy(original)
    _set_pointer(after_mismatch_reviewed, "/summary/fallback_type_count", 1)
    _recompute_summary(after_mismatch_reviewed)
    after_mismatch_change = _build_change(
        original,
        after_mismatch_reviewed,
        "/reviewed_contract/summary/fallback_type_count",
        "chg_012",
        "",
        "Invalid fixture.",
        "Invalid fixture.",
    )
    after_mismatch_change["after_value"] = 999
    _assert_validation(
        "after_value mismatch fails",
        _build_envelope(after_mismatch_reviewed, [after_mismatch_change], "Invalid after_value."),
        original,
        expected_ok=False,
        expected_error_substring="after_value must match the reviewed contract value",
    )

    invalid_nested = _build_envelope(deepcopy(original), [], "Invalid nested contract.")
    invalid_nested["reviewed_contract"]["worker"] = "wrong_worker"
    _assert_validation(
        "invalid nested contract fails",
        invalid_nested,
        original,
        expected_ok=False,
        expected_error_substring="reviewed_contract.worker must be 'canonical_column_contract_builder'",
    )

    _assert_validation(
        "malformed reviewer JSON fails",
        "{\"worker\":\"canonical_contract_reviewer\"",
        original,
        expected_ok=False,
        expected_error_substring="reviewer_output is not valid JSON",
    )

    print("PASS: canonical contract reviewer smoke checks")


if __name__ == "__main__":
    main()
