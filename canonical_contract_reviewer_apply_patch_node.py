import copy
import json


SUMMARY_COUNT_KEYS = [
    "total_source_columns",
    "included_column_count",
    "excluded_column_count",
    "unresolved_column_count",
    "reviewed_override_count",
    "family_default_count",
    "deterministic_baseline_count",
    "reviewed_type_count",
    "fallback_type_count",
]

# Keep this node self-contained because Dify code nodes cannot reliably import
# repo-local helper modules at runtime.
POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS = {
    "requires_child_table_review",
    "requires_wide_to_long_review",
}

MISSINGNESS_DISPOSITION_TO_ALLOWED_HANDLING = {
    "no_material_missingness": {"no_action_needed"},
    "token_missingness_present": {"retain_with_caution", "review_before_drop"},
    "structurally_valid_missingness": {"protect_from_null_penalty", "retain_with_caution"},
    "partially_structural_missingness": {"retain_with_caution", "review_before_drop"},
    "unexplained_high_missingness": {"review_before_drop", "candidate_drop_review"},
    "mixed_missingness_risk": {"retain_with_caution", "review_before_drop"},
}

SKIP_LOGIC_TRUE_ALLOWED_DISPOSITIONS = {
    "structurally_valid_missingness",
    "partially_structural_missingness",
}

FAMILY_DEFAULT_ALLOWED_MISSINGNESS_DISPOSITIONS = {
    "structurally_valid_missingness",
    "partially_structural_missingness",
}

PATCHABLE_FIELDS = {
    "canonical_modeling_status",
    "canonical_table_name",
    "canonical_assignment_role",
    "source_family_id",
    "recommended_logical_type",
    "recommended_storage_type",
    "transform_actions",
    "structural_transform_hints",
    "interpretation_hints",
    "missingness_disposition",
    "missingness_handling",
    "semantic_meaning",
    "codebook_note",
    "normalization_notes",
    "confidence",
    "needs_human_review",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _parse_json_object(raw_value, field_name):
    _require(isinstance(raw_value, str), f"{field_name} must be a string.")
    _require(raw_value.strip() != "", f"{field_name} is empty.")
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON: {exc}") from exc
    _require(isinstance(parsed, dict), f"{field_name} root must be an object.")
    return parsed


def _build_column_index_map(rows):
    _require(isinstance(rows, list), "canon_contract_json.column_contracts must be an array.")
    index_map = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"canon_contract_json.column_contracts[{index}] must be an object.")
        column = str(row.get("column") or "").strip()
        _require(column != "", f"canon_contract_json.column_contracts[{index}].column must be a non-empty string.")
        _require(column not in index_map, f"canon_contract_json.column_contracts has duplicate column entries for {column}.")
        index_map[column] = index
    return index_map


def _recompute_summary_counts(reviewed_contract):
    summary = reviewed_contract.get("summary")
    rows = reviewed_contract.get("column_contracts")
    _require(isinstance(summary, dict), "reviewed_contract.summary must be an object.")
    _require(isinstance(rows, list), "reviewed_contract.column_contracts must be an array.")

    summary["total_source_columns"] = len(rows)
    summary["included_column_count"] = sum(
        1 for row in rows
        if isinstance(row, dict) and row.get("canonical_modeling_status") not in {"excluded_from_outputs", "unresolved"}
    )
    summary["excluded_column_count"] = sum(
        1 for row in rows
        if isinstance(row, dict) and row.get("canonical_modeling_status") == "excluded_from_outputs"
    )
    summary["unresolved_column_count"] = sum(
        1 for row in rows
        if isinstance(row, dict) and row.get("canonical_modeling_status") == "unresolved"
    )
    summary["reviewed_override_count"] = sum(
        1
        for row in rows
        if isinstance(row, dict) and (
            row.get("type_decision_source") == "reviewed_type_worker"
            or row.get("missingness_decision_source") == "reviewed_missingness_worker"
        )
    )
    summary["family_default_count"] = sum(
        1
        for row in rows
        if isinstance(row, dict) and (
            row.get("type_decision_source") == "family_default"
            or row.get("missingness_decision_source") == "family_default"
        )
    )
    summary["deterministic_baseline_count"] = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and row.get("type_decision_source") == "a17_baseline"
        and row.get("missingness_decision_source") == "a17_baseline"
    )
    summary["reviewed_type_count"] = sum(
        1 for row in rows
        if isinstance(row, dict) and row.get("type_decision_source") == "reviewed_type_worker"
    )
    summary["fallback_type_count"] = sum(
        1 for row in rows
        if isinstance(row, dict) and row.get("type_decision_source") != "reviewed_type_worker"
    )


def _require_row_coherence(row, row_prefix):
    for error in _find_canonical_row_invariant_errors(row, row_prefix):
        _require(False, error)


def _find_canonical_row_invariant_errors(row, row_prefix):
    errors = []

    if row.get("canonical_modeling_status") == "child_repeat_member":
        for hint in row.get("structural_transform_hints") or []:
            if hint in POST_CANONICAL_CHILD_FORBIDDEN_STRUCTURAL_HINTS:
                errors.append(
                    f"{row_prefix}.structural_transform_hints must not contain {hint} "
                    "when canonical_modeling_status is child_repeat_member"
                )

    disposition = row.get("missingness_disposition")
    handling = row.get("missingness_handling")
    decision_source = row.get("missingness_decision_source")
    skip_logic_protected = row.get("skip_logic_protected")

    allowed_handling = MISSINGNESS_DISPOSITION_TO_ALLOWED_HANDLING.get(disposition)
    if allowed_handling and handling not in allowed_handling:
        errors.append(
            f"{row_prefix}.missingness_handling must be one of "
            f"{sorted(allowed_handling)} when missingness_disposition is {disposition}"
        )

    if handling == "protect_from_null_penalty" and skip_logic_protected is not True:
        errors.append(
            f"{row_prefix}.skip_logic_protected must be true when missingness_handling "
            "is protect_from_null_penalty"
        )

    if skip_logic_protected is True and disposition not in SKIP_LOGIC_TRUE_ALLOWED_DISPOSITIONS:
        errors.append(
            f"{row_prefix}.skip_logic_protected can only be true for "
            "structurally_valid_missingness or partially_structural_missingness"
        )

    if decision_source == "family_default" and disposition not in FAMILY_DEFAULT_ALLOWED_MISSINGNESS_DISPOSITIONS:
        errors.append(
            f"{row_prefix}.missingness_decision_source may not be family_default for "
            f"non-structural missingness_disposition {disposition}"
        )

    return errors


def _stamp_review_summary_metadata(reviewed_contract, change_count, changed_column_count):
    summary = reviewed_contract.get("summary")
    rows = reviewed_contract.get("column_contracts") or []
    _require(isinstance(summary, dict), "reviewed_contract.summary must be an object.")
    summary["overview"] = (
        f"Post-review canonical column contract for {len(rows)} columns. "
        f"Deterministic summary counts reflect the final reviewed rows after applying {change_count} "
        f"reviewer change(s) affecting {changed_column_count} column(s)."
    )
    summary["review_applied"] = True
    summary["review_change_count"] = change_count
    summary["review_changed_column_count"] = changed_column_count


def _build_unique_change_id(base_id, seen_ids):
    candidate = base_id
    suffix = 1
    while candidate in seen_ids:
        candidate = f"{base_id}_{suffix}"
        suffix += 1
    seen_ids.add(candidate)
    return candidate


def _build_final_change_log(change_set, original_contract, reviewed_contract):
    change_log = []
    seen_ids = set()
    original_rows = original_contract.get("column_contracts") or []
    reviewed_rows = reviewed_contract.get("column_contracts") or []
    column_index_map = _build_column_index_map(original_rows)

    for item in change_set:
        change_id = _build_unique_change_id(str(item["change_id"]), seen_ids)
        column_name = str(item["column"])
        field_name = str(item["field"])
        row_index = column_index_map[column_name]
        before_value = original_rows[row_index][field_name]
        after_value = reviewed_rows[row_index][field_name]
        target_path = f"/reviewed_contract/column_contracts/{row_index}/{field_name}"
        change_log.append({
            "change_id": change_id,
            "column": column_name,
            "target_path": target_path,
            "before_value": before_value,
            "after_value": after_value,
            "reasoning": str(item.get("reasoning") or ""),
            "justification": str(item.get("justification") or ""),
            "confidence": float(item.get("confidence")),
            "needs_human_review": bool(item.get("needs_human_review")),
        })

    for key in SUMMARY_COUNT_KEYS:
        original_value = original_contract.get("summary", {}).get(key)
        reviewed_value = reviewed_contract.get("summary", {}).get(key)
        if original_value == reviewed_value:
            continue
        change_log.append({
            "change_id": _build_unique_change_id(f"sys_summary_{key}", seen_ids),
            "column": "",
            "target_path": f"/reviewed_contract/summary/{key}",
            "before_value": original_value,
            "after_value": reviewed_value,
            "reasoning": "Summary counts are code-owned and were recomputed after applying the reviewer patch.",
            "justification": "The final reviewed contract must carry deterministic summary counts that match the final row states exactly.",
            "confidence": 1.0,
            "needs_human_review": False,
        })

    return change_log


def _validate_final_invariants(original_contract, reviewed_contract, change_log):
    original_rows = original_contract.get("column_contracts")
    reviewed_rows = reviewed_contract.get("column_contracts")
    _require(isinstance(original_rows, list), "canon_contract_json.column_contracts must be an array.")
    _require(isinstance(reviewed_rows, list), "reviewed_contract.column_contracts must be an array.")
    _require(len(original_rows) == len(reviewed_rows), "reviewed_contract.column_contracts length changed.")

    for index, (before_row, after_row) in enumerate(zip(original_rows, reviewed_rows)):
        before_col = before_row.get("column") if isinstance(before_row, dict) else None
        after_col = after_row.get("column") if isinstance(after_row, dict) else None
        _require(before_col == after_col, f"reviewed_contract.column_contracts order changed at index {index}.")
        if isinstance(after_row, dict):
            _require_row_coherence(after_row, f"reviewed_contract.column_contracts[{index}]")

    _recompute_summary_counts(reviewed_contract)

    expected_change_count = len(change_log)
    expected_changed_column_count = len({
        str(item.get("column") or "").strip()
        for item in change_log
        if str(item.get("column") or "").strip() != ""
    })

    summary = reviewed_contract.get("summary")
    _require(isinstance(summary, dict), "reviewed_contract.summary must be an object.")
    for key in SUMMARY_COUNT_KEYS:
        _require(key in summary, f"reviewed_contract.summary.{key} is required.")

    return expected_change_count, expected_changed_column_count


def main(canon_review_patch: str, canon_contract_json: str):
    patch_envelope = _parse_json_object(canon_review_patch, "canon_review_patch")
    original_contract = _parse_json_object(canon_contract_json, "canon_contract_json")

    _require(
        patch_envelope.get("worker") == "canonical_contract_reviewer",
        "canon_review_patch.worker must be 'canonical_contract_reviewer'.",
    )
    _require(
        original_contract.get("worker") == "canonical_column_contract_builder",
        "canon_contract_json.worker must be 'canonical_column_contract_builder'.",
    )

    reviewed_contract = copy.deepcopy(original_contract)
    change_set = patch_envelope.get("change_set") or []
    _require(isinstance(change_set, list), "canon_review_patch.change_set must be an array.")
    column_index_map = _build_column_index_map(reviewed_contract.get("column_contracts") or [])

    for index, item in enumerate(change_set):
        _require(isinstance(item, dict), f"change_set[{index}] must be an object.")
        column_name = str(item.get("column") or "")
        field_name = str(item.get("field") or "")
        _require(
            column_name in column_index_map,
            f"change_set[{index}].column must resolve against canon_contract_json.column_contracts.",
        )
        _require(field_name in PATCHABLE_FIELDS, f"change_set[{index}].field points to a forbidden field: {field_name}")
        row_index = column_index_map[column_name]
        row = reviewed_contract["column_contracts"][row_index]
        _require(field_name in row, f"change_set[{index}].field does not exist in canon_contract_json.column_contracts[{row_index}]: {field_name}")
        if field_name == "interpretation_hints":
            before_has_skip_hint = "skip_logic_protected" in (row.get("interpretation_hints") or [])
            after_has_skip_hint = "skip_logic_protected" in (item.get("after_value") or [])
            _require(
                before_has_skip_hint == after_has_skip_hint,
                f"change_set[{index}].field cannot add or remove skip_logic_protected in interpretation_hints because that bookkeeping is code-owned.",
            )
        row[field_name] = item.get("after_value")

    _recompute_summary_counts(reviewed_contract)
    change_log = _build_final_change_log(change_set, original_contract, reviewed_contract)
    change_count, changed_column_count = _validate_final_invariants(
        original_contract,
        reviewed_contract,
        change_log,
    )
    _stamp_review_summary_metadata(reviewed_contract, change_count, changed_column_count)

    review_summary = patch_envelope.get("review_summary") or {}
    _require(isinstance(review_summary, dict), "canon_review_patch.review_summary must be an object.")
    overview = str(review_summary.get("overview") or "").strip()
    _require(overview != "", "canon_review_patch.review_summary.overview must not be blank.")
    review_principles = review_summary.get("review_principles") or []
    _require(isinstance(review_principles, list), "canon_review_patch.review_summary.review_principles must be an array.")

    reviewer_flags = patch_envelope.get("review_flags") or []
    reviewer_assumptions = patch_envelope.get("assumptions") or []
    final_envelope = {
        "worker": "canonical_contract_reviewer",
        "review_summary": {
            "overview": overview,
            "change_count": change_count,
            "changed_column_count": changed_column_count,
            "review_principles": review_principles,
        },
        "reviewed_contract": reviewed_contract,
        "change_log": change_log,
        "reviewer_flags": reviewer_flags,
        "reviewer_assumptions": reviewer_assumptions,
        "review_flags": reviewer_flags,
        "assumptions": reviewer_assumptions,
    }

    return {
        "canon_review_json": json.dumps(final_envelope, separators=(",", ":"))
    }
