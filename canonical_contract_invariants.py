from copy import deepcopy


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


def find_row_invariant_errors(row, row_prefix, *, global_token_missing_placeholders_detected=None):
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

    if global_token_missing_placeholders_detected is False and disposition == "token_missingness_present":
        errors.append(
            f"{row_prefix}.missingness_disposition must not be token_missingness_present "
            "when global_contract.token_missing_placeholders_detected is false"
        )

    if decision_source == "family_default" and disposition not in FAMILY_DEFAULT_ALLOWED_MISSINGNESS_DISPOSITIONS:
        errors.append(
            f"{row_prefix}.missingness_decision_source may not be family_default for "
            f"non-structural missingness_disposition {disposition}"
        )

    return errors


def normalize_missingness_decision(decision, *, base_decision=None, global_token_missing_placeholders_detected=None):
    normalized = deepcopy(decision)
    notes = []

    disposition = normalized.get("missingness_disposition")
    handling = normalized.get("missingness_handling")
    decision_source = normalized.get("missingness_decision_source")

    if (
        decision_source == "family_default"
        and disposition not in FAMILY_DEFAULT_ALLOWED_MISSINGNESS_DISPOSITIONS
        and isinstance(base_decision, dict)
    ):
        normalized = deepcopy(base_decision)
        notes.append("reverted_invalid_family_default_missingness")
        disposition = normalized.get("missingness_disposition")
        handling = normalized.get("missingness_handling")
        decision_source = normalized.get("missingness_decision_source")

    if (
        decision_source == "family_default"
        and global_token_missing_placeholders_detected is False
        and disposition == "token_missingness_present"
        and isinstance(base_decision, dict)
        and base_decision.get("missingness_disposition") != "token_missingness_present"
    ):
        normalized = deepcopy(base_decision)
        notes.append("reverted_token_missingness_without_global_support")
        disposition = normalized.get("missingness_disposition")
        handling = normalized.get("missingness_handling")

    allowed_handling = MISSINGNESS_DISPOSITION_TO_ALLOWED_HANDLING.get(disposition)
    if allowed_handling and handling not in allowed_handling and len(allowed_handling) == 1:
        normalized["missingness_handling"] = next(iter(allowed_handling))
        handling = normalized["missingness_handling"]
        notes.append("coerced_missingness_handling_to_single_allowed_value")

    if normalized.get("missingness_disposition") == "no_material_missingness":
        if normalized.get("missingness_handling") != "no_action_needed":
            normalized["missingness_handling"] = "no_action_needed"
            notes.append("coerced_no_material_missingness_handling")
        if normalized.get("skip_logic_protected") is not False:
            normalized["skip_logic_protected"] = False
            notes.append("cleared_skip_logic_for_no_material_missingness")

    if normalized.get("missingness_handling") == "protect_from_null_penalty":
        if normalized.get("skip_logic_protected") is not True:
            normalized["skip_logic_protected"] = True
            notes.append("forced_skip_logic_for_protect_from_null_penalty")

    errors = find_row_invariant_errors(
        {
            "canonical_modeling_status": normalized.get("canonical_modeling_status"),
            "structural_transform_hints": normalized.get("structural_transform_hints") or [],
            "missingness_disposition": normalized.get("missingness_disposition"),
            "missingness_handling": normalized.get("missingness_handling"),
            "missingness_decision_source": normalized.get("missingness_decision_source"),
            "skip_logic_protected": normalized.get("skip_logic_protected"),
        },
        "row",
        global_token_missing_placeholders_detected=global_token_missing_placeholders_detected,
    )

    return normalized, notes, errors
