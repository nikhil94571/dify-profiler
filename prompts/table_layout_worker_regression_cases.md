# Table Layout Worker Regression Cases

Use these dataset-agnostic cases when editing the table-layout prompt, schema, or repair prompt.

## Common Assertions
- Output remains valid against the table-layout response schema.
- Reviewed family role is preserved over raw artifact hints unless reviewed family output or semantic context explicitly supports a different structure.
- `child_repeat` + `wide_to_long_family` tables include the derived repeat index in `grain_columns`.
- Accepted-family `table_suggestions` use compact membership fields rather than a full repeated member inventory.
- `column_table_assignments[].why` stays compact.
- Repeated caveats appear at table or global level, not on every family member.
- Output size and completion tokens should drop materially versus the pre-tuning prompt on representative fixtures without changing structural outcomes.

## Case 1 - Accepted Repeated Family, No Review
Input focus:
- `family_worker_json.family_result.family_id = fam_repeat_clean`
- `recommended_table_name = family_repeat_clean`
- `recommended_parent_key = entity_id`
- `recommended_repeat_index_name = item_idx`
- `recommended_family_role = repeated_survey_block`
- `recommended_handling = retain_as_child_table`
- `needs_human_review = false`
- `semantic_context_json = {"status":"skipped","reason":"blank_semantic_input"}`
- raw bundle hints: child-like `A12`, low `A9` noise

Expected assertions:
- table suggestion uses `kind = child_repeat`
- table suggestion uses `build_strategy = wide_to_long_family`
- table suggestion uses `grain_columns = [\"entity_id\", \"item_idx\"]`
- table suggestion uses compact membership fields with preview/count
- `needs_human_review = false`
- no family-specific review flag is required
- `reference_lookup` is not used

## Case 2 - Accepted Repeated Family, Review Required
Input focus:
- `family_worker_json.family_result.family_id = fam_repeat_review`
- `recommended_table_name = family_repeat_review`
- `recommended_parent_key = entity_id`
- `recommended_repeat_index_name = wave`
- `recommended_family_role = repeated_measure_set`
- `recommended_handling = retain_with_review`
- `needs_human_review = true`
- `semantic_context_json = {"status":"skipped","reason":"light_contract_accepted"}`
- raw bundle hints: high `A9` noise, moderate `A14` drift

Expected assertions:
- table suggestion uses `kind = child_repeat`
- table suggestion uses `build_strategy = wide_to_long_family`
- table suggestion uses `grain_columns = [\"entity_id\", \"wave\"]`
- table suggestion uses compact membership fields with preview/count
- `needs_human_review = true`
- at least one family-relevant `review_flags` entry exists
- `reference_lookup` is not used

## Case 3 - Explicit Reference-Like Family
Input focus:
- `family_worker_json.family_result.family_id = fam_answer_key`
- `recommended_table_name = family_answer_key`
- `recommended_parent_key = entity_id`
- `recommended_repeat_index_name = item_idx`
- `recommended_family_role = answer_key_or_reference_block`
- `recommended_handling = retain_with_review`
- `needs_human_review = true`
- semantic context says the block stores canonical answer values for scoring, not respondent observations
- raw bundle hints: reference-like `A12`

Expected assertions:
- table suggestion uses `kind = reference_lookup`
- table suggestion uses `build_strategy = reference_extract`
- `needs_human_review = true`
- at least one linkage or semantics `review_flags` entry exists if respondent linkage remains suspicious

## Case 4 - Raw Layout Conflict Does Not Override Reviewed Repeat
Input focus:
- `family_worker_json.family_result.family_id = fam_repeat_conflict`
- `recommended_table_name = family_repeat_conflict`
- `recommended_parent_key = entity_id`
- `recommended_repeat_index_name = item_idx`
- `recommended_family_role = repeated_survey_block`
- `recommended_handling = retain_with_review`
- `needs_human_review = true`
- `semantic_context_json = {"status":"skipped","reason":"blank_semantic_input"}`
- raw bundle hints: reference-like `A12`, high `A9` noise, high `A14` drift, low value variability

Expected assertions:
- table suggestion uses `kind = child_repeat`
- table suggestion uses `build_strategy = wide_to_long_family`
- table suggestion uses `grain_columns = [\"entity_id\", \"item_idx\"]`
- table suggestion uses compact membership fields with preview/count
- `needs_human_review = true`
- at least one `review_flags` entry captures the conflict between reviewed family structure and raw artifact hints
- `reference_lookup` is not used

## Case 5 - Excluded Source Column Uses Blank Assignment Target
Input focus:
- one known source column is an export-only index column such as `Unnamed: 0`
- reviewed layout keeps the real respondent grain elsewhere
- excluded column should not appear in any output table

Expected assertions:
- the excluded column appears exactly once in `column_table_assignments`
- the row uses `assignment_role = exclude_from_outputs`
- the row uses `assigned_table = \"\"`
- the row does not invent pseudo-table names such as `excluded_from_outputs`
- validator accepts the output

## Case 6 - Unresolved Source Column Uses Blank Assignment Target
Input focus:
- one known source column cannot be placed confidently into any table
- reviewed outputs require the column to remain explicitly unresolved

Expected assertions:
- the unresolved column appears exactly once in `column_table_assignments`
- the row uses `assignment_role = unresolved`
- the row uses `assigned_table = \"\"`
- the row keeps a short explanatory `why`
- validator accepts the output
