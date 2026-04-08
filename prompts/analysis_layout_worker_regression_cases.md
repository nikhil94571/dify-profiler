# Analysis Layout Worker Regression Cases

Use these dataset-agnostic cases when editing the analysis-layout prompt or repair logic.

## Common Assertions
- Output remains valid against the analysis-layout validator.
- Canonical table placement is preserved; this worker only proposes analysis-ready outputs.
- `wave_column` is non-blank and present in `grain_columns` when `build_strategy = merge_sibling_families_with_wave`.
- `pass_through` uses exactly one source canonical table.
- Structural missingness is not silently converted into incorrect-answer treatment.
- Empty output is allowed when no justified analysis-ready output exists.

## Case 1 - Merge Sibling Families Into One Long Response Table
Input focus:
- `table_layout_worker_json` contains sibling canonical child tables for the same instrument
- `family_worker_json` marks both as the same repeated survey or repeated measure family across waves
- semantic context supports wave-style interpretation

Expected assertions:
- one `analysis_table_suggestions` row uses `kind = long_response_table`
- `build_strategy = merge_sibling_families_with_wave`
- both source tables appear in `source_canonical_tables`
- both family ids appear in `included_family_ids`
- `wave_column` is non-empty and appears in `grain_columns`

## Case 2 - Similar Answer Codes Do Not Justify Merge
Input focus:
- two canonical child tables share answer-code shapes
- reviewed family semantics show they are different instruments

Expected assertions:
- they are not merged into one `long_response_table`
- any shared downstream table is an analysis mart or later aggregation only if justified
- no merge is proposed solely because the response codes look similar

## Case 3 - Answer Key Scoring Flow
Input focus:
- one canonical table is a response table
- one canonical table is an answer-key or lookup table
- reviewed evidence supports scoring semantics

Expected assertions:
- at least one derivation uses `derivation_kind = answer_key_scoring`
- `join_keys` are explicit
- `null_handling_policy` is explicit
- output does not hide scoring logic inside vague prose

## Case 4 - Structural Missingness Must Be Preserved
Input focus:
- `missingness_worker_json` and `A16` show skip-logic protection for response rows
- scoring is otherwise plausible

Expected assertions:
- derivation uses `null_handling_policy = preserve_structural_missing` or `needs_review`
- derivation does not use `treat_as_incorrect` without strong evidence
- at least one review flag exists when the scoring treatment remains ambiguous

## Case 5 - Pass-Through Reference Table
Input focus:
- a canonical reference table should remain available downstream
- no derivation is required

Expected assertions:
- `analysis_table_suggestions` uses `kind = reference_passthrough`
- `build_strategy = pass_through`
- `source_canonical_tables` contains exactly one table

## Case 6 - No Justified Analysis Output
Input focus:
- canonical outputs are already complete for downstream use
- no sibling merge, scoring flow, or analysis mart is well supported

Expected assertions:
- `analysis_table_suggestions` may be empty
- `derivations` may be empty
- `summary.overview` explicitly states that no justified analysis-ready outputs were proposed
