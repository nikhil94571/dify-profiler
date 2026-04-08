# Canonical Contract Reviewer Regression Cases

Use these dataset-agnostic cases when editing the canonical-contract-reviewer prompt or repair logic.

## Common Assertions
- Output remains valid against the canonical-contract-reviewer validator.
- `reviewed_contract.column_contracts` preserves row order and row count exactly.
- Every substantive diff appears in `change_log`.
- `change_log[].target_path` is rooted at `/reviewed_contract/...`.
- `reviewed_contract.summary` is recomputed from the final rows.
- Provenance fields remain coherent with the final reviewed values.

## Case 1 - Structural Contradiction Requires Row Correction
Input focus:
- `table_layout_worker_json` places a source column in a child repeat table
- the deterministic contract still marks it as a base field

Expected assertions:
- structural fields are corrected to match the reviewed layout
- the row remains in the same array position
- the edit appears in `change_log`
- `needs_human_review` is conservative when the structural correction is non-trivial

## Case 2 - Raw Artifact Conflict Does Not Override Reviewed Structure
Input focus:
- raw reviewer bundle hints look reference-like
- reviewed layout and reviewed family output keep the row in respondent-linked child structure

Expected assertions:
- no reference-style structural rewrite occurs
- at most, cautionary hints or review flags are preserved
- raw artifact pressure alone does not trigger a structural change

## Case 3 - Family-Default Missingness Spillover
Input focus:
- a family-wide missingness default propagated into multiple sibling rows
- `missingness_worker_json` and `A16` only support structural protection for some of them

Expected assertions:
- unsupported rows are corrected
- protected rows remain protected
- each substantive row correction is logged

## Case 4 - Stale Summary Metrics
Input focus:
- the reviewed contract rows change
- the draft summary counts remain unchanged

Expected assertions:
- summary counts are recomputed from the final reviewed rows
- stale counts are corrected even if no row ordering changed
- summary edits are logged

## Case 5 - Provenance Mismatch
Input focus:
- final reviewed type or missingness values clearly come from a reviewed worker
- the draft still claims a deterministic baseline source

Expected assertions:
- the decision-source fields are corrected
- `applied_sources` is corrected
- provenance-only substantive fixes are still logged

## Case 6 - Excluded Field Carries Stale Semantics
Input focus:
- a row is `excluded_from_outputs`
- the draft leaves a non-blank canonical table name or unsupported semantic note

Expected assertions:
- exclusion-state coherence is restored
- `canonical_table_name` is blank
- unsupported semantics are removed or softened
- the change is logged

## Case 7 - Justified No-Change Review
Input focus:
- reviewed structure, reviewed type, reviewed missingness, semantic context, and summary counts all agree with the draft

Expected assertions:
- `reviewed_contract` is unchanged
- `change_log = []`
- `review_summary.change_count = 0`
- `review_summary.changed_column_count = 0`
