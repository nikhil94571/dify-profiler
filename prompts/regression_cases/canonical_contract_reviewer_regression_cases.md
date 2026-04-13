# Canonical Contract Reviewer Regression Cases

Use these dataset-agnostic cases when editing the patch-only reviewer prompt, repair prompt, validator, or deterministic apply node.

## Common Assertions
- Reviewer output remains valid against the patch validator.
- `change_set[]` identifies edits by `column` plus `field`; the LLM does not emit row indices.
- Every patch entry is row-level and leaf-level.
- Frozen/code-owned fields are never patched:
  - `skip_logic_protected`
  - `type_decision_source`
  - `structure_decision_source`
  - `missingness_decision_source`
  - `semantic_decision_source`
  - `applied_sources`
- After deterministic apply, `reviewed_contract.column_contracts` preserves row order and row count exactly.
- After deterministic apply, `reviewed_contract.summary` numeric count fields are recomputed from the final rows.
- After deterministic apply, `reviewed_contract.summary.review_applied = true` and the summary carries review-stage metadata (`review_change_count`, `review_changed_column_count`).
- After deterministic apply, `change_log` contains reviewer patches plus any synthetic summary-count corrections.
- After deterministic apply, reviewer metadata is available under `reviewer_flags` / `reviewer_assumptions`, and the legacy top-level aliases `review_flags` / `assumptions` match exactly.

## Case 1 - Structural Contradiction Requires Row Correction
Input focus:
- `table_layout_worker_json` places a source column in a child repeat table
- the deterministic contract still marks it as a base field

Expected assertions:
- reviewer emits only row-level structural patches
- no full contract is returned
- after deterministic apply, the row is corrected and order is preserved

## Case 2 - Raw Artifact Conflict Does Not Override Reviewed Structure
Input focus:
- raw reviewer-bundle hints look reference-like
- reviewed layout and reviewed family output keep the row in respondent-linked child structure

Expected assertions:
- no structural patch is emitted
- raw artifact pressure alone does not trigger a change

## Case 3 - Family-Default Missingness Spillover
Input focus:
- a family-wide missingness default propagated into multiple sibling rows
- `missingness_worker_json` and `A16` only support structural protection for some of them

Expected assertions:
- reviewer emits only the affected row-field patches
- after deterministic apply, unsupported rows are corrected
- protected rows remain protected

## Case 4 - Provenance Mismatch Is Not Reviewer-Owned
Input focus:
- final reviewed type or missingness values clearly come from a reviewed worker
- the draft still claims a deterministic baseline source

Expected assertions:
- reviewer does not patch provenance fields
- reviewer may emit a `review_flags` entry, but `change_set` remains unchanged for provenance-only concerns
- after deterministic apply, row values are unchanged

## Case 5 - Excluded Field Carries Stale Placement
Input focus:
- a row is `excluded_from_outputs`
- the draft leaves a non-blank `canonical_table_name`

Expected assertions:
- reviewer patches only `canonical_table_name`
- after deterministic apply, exclusion-state coherence is restored

## Case 6 - Justified No-Change Review
Input focus:
- reviewed structure, reviewed type, reviewed missingness, semantic context, and draft row values all agree

Expected assertions:
- reviewer returns `change_set = []`
- after deterministic apply, `reviewed_contract` row values are unchanged
- if the draft summary counts were already truthful, final `change_log = []`

## Case 7 - Stale Summary Counts With No Reviewer Row Edits
Input focus:
- reviewer finds no row-level problems
- draft `summary` counts are stale

Expected assertions:
- reviewer still returns `change_set = []`
- deterministic apply recomputes summary counts
- synthetic summary-count `change_log` entries are emitted

## Case 8 - Statistics Dataset Frozen-Field Regression
Input focus:
- `Mjr2` has conflicting reviewed type vs reviewed missingness signals on `skip_logic_protected`
- `A1_Q2` carries family-default missingness provenance
- `GPA` and `Final_Grade` remain baseline-typed respondent measures

Expected assertions:
- reviewer cannot patch `Mjr2.skip_logic_protected`
- reviewer cannot patch `A1_Q2.missingness_decision_source`
- reviewer cannot patch `A1_Q2.applied_sources`
- reviewer cannot patch `GPA.type_decision_source`
- reviewer cannot patch `Final_Grade.type_decision_source`

## Case 9 - Stale Canon Contract Fails Before Patch Validation
Input focus:
- `canon_contract_json` contains an incoherent row carried over from older synthesis logic
- reviewer patch shape itself may otherwise be valid

Expected assertions:
- validator rejects the stale `canon_contract_json` before validating `change_set`
- the first validation error points at the source contract row coherence issue
- repair is not asked to infer a workaround for a stale builder output

## Case 10 - Skip-Logic Bookkeeping Cannot Be Smuggled Through Interpretation Hints
Input focus:
- reviewer attempts to add or remove `skip_logic_protected` inside `interpretation_hints`
- underlying concern is frozen skip-logic bookkeeping rather than a substantive semantic correction

Expected assertions:
- validator rejects the patch
- reviewer must leave the row unchanged and use `review_flags` instead

## Case 11 - Finalized Child Placement Cannot Be Reintroduced As A Planning Hint
Input focus:
- `canon_contract_json` already resolves the row as `child_repeat_member`
- reviewer attempts to add `requires_child_table_review` or `requires_wide_to_long_review` to `structural_transform_hints`

Expected assertions:
- reviewer validator rejects the patch
- deterministic apply also rejects the patch if validation is bypassed
- canonical contract validation rejects any final contract row that carries either hint on a `child_repeat_member`
