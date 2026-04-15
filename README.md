# Dify Profiler

`dify-profiler` is a deterministic dataset profiling and artifact-delivery service for downstream LLM workers.

At its core, the service:

- ingests a tabular dataset
- compiles a structured artifact bundle that captures evidence about grain, typing, repeat structure, relationships, quality, and candidate layouts
- stores the canonical bundle in Google Cloud Storage
- serves either raw artifacts or worker-specific pruned views for Dify and other orchestrators

This repository reflects the current service implementation in [`app.py`](/Users/nikhil/Automations/dify-profiler/app.py). The broader project vision below also draws on the provided project summary, but where there is any mismatch, the code in this repo is the ground truth.

## Vision

The long-term vision is a compiler-style data understanding pipeline:

1. profile the raw dataset deterministically
2. generate an immutable evidence bundle
3. let specialist workers consume task-specific artifact views
4. adjudicate structural and semantic questions through constrained LLM steps
5. consolidate those decisions into a strict, machine-readable cleaning contract
6. eventually generate executable cleaning code and cleaned outputs in both canonical and analysis-ready target layouts

Today, this repository implements the artifact compiler and artifact serving layers. It does not yet expose a final cleaning-contract endpoint, code-generation endpoint, or execution endpoint for writing cleaned output tables.

## Current Philosophy

The service is designed around:

- deterministic evidence generation
- immutable artifact storage
- deterministic artifact adaptation at view time
- selective LLM consumption through worker-specific bundles

That means the canonical artifact in storage is treated as ground truth, while the LLM-facing response is a compiled view produced on demand through pruning, typed selection policies, and transforms.

## Canonical vs Analysis Outputs

The intended product shape is now explicitly two-layered:

- the **canonical layer** preserves structure, provenance, family identity, reference blocks, and reviewed missingness semantics
- the **analysis layer** merges compatible sibling families, applies derivation rules, and produces score-ready/model-ready tables

Reference tables are not summary tables.
They are reusable lookup or reference entities/blocks that justify their own table because they carry their own meaning, attributes, or derivation value. A standalone grouping column with no supporting attributes should usually remain on the canonical base table.

## What The Service Does Today

### 1. Build a full artifact bundle

`POST /full-bundle` is the main ingestion endpoint.

It:

- reads the uploaded file
- computes dataset identity metadata
- profiles the dataset
- builds the current artifact set
- uploads artifacts to GCS
- writes a run manifest
- returns a `run_id` and discovery URLs

### 2. Store artifacts immutably in GCS

Artifacts are uploaded into the configured `EXPORT_BUCKET` and indexed through a run manifest.

The service exposes:

- `GET /artifacts?run_id=...`
- `GET /artifacts/{artifact_id}/meta`
- `GET /artifacts/{artifact_id}/download`

These are the raw discovery and retrieval paths.

### 3. Serve worker-specific views

The service can return pruned or transformed versions of stored artifacts without mutating the underlying payload.

Current view surfaces:

- `GET /artifacts/{artifact_id}`
- `POST /artifacts/{artifact_id}/view`
- `GET /artifact-bundles`
- `POST /artifact-bundles/view`

`GET /artifact-bundles` is the simplest Dify-facing interface for workers that only need profile-default pruning.

For workers that need request-scoped pruning inputs, use `POST /artifact-bundles/view`. This now matters for `type_transform_worker`, `missingness_worker`, `semantic_context_worker`, `family_worker`, `table_layout_worker`, `scale_mapping_worker`, `analysis_layout_worker`, and `canonical_contract_reviewer`.

- `type_transform_worker`, `missingness_worker`, `semantic_context_worker`, `table_layout_worker`, and `analysis_layout_worker` can accept `global.value_filter.force_include_columns` so finalized grain, reference, and family linkage columns are always retained in a scoped bundle.
- `canonical_contract_reviewer` can also accept `global.value_filter.force_include_columns`, but the reviewer bundle is now risk-scoped. Keep that list narrow to structural carry-through columns; do not pass the full reviewed contract column list.
- `family_worker` can additionally accept `global.value_filter.force_include_family_ids` so `A8` and `B1` are reduced to just the accepted families under review.
- `table_layout_worker` can additionally accept `global.value_filter.preferred_primary_grain_keys` so `A12` layout candidates are re-ranked toward the accepted primary grain before projection.

The post-light-contract worker path also applies server-side auto-scope buckets from stored artifacts:

- `review_columns` from `A3-T` and `A3-V`
- `structural_columns` from `A9` role evidence
- `skip_trigger_columns` and `skip_affected_preview_columns` from `A16`
- `reviewer_focus_columns` from ranked `A17` reviewer-risk signals, ordered by corroborated severity and budgeted before merge

Those buckets are then consumed selectively per artifact, rather than being merged into one broad force-include list. Merged bucket values preserve insertion order, so ranked reviewer focus survives into seeded pruning.

Example:

```http
GET /artifact-bundles?run_id=<run_id>&mode=grain_worker
```

If `artifact_ids` is omitted and `mode` maps to a named profile, the service infers the worker's artifact list from that profile.

Example scoped request for a post-light-contract worker:

```http
POST /artifact-bundles/view
Content-Type: application/json

{
  "run_id": "<run_id>",
  "mode": "type_transform_worker",
  "global": {
    "value_filter": {
      "force_include_columns": ["order_id", "customer_id", "event_date"]
    }
  }
}
```

Example scoped request for the family worker:

```http
POST /artifact-bundles/view
Content-Type: application/json

{
  "run_id": "<run_id>",
  "mode": "family_worker",
  "global": {
    "value_filter": {
      "force_include_columns": ["respondent_id", "wave"],
      "force_include_family_ids": ["a_1", "m_2"]
    }
  }
}
```

Example scoped request for the table-layout worker:

```http
POST /artifact-bundles/view
Content-Type: application/json

{
  "run_id": "<run_id>",
  "mode": "table_layout_worker",
  "global": {
    "value_filter": {
      "force_include_columns": ["respondent_id", "wave"],
      "preferred_primary_grain_keys": ["respondent_id"]
    }
  }
}
```

Example scoped request for the analysis-layout worker:

```http
POST /artifact-bundles/view
Content-Type: application/json

{
  "run_id": "<run_id>",
  "mode": "analysis_layout_worker",
  "global": {
    "value_filter": {
      "force_include_columns": ["respondent_id", "wave"],
      "force_include_family_ids": ["a_1", "a_2", "m_1", "m_2", "q"]
    }
  }
}
```

## Artifact Set

The current artifact registry defined in [`app.py`](/Users/nikhil/Automations/dify-profiler/app.py) is:

- `A1` run manifest
- `A2` column dictionary
- `A3-T` transform review queue
- `A3-V` variable type review queue
- `A4` missingness catalog
- `A5` key candidates and integrity
- `A6` grain tests
- `A7` duplicate report
- `A8` repeat-dimension candidates
- `A9` role scores
- `A10` relationships and derivations
- `A11` glimpses
- `A12` table layout candidates
- `A13` semantic anchors
- `A14` quality heatmap
- `A16` conditional missingness / skip-logic proofs
- `A17` baseline column resolution
- `B1` family packets

These artifacts are intended to form an evidence graph for downstream workers rather than a single flat profiling report.

`A4` and `A16` serve different purposes:

- `A4` explains how much missingness exists and how it is encoded
- `A16` explains when missingness appears to be structurally valid because of skip logic, using trigger-centered summaries and master-switch candidates rather than exploding pairwise column lists

## Pruning Architecture

The service includes a formal pruning subsystem driven by the local pruning ledger and named worker profiles.

### Ledger-backed pruning

The pruning system supports three layers overall:

- Tier 1: global drops
- Tier 2: artifact-local drops
- Tier 3: limits, typed policies, and transforms

In practice, current worker-specific profiles such as `grain_worker` effectively collapse drop behavior into artifact-specific rules:

- Tier 1 is empty
- Tier 2 carries the profile's actual drop rules
- Tier 3 carries limits, typed policies, and transforms

The fuller three-tier shape still exists for baseline pruning and for the pruning engine itself.

Examples from the current implementation:

- `A6` ranked retention preserves the best grain candidate
- `A8` uses transforms to create compact `family_signature` views
- `A9` uses role-aware selection to preserve structurally important columns
- `type_transform_worker` uses typed scoped column selection plus compact evidence previews to reduce token waste in `A2`, `A3-T`, `A4`, `A9`, `A13`, and `A14`
- `family_worker` uses `force_include_family_ids` to reduce `A8` and `B1` to accepted families only

### Worker profiles

Profiles define worker-specific artifact lists and pruning behavior.

The service supports two profile sources:

- `gcs`
- `local`

Relevant environment variables:

- `PROFILE_SOURCE`
- `PROFILE_PREFIX`
- `LOCAL_PROFILES_DIR`

Current production direction is GCS-backed profiles, stored under:

```text
profiles/<mode>.json
```

The repository also contains a local profile example:

- [`profiles/grain_worker.json`](/Users/nikhil/Automations/dify-profiler/profiles/grain_worker.json)
- [`profiles/family_worker.json`](/Users/nikhil/Automations/dify-profiler/profiles/family_worker.json)
- [`profiles/table_layout_worker.json`](/Users/nikhil/Automations/dify-profiler/profiles/table_layout_worker.json)
- [`profiles/analysis_layout_worker.json`](/Users/nikhil/Automations/dify-profiler/profiles/analysis_layout_worker.json)
- [`profiles/canonical_contract_reviewer.json`](/Users/nikhil/Automations/dify-profiler/profiles/canonical_contract_reviewer.json)

## Current Dify Integration Model

The intended Dify pattern is:

1. upload a dataset to `POST /full-bundle`
2. receive a `run_id`
3. request worker-specific bundles from `/artifact-bundles` or `/artifact-bundles/view`
4. convert the returned artifact payload into an LLM-safe string inside Dify
5. run specialist workers on those pruned bundles

Post-grain workers can also consume `A16` so they do not misread structurally valid skip-logic nulls as generic low-quality missingness.

After semantic, type, and missingness workers have produced validated JSON outputs, the intended next stage is a per-family loop:

1. parse finalized light-contract decisions and derive accepted `family_id` values
2. request `POST /artifact-bundles/view` with `mode = "family_worker"`
3. build one loop item per accepted family using:
   - `light_contract_decisions`
   - `semantic_context_json`
   - `type_transform_worker_json`
   - `missingness_worker_json`
   - matched `A8` and `B1` family evidence
4. run [`prompts/family_worker_system_prompt.md`](/Users/nikhil/Automations/dify-profiler/prompts/family_worker_system_prompt.md) once per family
5. validate each family JSON item, repair once if needed, then aggregate into one `family_worker_json`
   - each family item may optionally include `member_defaults` for safe family-wide type defaults and structural-only missingness defaults that should propagate to sibling columns later
   - for repair-node use, use [`prompts/REPAIR_family.md`](/Users/nikhil/Automations/dify-profiler/prompts/REPAIR_family.md)

The intended next stage after family is ordered-scale / codebook mapping:

1. request `POST /artifact-bundles/view` with `mode = "scale_mapping_worker"` and supply:
   - `light_contract_decisions`
   - `family_worker_json`
2. inspect `artifacts.scale_mapping_bundle.has_mapping_evidence`
3. if it is `false`, skip the extractor and pass an empty extractor sentinel to the resolver
4. otherwise run [`prompts/scale_mapping_worker_system_prompt.md`](/Users/nikhil/Automations/dify-profiler/prompts/scale_mapping_worker_system_prompt.md)
5. validate the output with [`JSON validators/scale_mapping_validator.json`](/Users/nikhil/Automations/dify-profiler/JSON%20validators/scale_mapping_validator.json)
6. repair once with [`prompts/REPAIR_scale_mapping.md`](/Users/nikhil/Automations/dify-profiler/prompts/REPAIR_scale_mapping.md) if needed
7. call `POST /contracts/scale-mappings` with:
   - `run_id`
   - `light_contract_decisions`
   - `family_worker_json`
   - validated `scale_mapping_extractor_json` when present
8. pass the resulting `scale_mapping_json` into canonical synthesis and later analysis-layout planning

This stage is intentionally narrow:
- it consumes a backend-built compact bundle rather than the full canonical bundle
- it may use a codebook PDF when one has been uploaded through `POST /codebooks/upload`
- the deterministic resolver is the final authority; the extractor is proposal-only

The intended next stage after family is a single-pass canonical table-layout proposal worker:

1. request `POST /artifact-bundles/view` with `mode = "table_layout_worker"` and supply:
   - `global.value_filter.force_include_columns` from finalized light-contract keys
   - `global.value_filter.preferred_primary_grain_keys` from `light_contract_decisions.primary_grain_decision.keys`
2. combine the returned bundle with:
   - `light_contract_decisions`
   - `semantic_context_json`
   - `type_transform_worker_json`
   - `missingness_worker_json`
   - `family_worker_json`
3. run [`prompts/table_layout_worker_system_prompt.md`](/Users/nikhil/Automations/dify-profiler/prompts/table_layout_worker_system_prompt.md)
4. validate the returned JSON, repair once if needed, then resolve one canonical `table_layout_worker_json`
   - when `column_table_assignments[].assignment_role` is `exclude_from_outputs` or `unresolved`, the canonical contract allows blank `assigned_table`
   - in the repair branch, the repair validator must validate the repair LLM raw JSON text, not the original first-pass worker output
   - the repair prompt should receive the original invalid table-layout JSON plus the first validator's `validation_errors_json`

For Dify structured-output mode, use:
- [`schemas/table_layout_worker.response.schema.json`](/Users/nikhil/Automations/dify-profiler/schemas/table_layout_worker.response.schema.json)
  - paste this object directly into Dify's `json_schema` box
  - it already includes the Dify/OpenAI `json_schema` wrapper fields: `name`, `strict`, and `schema`
  - do not wrap it again in `response_format`

For direct API / HTTP-node usage, use:
- [`schemas/table_layout_worker.openai_response_format.example.json`](/Users/nikhil/Automations/dify-profiler/schemas/table_layout_worker.openai_response_format.example.json)
  - this example includes the outer `response_format` wrapper
  - do not paste this file into Dify's `json_schema` box

For prompt regression checks, use:
- [`prompts/regression_cases/table_layout_worker_regression_cases.md`](/Users/nikhil/Automations/dify-profiler/prompts/regression_cases/table_layout_worker_regression_cases.md)

For a local validator smoke check of the exclude/unresolved assignment contract, run:
- `python scripts/table_layout_validator_smoke.py`


This canonical stage is intended to produce:
- a final proposed table set
- explicit source-column placement for every known column
- parent-child/reference layout decisions for the canonical layer
- compact family-table summaries in `table_suggestions`; use `column_table_assignments` as the exhaustive source-column map

It is still a proposal-stage worker. It does not emit merged wave tables, score tables, or other analysis-ready outputs.

The intended next stage after canonical layout is a deterministic canonical-column-contract synthesis node:

1. call `POST /contracts/canonical-columns`
2. send:
   - `run_id`
   - `light_contract_decisions`
   - `semantic_context_json`
   - `type_transform_worker_json`
   - `missingness_worker_json`
   - `family_worker_json`
   - `table_layout_worker_json`
   - `scale_mapping_json`
3. let the backend load `A17` directly for the same `run_id`
   - if `A17` is missing, the backend deterministically rebuilds it from `A2`, `A3-T`, `A3-V`, `A4`, `A9`, `A13`, `A14`, and `A16`
4. consume one canonical contract object with:
   - `summary`
   - `column_contracts`
   - `global_value_rules`
   - `review_flags`
   - `assumptions`

This node is deterministic by design. It does not reinterpret free-form prose from light-contract comments or semantic notes. It only consumes structured reviewed outputs plus profiler artifacts.

Its job is to:
- emit one per-column contract row for every A2 source column
- preserve reviewed overrides where they exist
- propagate optional `family_worker_json.member_defaults` where a shared family default is safe
- fill remaining non-structural control fields from the deterministic `A17` baseline layer
- leave semantic enrichment blank when no structured evidence exists
- expose provenance for type, structure, missingness, and semantic decisions

For missingness specifically, deterministic synthesis now treats the reviewed missingness contract as structured input, not prose:
- `missingness_worker_json.global_contract.token_missing_placeholders_detected` is the machine-readable global token-missingness signal
- `missingness_worker_json.global_findings` remains human-readable support text only
- `family_worker_json.member_defaults` may not push non-structural missingness outcomes such as `token_missingness_present` or `no_material_missingness`
- final row states are reconciled through shared missingness invariants before validation

The canonical merge order is:
1. structure from `table_layout_worker_json` plus light-contract fallbacks
2. semantic enrichment from `semantic_context_json`
3. family-shared defaults from `family_worker_json.member_defaults`
4. reviewed column overrides from `type_transform_worker_json`
5. reviewed missingness overrides from `missingness_worker_json`
6. `A17` for all remaining non-structural gaps

Within missingness resolution, the deterministic precedence is:
1. reviewed per-column missingness from `missingness_worker_json.column_decisions`
2. deterministic baseline missingness from `A17` / `A4` / `A16`
3. family defaults, but only for structural missingness categories
4. final deterministic invariant reconciliation

For Dify structured-output or contract documentation, use:
- [`schemas/canonical_column_contract.response.schema.json`](/Users/nikhil/Automations/dify-profiler/schemas/canonical_column_contract.response.schema.json)

For validator-node use, use:
- [`JSON validators/canonical_column_contract_validator.json`](/Users/nikhil/Automations/dify-profiler/JSON%20validators/canonical_column_contract_validator.json)
  - this validator now accepts:
    - `canonical_column_contract_output`
    - `expected_source_columns_json`
    - optional `missingness_worker_json`
  - pass `missingness_worker_json` when you want the validator to enforce `global_contract.token_missing_placeholders_detected`

For a local synthesis + validator smoke check, run:
- `python scripts/canonical_column_contract_smoke.py`

The intended next stage after the canonical-column contract is a canonical-contract reviewer:

1. request `POST /artifact-bundles/view` with `mode = "canonical_contract_reviewer"`
   If you pass `global.value_filter.force_include_columns`, keep it limited to structural keys or screening columns that must survive into the reviewer context. Do not pass the full canonical contract column list.
2. rebuild `canon_contract_json` with the latest canonical-column synthesis code and validate it before reviewer execution
3. combine the returned bundle with:
   - `canon_contract_json`
   - `light_contract_decisions`
   - `semantic_context_json`
   - `type_transform_worker_json`
   - `missingness_worker_json`
   - `family_worker_json`
   - `table_layout_worker_json`
4. run [`prompts/canonical_contract_reviewer_system_prompt.md`](/Users/nikhil/Automations/dify-profiler/prompts/canonical_contract_reviewer_system_prompt.md)
5. validate the returned patch JSON, repair once if needed, then resolve one authoritative `canon_review_patch`
6. apply the resolved patch deterministically to `canon_contract_json`
7. emit one final `canon_review_json` containing:
   - `review_summary`
   - `reviewed_contract`
   - `change_log`
   - `reviewer_flags`
   - `reviewer_assumptions`
   - deprecated compatibility aliases: `review_flags`, `assumptions`

This reviewer is now intentionally patch-only. The LLM no longer returns the full edited contract. It returns a compact `change_set`, and deterministic code applies those row-level edits, recomputes summary counts, and builds the final ledger.

Reviewer patch entries are now LLM-facing `column + field + after_value` edits. The LLM does not emit row indices or `target_path`. Deterministic code resolves the matching row index from `canon_contract_json.column_contracts[*].column` and derives the final `target_path` only for the applied `change_log`.

Post-canonical child-placement hints are builder-owned. If a row is already `child_repeat_member`, neither the builder nor the reviewer should carry `requires_child_table_review` or `requires_wide_to_long_review` in `structural_transform_hints`. Those are pre-placement planning hints, not finalized child-row metadata.

The nested `reviewed_contract` is intended to be standalone. Its `summary` now includes:
- `review_applied`
- `review_change_count`
- `review_changed_column_count`
- a post-review `overview`

For validator-node use, use:
- [`JSON validators/canonical_contract_reviewer_validator.json`](/Users/nikhil/Automations/dify-profiler/JSON%20validators/canonical_contract_reviewer_validator.json)

For deterministic apply-node use, use:
- [`canonical_contract_reviewer_apply_patch_node.py`](/Users/nikhil/Automations/dify-profiler/canonical_contract_reviewer_apply_patch_node.py)

The Dify apply node should remain self-contained. Do not rely on repo-local Python imports inside the deployed code-node body unless your Dify runtime explicitly packages those helper modules.

For repair-node use, use:
- [`prompts/REPAIR_canonical_contract_reviewer.md`](/Users/nikhil/Automations/dify-profiler/prompts/REPAIR_canonical_contract_reviewer.md)

Upstream repair-node prompts that must stay aligned with the current worker contracts:
- [`prompts/REPAIR_missingness.md`](/Users/nikhil/Automations/dify-profiler/prompts/REPAIR_missingness.md)
- [`prompts/REPAIR_family.md`](/Users/nikhil/Automations/dify-profiler/prompts/REPAIR_family.md)

Repair payload guidance:
- send the original invalid reviewer JSON plus the first validator's `validation_errors_json`
- do not resend the full reviewer input bundle
- during rollout only, you may also pass a compact `column_index_map_json` derived from the current `canon_contract_json`; treat it as lookup metadata, not evidence
- run repair in an isolated node session or subworkflow with no prior assistant/user history carry-over
- use response format `json_object` and temperature `0`
- treat the repair path as miswired unless repair prompt tokens are both:
  - `< 15,000`
  - `< 15%` of first-pass reviewer prompt tokens

Debug artifact guidance for local review:
- prefer unwrapped payload artifacts such as `canon_contract.payload.json`, `canon_review_patch.payload.json`, and `canon_review.payload.json`
- keep transport wrappers only as optional `*.transport.json` sidecars when HTTP diagnostics are needed
- the Dify workflow capture rename is still a manual rollout step because no workflow export is checked into this repo
- to unwrap legacy mixed-format reviewer artifacts in-place, run:
  - `python scripts/unwrap_canonical_reviewer_artifacts.py Outputs/Placeholder`
- keep tracked regression fixtures under [`testdata/canonical_reviewer/`](/Users/nikhil/Automations/dify-profiler/testdata/canonical_reviewer) rather than relying on ignored `Outputs/Placeholder` runs as the only corpus

For a local reviewer-validator smoke check, run:
- `python scripts/canonical_contract_reviewer_smoke.py`

For a local reviewer bundle-pruning regression smoke check, run:
- `python scripts/canonical_contract_reviewer_pruning_smoke.py`

The canonical reviewer bundle is intentionally risk-scoped:
- `reviewer_focus_columns` is an ordered, risk-ranked bucket built from corroborated `A17` evidence rather than broad review flags alone
- `A13` is anchor-only, so empty-anchor rows are not retained just because they were risky elsewhere
- strong standalone reviewer heuristics in `A14` and `A17` are limited to drift, skip protection, and materially low confidence / quality

The intended next stage after the canonical-contract reviewer remains an analysis-layout worker:

1. request `POST /artifact-bundles/view` with `mode = "analysis_layout_worker"`
2. combine the returned bundle with:
   - `light_contract_decisions`
   - `semantic_context_json`
   - `scale_mapping_json`
   - `type_transform_worker_json`
   - `missingness_worker_json`
   - `family_worker_json`
   - `table_layout_worker_json`
3. run [`prompts/analysis_layout_worker_system_prompt.md`](/Users/nikhil/Automations/dify-profiler/prompts/analysis_layout_worker_system_prompt.md)
4. validate the returned JSON, repair once if needed, then resolve one canonical `analysis_layout_worker_json`

This stage is intended to produce:
- merged analysis-ready response tables where justified
- explicit derivation plans for answer-key scoring and score aggregation
- respondent-wave analysis marts

The long-term hard/final contract shape is now intended to separate:
- `canonical_layer`
- `analysis_layer`
- `derivations`

## Light Contract Semantic Context

The light-contract workbook now captures two kinds of early semantic input in the `Overrides` sheet:

- `dataset_context_and_collection_notes`
- `semantic_codebook_and_important_variables`

It also captures one structured optional mapping sheet:

- `Scale Mappings`

These rows are intended for:

- dataset purpose and row-meaning notes
- collection-process changes across waves, forms, or exports
- known optional or conditioned sections
- condition, screener, or master-switch variables
- simple code meanings and label mappings
- semantic placeholders or status flags

The `Scale Mappings` sheet is intended for:
- family-level ordered label ladders
- standalone column ordered label ladders
- optional explicit numeric score mappings when the human knows them

They are intentionally not the place for final table-layout instructions.

When a light contract is accepted or a modified workbook is parsed, the downstream handoff now includes:

```json
"semantic_context_input": {
  "dataset_context_and_collection_notes": "...",
  "semantic_codebook_and_important_variables": "..."
},
"scale_mapping_input": [
  {
    "target_kind": "family",
    "target_id": "q_9_main_cell_group",
    "response_scale_kind": "familiarity_scale",
    "ordered_labels": ["Never Heard of It 0", "Very Familiar 6"],
    "numeric_scores": [],
    "notes": "..."
  }
]
```

This keeps semantic truth available early without mixing it into the later hard-contract stage, while also giving the scale-mapping resolver a structured human override path.

For example, the grain worker can call:

```http
GET /artifact-bundles?run_id=<run_id>&mode=grain_worker
```

and receive the profile-defined artifact set for grain reasoning.

The type/value worker should prefer:

```http
POST /artifact-bundles/view
```

with `global.value_filter.force_include_columns` derived from finalized `light_contract_decisions`:

- `primary_grain_decision.keys`
- `reference_decisions[].keys` (legacy `dimension_decisions[].keys` may still appear during migration)
- `family_decisions[].parent_key`

This keeps Dify focused on orchestration and LLM adjudication while leaving evidence generation, storage, and compression inside this service.

For the type/value worker, add a post-LLM validation step in Dify:

1. parse the LLM `text` as JSON
2. verify the top-level object contains:
   - `worker`
   - `summary`
   - `column_decisions`
   - `global_transform_rules`
   - `review_flags`
   - `assumptions`
3. fail fast or retry once if parsing fails

Why this matters:
- the worker is intended to emit strict JSON only
- downstream stages should not silently accept malformed outputs
- this is the safest place to guard against prompt drift or formatting failures

After light-contract finalization, add one more Dify step:

1. if `light_contract_status == "accepted"`, skip the semantic bundle + semantic LLM and emit:
   - `{"status":"skipped","reason":"light_contract_accepted"}`
2. if `light_contract_status == "modified"`, request `POST /artifact-bundles/view` with `mode = "semantic_context_worker"`
3. pass both `light_contract_decisions` and that semantic grounding bundle into [`prompts/semantic_context_interpreter_system_prompt.md`](/Users/nikhil/Automations/dify-profiler/prompts/semantic_context_interpreter_system_prompt.md)
4. validate the returned JSON
5. pass the resulting `semantic_context_json` downstream

This semantic-context step should run before later type, missingness, family, or table-modeling stages consume the user notes.

## Codebook Upload

Use:
- `POST /codebooks/upload`
- `GET /codebooks/context`

The upload step stores:
- `runs/<run_id>/codebook.pdf`
- `runs/<run_id>/codebook_pages.json`
- `runs/<run_id>/codebook_document.json`

The scale-mapping bundle builder uses extracted page text to shortlist relevant snippets before the extractor runs. The original PDF remains available by signed URL for Dify attachment when needed, but the snippet shortlist should be the primary grounding context.

### Semantic Context Validator

Use a validator-only Code node after the semantic-context interpreter. The node should fail fast on malformed output and return only a trivial success marker.

Suggested Dify config:

- input: `semantic_context_output` (`string`)
- output: `validation_ok` (`string`)

```python
import json

ALLOWED_KINDS = {
    "condition_column",
    "status_or_flag",
    "code_column",
    "family_context",
    "date_or_phase_context",
    "business_key_context",
    "placeholder_value_context",
    "other",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_non_empty_string(value, field_name):
    _require(isinstance(value, str), f"{field_name} must be a string.")
    _require(value.strip() != "", f"{field_name} must not be blank.")


def _require_dict(value, field_name):
    _require(isinstance(value, dict), f"{field_name} must be an object.")


def _require_list(value, field_name):
    _require(isinstance(value, list), f"{field_name} must be an array.")


def _validate_summary(summary):
    _require_dict(summary, "summary")
    _require("overview" in summary, "summary.overview is required.")
    _require("key_points" in summary, "summary.key_points is required.")
    _require_non_empty_string(summary["overview"], "summary.overview")
    _require_list(summary["key_points"], "summary.key_points")
    for i, item in enumerate(summary["key_points"]):
        _require_non_empty_string(item, f"summary.key_points[{i}]")


def _validate_dataset_context(dataset_context):
    _require_dict(dataset_context, "dataset_context")
    for key in [
        "dataset_purpose",
        "row_meaning_notes",
        "collection_change_notes",
        "known_optional_or_conditioned_sections",
    ]:
        _require(key in dataset_context, f"dataset_context.{key} is required.")
    _require(isinstance(dataset_context["dataset_purpose"], str), "dataset_context.dataset_purpose must be a string.")
    _require(isinstance(dataset_context["row_meaning_notes"], str), "dataset_context.row_meaning_notes must be a string.")
    for key in ["collection_change_notes", "known_optional_or_conditioned_sections"]:
        _require_list(dataset_context[key], f"dataset_context.{key}")
        for i, item in enumerate(dataset_context[key]):
            _require_non_empty_string(item, f"dataset_context.{key}[{i}]")


def _validate_important_variables(items):
    _require_list(items, "important_variables")
    for i, item in enumerate(items):
        field_prefix = f"important_variables[{i}]"
        _require_dict(item, field_prefix)
        for key in ["column_or_family", "kind", "meaning", "downstream_importance", "confidence"]:
            _require(key in item, f"{field_prefix}.{key} is required.")
        _require_non_empty_string(item["column_or_family"], f"{field_prefix}.column_or_family")
        _require_non_empty_string(item["kind"], f"{field_prefix}.kind")
        _require(item["kind"] in ALLOWED_KINDS, f"{field_prefix}.kind must be one of: {', '.join(sorted(ALLOWED_KINDS))}")
        _require_non_empty_string(item["meaning"], f"{field_prefix}.meaning")
        _require_non_empty_string(item["downstream_importance"], f"{field_prefix}.downstream_importance")
        _require(_is_number(item["confidence"]), f"{field_prefix}.confidence must be numeric.")
        _require(0 <= float(item["confidence"]) <= 1, f"{field_prefix}.confidence must be between 0 and 1.")


def _validate_codebook_hints(items):
    _require_list(items, "codebook_hints")
    for i, item in enumerate(items):
        field_prefix = f"codebook_hints[{i}]"
        _require_dict(item, field_prefix)
        for key in ["column", "codes_or_labels_note", "meaning", "confidence"]:
            _require(key in item, f"{field_prefix}.{key} is required.")
        _require_non_empty_string(item["column"], f"{field_prefix}.column")
        _require_non_empty_string(item["codes_or_labels_note"], f"{field_prefix}.codes_or_labels_note")
        _require_non_empty_string(item["meaning"], f"{field_prefix}.meaning")
        _require(_is_number(item["confidence"]), f"{field_prefix}.confidence must be numeric.")
        _require(0 <= float(item["confidence"]) <= 1, f"{field_prefix}.confidence must be between 0 and 1.")


def _validate_flat_note_list(items, field_name, item_key):
    _require_list(items, field_name)
    for i, item in enumerate(items):
        field_prefix = f"{field_name}[{i}]"
        _require_dict(item, field_prefix)
        for key in [item_key, "issue" if field_name == "review_flags" else "explanation", "why" if field_name == "review_flags" else None]:
            if key is None:
                continue
            _require(key in item, f"{field_prefix}.{key} is required.")
            _require_non_empty_string(item[key], f"{field_prefix}.{key}")


def main(semantic_context_output: str):
    _require(isinstance(semantic_context_output, str), "semantic_context_output must be a string.")
    _require(semantic_context_output.strip() != "", "semantic_context_output is empty.")

    try:
        parsed = json.loads(semantic_context_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Semantic context output is not valid JSON: {e}")

    _require_dict(parsed, "root")
    if parsed.get("status") == "skipped":
        _require_non_empty_string(parsed.get("reason"), "reason")
        return {
            "validation_ok": "true"
        }

    for key in ["worker", "summary", "dataset_context", "important_variables", "codebook_hints", "review_flags", "assumptions"]:
        _require(key in parsed, f"Missing required top-level key: {key}")

    _require_non_empty_string(parsed["worker"], "worker")
    _require(parsed["worker"] == "semantic_context_interpreter", "worker must be 'semantic_context_interpreter'.")
    _validate_summary(parsed["summary"])
    _validate_dataset_context(parsed["dataset_context"])
    _validate_important_variables(parsed["important_variables"])
    _validate_codebook_hints(parsed["codebook_hints"])
    _validate_flat_note_list(parsed["review_flags"], "review_flags", "item")
    _validate_flat_note_list(parsed["assumptions"], "assumptions", "assumption")

    return {
        "validation_ok": "true"
    }
```

## Likely Worker Structure

The exact Dify workflow is not encoded in this repository, but the current artifact design strongly suggests a specialist pipeline along these lines:

- grain specialist
- semantic-context interpreter
- type/transform specialist
- missingness specialist
- repeat/family specialist
- layout/model specialist
- final consolidator

The intended human checkpoints described in the project summary are:

- a lighter structural review after the grain stage
- a heavier contract review before final contract generation

These are part of the broader product direction rather than currently implemented endpoints in this repo.

## What Is Implemented vs Planned

### Implemented now

- deterministic profiling
- artifact generation
- GCS persistence
- raw artifact retrieval
- worker-specific artifact views
- bundle retrieval for downstream LLM workers
- GCS-backed or local-backed profile loading
- Cloud Run-oriented runtime and signed URL export support

### Planned or inferred future layers

- final validated cleaning contract
- contract schema validation
- code generation from that validated contract
- execution of the generated cleaning script
- production of final cleaned tables
- optional post-clean EDA and reporting

## Key Endpoints

### Primary current endpoints

- `POST /full-bundle`
- `GET /artifacts`
- `GET /artifacts/{artifact_id}/meta`
- `GET /artifacts/{artifact_id}/download`
- `GET /artifacts/{artifact_id}`
- `POST /artifacts/{artifact_id}/view`
- `GET /artifact-bundles`
- `POST /artifact-bundles/view`
- `GET /health`

### Legacy or compatibility endpoints

These remain in the codebase but are hidden or deprecated:

- `POST /profile`
- `POST /profile_summary`
- `POST /profile_column_detail`
- `POST /evidence_associations`
- `POST /export/light-contract-xlsx`
- `POST /export/manifest-txt`

## Runtime And Storage Model

### Authentication

The service authenticates API requests using `PROFILER_API_KEY`.

- most protected endpoints expect a bearer token
- raw download also supports `x-api-key`

### GCS

Artifacts and manifests are stored in `EXPORT_BUCKET`.

The service also supports signed URL generation for export endpoints using:

- `EXPORT_BUCKET`
- `SIGNING_SA_EMAIL`
- `EXPORT_SIGNED_URL_TTL_MINUTES`

### Cloud-native credentials

The code is written around Application Default Credentials and Google auth libraries, which makes it suitable for Cloud Run.

## Local Development

Create and activate a virtual environment, install dependencies, and run with `uvicorn`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8080
```

For local profile testing:

```bash
export PROFILE_SOURCE=local
export LOCAL_PROFILES_DIR=/Users/nikhil/Automations/dify-profiler/profiles
```

For GCS-backed profile loading:

```bash
export PROFILE_SOURCE=gcs
export PROFILE_PREFIX=profiles/
export EXPORT_BUCKET=<your-bucket>
```

## Deployment

The repo includes:

- [`Dockerfile`](/Users/nikhil/Automations/dify-profiler/Dockerfile)
- [`deploy.ps1`](/Users/nikhil/Automations/dify-profiler/deploy.ps1)

The current deployment target is Cloud Run using source deployment plus explicit runtime environment variables.

The Cloud Run deployment path now assumes:

- `PROFILE_SOURCE=gcs`
- `PROFILE_PREFIX=profiles/`
- no local profile fallback in production

## Testing And Debugging

The service includes:

- request logging middleware
- `X-Request-Id` propagation
- rate limiting
- upload size limits
- stage logging for ingestion/profiling phases
- pruning debug support
- smoke checks gated by `RUN_PRUNING_SMOKE_TESTS`

This supports a workflow where canonical artifacts can be compared against pruned views and runtime stages can be traced through logs.

## Why This Project Exists

The project is trying to solve a hard problem: messy tabular datasets often contain multiple overlapping grains, ambiguous types, token-coded missingness, repeated families, layout drift, and mixed semantic signals. A single prompt asking an LLM to “clean the CSV” is hard to trust, hard to audit, and hard to debug.

This service takes a different path:

- compile deterministic evidence first
- let specialist reasoning happen on bounded views of that evidence
- keep the process inspectable and auditable
- eventually turn those decisions into a strict, executable cleaning plan

In that sense, the project is moving toward a compiler for data cleaning and data model reconstruction, not just a profiler.
