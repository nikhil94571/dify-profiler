# Pruning / Bundle Quality Benchmark Report

## Bundle Expectations

### `run_1_fifa`
- Expected bundle shape: compact flat-entity bundle with structural columns retained, a small transform/review layer, and no family-driven expansion.
- Main pruning risk: minor dead weight from force-included rows appearing in low-yield artifacts.

### `run_2_statistics`
- Expected bundle shape: medium-sized survey bundle with strong structural columns, a moderate amount of missingness/quality context, and no full-dataset carryover.
- Main pruning risk: wide-survey question blocks leaking too broadly into `A4` or `A14`.

### `run_3_financial`
- Expected bundle shape: very small flat-transaction bundle with only the key structural columns and a small transform context.
- Main pruning risk: over-including accepted dimensions across every artifact despite weak signal.

### `run_4_responseorder`
- Expected bundle shape: larger survey bundle, but still scoped tightly around the light-contract columns plus a small number of high-value transform/quality signals.
- Main pruning risk: intrinsic keep-rules pulling in broad high-missingness matrix families that the type worker does not actually need.

## Score Table

| Run | Evidence Coverage / Completeness | Relevance / Focus | Efficiency / Size Discipline | Force-Include Precision | Operational Usefulness | Catastrophic Pruning Error | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `run_1_fifa` | 5 | 4 | 4 | 5 | 4 | No | Pass |
| `run_2_statistics` | 5 | 4 | 4 | 5 | 4 | No | Pass |
| `run_3_financial` | 5 | 5 | 5 | 5 | 5 | No | Pass |
| `run_4_responseorder` | 4 | 2 | 2 | 5 | 3 | No | Borderline |

Scoring rubric:
- `5`: strong
- `4`: good with minor issues
- `3`: usable but notable weaknesses
- `2`: poor / risky
- `1`: broken

## Findings

### 1. `run_4_responseorder`: `A2` is still materially over-retained on wide survey data
- Main risk: the bundle is `108812` bytes, with `A2` alone consuming `83401` bytes across `90` rows.
- The retained `A2` rows are dominated by matrix-family columns:
  - `Q13Main_cell_group`: 21 rows
  - `Q15Main_cell_group`: 20 rows
  - `Q16Main_cell_group`: 15 rows
  - `Q17Main_cell_group`: 13 rows
  - `Q18Main_cell_group`: 8 rows
- Why it matters: this is not coming from light-contract force-includes. It is mostly intrinsic retention from broad keep-rules on wide high-missingness families, which is exactly the kind of evidence the type worker should not receive in bulk.

### 2. `run_4_responseorder`: `A4` shows the same wide-family leakage pattern
- Main risk: `A4` is capped at `80` rows but is dominated by the same matrix families as `A2`.
- Why it matters: this suggests the current missingness guardrail is still too permissive on wide survey exports. It is carrying a large amount of family-wide missingness context that is only weakly tied to the type worker’s structural remit.

### 3. Force-include behavior is precise across all four runs
- Main strength: all expected light-contract columns appear in `A2`, `A4`, `A9`, `A13`, and `A14` in every run.
- Why it matters: the POST `/artifact-bundles/view` path is working correctly. The remaining bundle issues are not caused by force-includes being too broad.

### 4. `run_2_statistics`: the bundle is larger, but most of that size is legitimate signal
- Main evidence:
  - bundle size `69076` bytes
  - `A14`: `80` rows / `10291` bytes
  - `A4`: `74` rows / `12813` bytes
  - `A16`: `7` skip-logic rules / `4530` bytes
- Why it matters: unlike `run_4_responseorder`, this looks mostly justified. `A14` is signal-heavy (`74/80` drift-detected; `76/80` low-quality), and `A16` contains real skip-logic evidence the worker can use.

### 5. `run_3_financial` is the best-pruned bundle
- Main evidence:
  - bundle size `9672` bytes
  - `A2`: `5` rows
  - `A4`: `5` rows
  - `A9`: `6` rows
  - `A13`: `5` rows
  - `A14`: `5` rows
- Why it matters: this is the clearest proof that the current pruning architecture can be efficient and sufficient when the dataset shape is simple.

### 6. There is still low-value field-level residue across all runs
- Universal pattern:
  - `A4.token_breakdown` is empty in every retained row across all four runs
  - `A2.missing_tokens_observed` is empty in every retained row across all four runs
- Why it matters: this is small compared with the survey-wide row-retention problem, but it is still pure dead weight and should be removable generically.

### 7. `A13` remains low-yield in most runs
- Anchored-row counts:
  - `run_1_fifa`: `5 / 16`
  - `run_2_statistics`: `1 / 4`
  - `run_3_financial`: `1 / 5`
  - `run_4_responseorder`: `0 / 6`
- Why it matters: `A13` is no longer large, so this is not a major token problem, but most retained `A13` rows are present only because of force-includes and have no anchor signal.

## Per-Run Notes

### `run_1_fifa`
- Evidence coverage / completeness: **5**
  - Strength: all structural columns needed by the accepted contract are present.
  - Risk: none material.
- Relevance / focus: **4**
  - Strength: the bundle stays centered on player-level structural fields and a modest transform layer.
  - Risk: some force-included rows appear in `A13` and `A14` even when they carry little signal.
- Efficiency / size discipline: **4**
  - Strength: `36372` bytes is reasonable for a 77-column entity table.
  - Risk: small low-yield artifacts remain.
- Force-include precision: **5**
  - Strength: `ID`, `Club`, `Nationality`, and `Best Position` are retained exactly where expected.
  - Risk: none.
- Operational usefulness: **4**
  - Strength: the type worker gets enough context without obvious bloat.
  - Risk: a little artifact residue still survives.

### `run_2_statistics`
- Evidence coverage / completeness: **5**
  - Strength: the bundle includes the accepted structural columns plus the review-heavy question fields and skip-logic context the worker plausibly needs.
  - Risk: none material.
- Relevance / focus: **4**
  - Strength: `A13` is tight and `A9` is controlled.
  - Risk: `A4` and `A14` are still broad, though mostly justified here.
- Efficiency / size discipline: **4**
  - Strength: `69076` bytes is acceptable for a 204-column wide survey with real missingness and skip-logic signal.
  - Risk: it is near the upper end of what still feels efficient.
- Force-include precision: **5**
  - Strength: `Grad_Country`, `Grad_Prov`, `ID`, and `Mjr1` are retained correctly without obvious spillover from force-include logic itself.
  - Risk: none.
- Operational usefulness: **4**
  - Strength: this is a usable type-worker bundle with real structural and quality evidence.
  - Risk: some missingness-heavy rows still add noise.

### `run_3_financial`
- Evidence coverage / completeness: **5**
  - Strength: all accepted structural columns are present and the bundle still carries one transform-review row.
  - Risk: none material.
- Relevance / focus: **5**
  - Strength: nearly every retained row is directly tied to the light contract or an obvious type/transform need.
  - Risk: none material.
- Efficiency / size discipline: **5**
  - Strength: `9672` bytes is exactly the kind of size target the current architecture should hit on flat datasets.
  - Risk: none material.
- Force-include precision: **5**
  - Strength: `Transaction_ID`, `Customer_ID`, `Product_Name`, and `Payment_Method` are retained correctly.
  - Risk: none.
- Operational usefulness: **5**
  - Strength: the worker gets a tight, readable, high-signal bundle.
  - Risk: none material.

### `run_4_responseorder`
- Evidence coverage / completeness: **4**
  - Strength: all contract-critical structural columns are present, plus `A3-T` and `A16` provide direct transform and skip-logic context.
  - Risk: there is a lot more evidence than needed.
- Relevance / focus: **2**
  - Strength: `A9`, `A13`, and `A14` are relatively focused.
  - Risk: `A2` and `A4` are dominated by broad high-missingness matrix families unrelated to the light-contract columns.
- Efficiency / size discipline: **2**
  - Strength: the bundle is still smaller than an unpruned baseline would have been.
  - Risk: `108812` bytes is too large relative to the actual type-worker scope, and most of the excess is concentrated in one artifact.
- Force-include precision: **5**
  - Strength: `RespondentId`, `LanguageCode`, `participant`, and `debrief` are retained exactly as intended.
  - Risk: none.
- Operational usefulness: **3**
  - Strength: the bundle is still usable because the critical structural context is present.
  - Risk: the worker is exposed to a large amount of likely irrelevant matrix-family context, which increases the chance of overreach and token waste.

## Cross-Run Synthesis

- Robust behavior:
  - force-includes behave correctly in all four runs
  - no bundle is missing the finalized light-contract columns
  - small flat datasets are pruned extremely well
  - `A9`, `A13`, and `A14` are much more controlled than earlier iterations
- Main remaining weakness:
  - intrinsic keep-rules in `A2` and `A4` are still too broad on wide survey datasets
  - this is most visible in `run_4_responseorder`, where high-missingness matrix families overwhelm the bundle even though they were not directly requested by the light contract
- Most common low-value residue:
  - empty `A4.token_breakdown`
  - empty `A2.missing_tokens_observed`
  - low-yield `A13` rows retained only because of force-includes

## Overall Verdict

**Usable but needs tightening.**

The pruning stage passes the basic acceptance bar:
- no catastrophic pruning failures
- 3 of 4 runs are clear `Pass`
- the remaining run is a workable `Borderline`
- force-includes are precise
- critical structural evidence is preserved

The next tightening pass should focus on two things:
- narrow `A2` and `A4` intrinsic retention on wide survey datasets, especially high-missingness family spillover
- remove generic dead-weight fields such as empty `token_breakdown` and empty `missing_tokens_observed`
