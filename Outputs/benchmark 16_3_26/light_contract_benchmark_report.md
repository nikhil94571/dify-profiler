# Light-Contract Benchmark Report

## Contract Expectations

### `run_1_fifa`
- Expected contract shape: accept the player-level grain, keep families empty, and only promote obviously reusable dimensions.
- Main contract risk: over-promoting stable categories such as `Best Position` into dimensions.

### `run_2_statistics`
- Expected contract shape: accept respondent-level grain, accept the clearly repeated survey families, and keep any borderline generic family cautious.
- Main contract risk: over-finalizing families or dimensions without preserving enough uncertainty.

### `run_3_financial`
- Expected contract shape: accept transaction-level grain, keep families empty, and stay conservative on dimensions because the table is flat and dirty.
- Main contract risk: treating noisy descriptive attributes as finalized dimensions.

### `run_4_responseorder`
- Expected contract shape: accept respondent-level grain, accept the obvious `RowN` matrix families, and avoid over-promoting weak metadata fields into accepted dimensions.
- Main contract risk: finalizing weak survey metadata dimensions too aggressively.

## Score Table

| Run | Primary Grain Contract Quality | Dimension Decision Quality | Family Decision Quality | Contract Conservatism and Coherence | Operational Usefulness | Catastrophic Contract Error | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `run_1_fifa` | 5 | 4 | 5 | 4 | 4 | No | Pass |
| `run_2_statistics` | 5 | 4 | 5 | 5 | 5 | No | Pass |
| `run_3_financial` | 4 | 4 | 5 | 4 | 4 | No | Pass |
| `run_4_responseorder` | 5 | 3 | 5 | 4 | 4 | No | Pass |

Scoring rubric:
- `5`: strong
- `4`: good with minor issues
- `3`: usable but notable weaknesses
- `2`: poor / risky
- `1`: broken

## Findings

### 1. `run_4_responseorder`: the contract still finalizes one weak dimension too aggressively
- Main risk: `dim_debrief` is marked `accept` even though the semantics are explicitly uncertain and the sample rows do not make it look like a clearly reusable entity.
- Why it matters: this is the clearest case where the light-contract step should have narrowed the grain output more aggressively by downgrading a weak dimension to `unsure`.

### 2. `run_1_fifa`: `positions_dim` is accepted even though it looks optional rather than structurally necessary
- Main risk: `Best Position` is a stable category, but the contract treats it as a fully accepted dimension rather than a plausible optional normalization target.
- Why it matters: this is not a structural error, but it shows the contract still inherits some over-eager dimensioning from the grain stage.

### 3. `run_3_financial`: the light-contract step improves on grain by downgrading `products` to `unsure`
- Main strength: this is the cleanest example of the contract doing useful narrowing rather than just repeating grain output.
- Remaining risk: `customers` and `payment_methods` are still accepted with limited evidence beyond many-to-one reuse, but that is materially safer than accepting `Product_Name`.

### 4. `run_2_statistics`: the contract handles uncertainty correctly on the generic `q` family
- Main strength: the strong families are accepted, while the generic `q` family is left `unsure`.
- Why it matters: this is the right contract behavior for a wide survey dataset where most structure is clear but one repeated block is semantically weaker.

### 5. Across all four runs, the light-contract step is stronger than the grain step at family finalization than at dimension restraint
- Main strength: family decisions are consistently high-quality.
- Main weakness: dimension promotion is still the most common overreach, especially on stable but weakly semantic attributes.

## Per-Run Notes

### `run_1_fifa`
- Primary grain contract quality: **5**
  - Strength: `ID` is correctly finalized as the primary grain with no drift from the grain-stage judgment.
  - Risk: none at the grain level.
- Dimension decision quality: **4**
  - Strength: `Club` and `Nationality` are defensible accepted dimensions.
  - Risk: `positions_dim` is accepted where `unsure` would have been more conservative.
- Family decision quality: **5**
  - Strength: `family_decisions = []` is correct for a flat player-profile dataset.
  - Risk: none.
- Contract conservatism and coherence: **4**
  - Strength: no user overrides were needed, and the contract remains internally coherent.
  - Risk: dimension acceptance is slightly more assertive than necessary.
- Operational usefulness: **4**
  - Strength: the contract is specific enough to constrain downstream typing safely.
  - Risk: optional normalization ideas are presented as stronger commitments than they need to be.

### `run_2_statistics`
- Primary grain contract quality: **5**
  - Strength: `ID` is correctly finalized as the respondent grain.
  - Risk: none.
- Dimension decision quality: **4**
  - Strength: `Grad_Country`, `Grad_Prov`, and `Mjr1` are reasonable accepted dimensions.
  - Risk: they are still somewhat design-oriented, but defensible.
- Family decision quality: **5**
  - Strength: `a_1`, `a_2`, `m_1`, and `m_2` are accepted, while the generic `q` family is kept `unsure`.
  - Risk: none material.
- Contract conservatism and coherence: **5**
  - Strength: this is the best-balanced contract of the four runs; it resolves the clear structure and preserves uncertainty where needed.
  - Risk: none material.
- Operational usefulness: **5**
  - Strength: this contract is immediately usable for the type worker with minimal reviewer correction.
  - Risk: none material.

### `run_3_financial`
- Primary grain contract quality: **4**
  - Strength: `Transaction_ID` is finalized correctly despite duplicate/export noise.
  - Risk: the duplicate-noise interpretation is still an assumption that would need validation before execution.
- Dimension decision quality: **4**
  - Strength: the contract improves on grain by marking `products` as `unsure` instead of `accept`.
  - Risk: `customers` and `payment_methods` are still accepted on fairly light evidence.
- Family decision quality: **5**
  - Strength: `family_decisions = []` is correct and avoids invented repeated structure.
  - Risk: none.
- Contract conservatism and coherence: **4**
  - Strength: the contract narrows one weak grain-stage proposal and stays internally consistent.
  - Risk: it still leans toward normalization on a dirty flat export.
- Operational usefulness: **4**
  - Strength: this is usable for downstream typing and safer than the raw grain output.
  - Risk: a human should still validate deduplication assumptions and the accepted dimensions.

### `run_4_responseorder`
- Primary grain contract quality: **5**
  - Strength: `RespondentId` is finalized correctly and the export index is excluded.
  - Risk: none.
- Dimension decision quality: **3**
  - Strength: `dim_language` is a sensible accepted dimension and `dim_participant_group` is correctly left `unsure`.
  - Risk: `dim_debrief` is accepted even though it looks more like weak survey metadata than a clearly reusable entity.
- Family decision quality: **5**
  - Strength: the RowN matrix families are correctly accepted and anchored to `RespondentId`.
  - Risk: none.
- Contract conservatism and coherence: **4**
  - Strength: the contract largely narrows uncertainty appropriately and preserves the central survey structure.
  - Risk: one weak accepted dimension keeps it from being fully conservative.
- Operational usefulness: **4**
  - Strength: this contract is strong enough to safely constrain the type worker on the important structure.
  - Risk: dimension cleanup would improve downstream focus.

## Cross-Run Synthesis

- Robust behavior:
  - primary grain finalization is strong across all four runs
  - flat datasets correctly keep `family_decisions = []`
  - survey-style datasets correctly finalize repeated families
  - no user override notes were needed in any run
- Improvement over grain stage:
  - the contract step does real narrowing, not just pass-through
  - the clearest example is `run_3_financial`, where `Product_Name` is downgraded to `unsure`
  - the statistics contract also handles the generic `q` family correctly by keeping it uncertain
- Most common failure mode:
  - accepted dimensions are still slightly too broad
  - the contract is better than grain at this, but not fully disciplined yet

## Overall Verdict

**Ready to proceed.**

The light-contract stage passes the acceptance criteria:
- no catastrophic contract errors
- all 4 runs are `Pass`
- flat datasets kept `family_decisions = []`
- survey-like datasets accepted the real families and preserved uncertainty where it mattered most
- the finalized contracts are specific enough to constrain the type worker safely

The next tightening pass should focus on one issue:
- make dimension acceptance more conservative, especially for weak survey metadata and optional categorical entities.
