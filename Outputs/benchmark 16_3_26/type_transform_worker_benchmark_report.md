# Type Transform Worker Benchmark Report

## Evaluation Frame

Inputs used per run:
- sample rows CSV
- finalized light contract
- pruned type-worker bundle
- type-worker output

Primary questions:
- Did the worker respect the light contract?
- Were the type/storage decisions semantically defensible?
- Did it stay conservative when evidence was weak?
- Was the output operationally usable as strict JSON?

## Score Table

| Run | JSON / Schema Adherence | Structural Precedence and Scope Discipline | Semantic Type / Transform Quality | Conservatism and Calibration | Operational Usefulness | Catastrophic Worker Error | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `run_1_fifa` | 1 | 2 | 3 | 2 | 2 | Yes | Fail |
| `run_2_statistics` | 5 | 4 | 4 | 4 | 4 | No | Pass |
| `run_3_financial` | 5 | 4 | 4 | 4 | 4 | No | Pass |
| `run_4_responseorder` | 5 | 4 | 4 | 3 | 4 | No | Pass |

Scoring rubric:
- `5`: strong
- `4`: good with minor issues
- `3`: usable but notable weaknesses
- `2`: poor / risky
- `1`: broken

## Findings

### 1. `run_1_fifa`: the worker emitted invalid JSON
- Main failure: the output is not parseable because it contains an invalid numeric literal:
  - `"confidence": 0. nine`
- Why it matters: this is an immediate operational failure. Even if the surrounding reasoning is mostly salvageable, downstream systems cannot safely consume the result.

### 2. `run_1_fifa`: the worker over-classifies names and URLs as identifiers
- Main risk:
  - `Name -> identifier`
  - `LongName -> identifier`
  - `photoUrl -> identifier`
  - `playerUrl -> identifier`
- Why it matters: `ID` is the true primary identifier. The others are row attributes or external locators. Treating them as identifiers is a semantic overreach and suggests the worker is still too willing to equate uniqueness with identifier semantics.

### 3. `run_2_statistics`: the worker handles family structure well, but slightly expands scope via A16 trigger fields
- Main behavior:
  - outputs `ANXATT`, `MATHATT`, and `Age` even though they were not direct light-contract fields
  - these are grounded in `A16` trigger evidence, not hallucinated from nowhere
- Why it matters: this is mostly defensible, but it shows the worker is willing to surface additional structurally relevant fields beyond the explicit contract. That is acceptable only if kept disciplined.

### 4. `run_2_statistics`: uncertainty handling is generally strong
- Main strengths:
  - preserves `ID` as identifier string
  - keeps repeat-family members as repeat-context coded fields
  - marks `Q1` as `mixed_or_ambiguous`
  - does not over-assert skip-logic protection on `M2_Q6`
- Why it matters: this is the best overall example of the worker using both the light contract and the pruned bundle correctly.

### 5. `run_3_financial`: mostly good, but `Transaction_Status` is likely over-typed as ordinal
- Main risk:
  - `Transaction_Status -> ordinal_category`
- Why it matters: without a clear ordered codebook, statuses like `Pending` / `completed` are safer as nominal categories. The worker does flag review, which limits the damage, but the semantic default is still too assertive.

### 6. `run_4_responseorder`: the worker is useful, but still too permissive with redundant index fields
- Main risk:
  - `Unnamed: 0 -> identifier` with no human review
- Why it matters: the worker does add a review flag noting redundant index risk, but the main decision is still too confident for a likely export-index field that the light contract did not choose as the base grain.

### 7. Across the valid runs, the worker is better at transforms than at pure semantic restraint
- Main strength:
  - transform and storage recommendations are usually sensible and conservative
  - examples:
    - `Transaction_Date -> date` with parse-failure review
    - `CompletedDate/StartDate -> datetime`
    - `debrief` flagged for multiselect modeling
    - survey family members sent to child-table review
- Main weakness:
  - semantic category choice still sometimes overreaches
  - examples:
    - `Transaction_Status -> ordinal_category`
    - `Unnamed: 0 -> identifier`
    - `Name/LongName -> identifier`

## Per-Run Notes

### `run_1_fifa`
- JSON / schema adherence: **1**
  - Strength: the output is close to valid and only appears to fail on a literal formatting error.
  - Risk: strict JSON validity is still broken, which is enough to fail the run operationally.
- Structural precedence and scope discipline: **2**
  - Strength: `ID` is preserved correctly as the primary identifier.
  - Risk: `Name`, `LongName`, `photoUrl`, and `playerUrl` are all treated as identifiers, which shows weak semantic discipline around non-primary unique fields.
- Semantic type / transform quality: **3**
  - Strength: `Height`, `Weight`, `Joined`, `Contract`, and currency fields are handled in a mostly sensible way.
  - Risk: the identifier overreach materially lowers trust in the column typing.
- Conservatism and calibration: **2**
  - Strength: some ambiguous fields such as `Contract`, `Positions`, `Hits`, and `Loan Date End` are reviewed.
  - Risk: the worker is too confident on non-primary identifiers that should have been normal attributes or reviewed.
- Operational usefulness: **2**
  - Strength: after manual repair, much of the output is interpretable.
  - Risk: as delivered, it is not machine-consumable.

### `run_2_statistics`
- JSON / schema adherence: **5**
  - Strength: valid JSON, correct top-level shape, and coherent enums.
  - Risk: none material.
- Structural precedence and scope discipline: **4**
  - Strength: the worker respects the light contract and preserves repeat-family structure rather than collapsing it into base-table attributes.
  - Risk: it expands scope slightly by surfacing `ANXATT`, `MATHATT`, and `Age` from `A16` trigger evidence.
- Semantic type / transform quality: **4**
  - Strength: `ID`, geographic fields, family members, and gating fields are handled sensibly.
  - Risk: some label-vs-code distinctions remain slightly too coarse, especially for `Grad_Country` and `Mjr1`.
- Conservatism and calibration: **4**
  - Strength: uncertainty is preserved on `Mjr2`, `M2_Q6`, and especially `Q1`.
  - Risk: skip-logic protection is still used fairly broadly, though mostly defensibly here.
- Operational usefulness: **4**
  - Strength: this is a workable type-transform output with targeted review flags.
  - Risk: it still needs some reviewer discipline around trigger-field scope.

### `run_3_financial`
- JSON / schema adherence: **5**
  - Strength: valid JSON and clean schema.
  - Risk: none.
- Structural precedence and scope discipline: **4**
  - Strength: the worker respects the accepted grain and accepted/unsure dimension decisions.
  - Risk: `Customer_ID` is treated as an identifier rather than just a dimension key, which is defensible but slightly stronger than necessary.
- Semantic type / transform quality: **4**
  - Strength: `Payment_Method`, `Product_Name`, and `Transaction_Date` are handled conservatively and with useful review flags.
  - Risk: `Transaction_Status` is probably nominal rather than ordinal.
- Conservatism and calibration: **4**
  - Strength: low-confidence and dirty fields are marked for review instead of being silently normalized.
  - Risk: the ordinal interpretation shows some residual overreach.
- Operational usefulness: **4**
  - Strength: this output is easy to consume and would be useful downstream.
  - Risk: a reviewer should correct the status semantics before locking the contract.

### `run_4_responseorder`
- JSON / schema adherence: **5**
  - Strength: valid JSON and strong schema adherence.
  - Risk: none.
- Structural precedence and scope discipline: **4**
  - Strength: the worker respects `RespondentId` as the main grain and handles family members as repeat-context fields with child-table hints.
  - Risk: `Unnamed: 0` is still promoted as an identifier despite being explicitly non-primary and likely redundant.
- Semantic type / transform quality: **4**
  - Strength: `CompletedDate`, `StartDate`, `debrief`, `Q13Main_cell_groupRow1`, and `Q2_Other_Other__please_specify` are handled in mostly sensible ways.
  - Risk: `Unnamed: 0` remains the clearest semantic misstep.
- Conservatism and calibration: **3**
  - Strength: `participant` and `Q2_Other_Other__please_specify` are reviewed appropriately.
  - Risk: the worker is still too comfortable assigning identifier semantics to likely export-index columns.
- Operational usefulness: **4**
  - Strength: the output is valid, useful, and structurally aligned with the light contract.
  - Risk: index-field handling still needs human oversight.

## Cross-Run Synthesis

- Strong behavior:
  - valid runs generally respect the light contract
  - transform and storage recommendations are usually sensible
  - repeat-family handling is solid on both survey-style datasets
  - review flags are generally relevant and actionable
- Main remaining weakness:
  - the worker is still too willing to assign strong semantics to fields that are merely unique, index-like, or code-like
  - this shows up as:
    - non-primary identifiers on FIFA
    - `Unnamed: 0` on response-order
    - ordinal overreach on financial status values
- Scope note:
  - the worker sometimes surfaces extra columns from `A16` trigger evidence
  - this is not necessarily wrong, but it should remain explicit and tightly bounded
- Reliability note:
  - one invalid JSON run is enough to keep this stage out of the “ready” category

## Overall Verdict

**Usable but needs tightening.**

The type-transform stage is not ready to declare stable yet because:
- 1 of 4 runs is a hard operational failure due to invalid JSON
- semantic overreach still appears on unique/index-like columns
- ordinal/category distinctions are not consistently conservative

The next tightening pass should focus on three things:
- harden strict JSON reliability
- make identifier semantics more conservative for non-primary unique fields and export-index columns
- make ordinal assignment stricter unless there is strong explicit evidence
