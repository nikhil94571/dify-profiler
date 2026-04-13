# Grain Worker Benchmark Report

## Dataset Expectations

### `run_1_fifa`
- Likely unit of observation: one player profile per row.
- Obvious key-like column: `ID`.
- Repeated-block expectation: none.
- Structural shape: flat entity/master-data table with many descriptive attributes.

### `run_2_statistics`
- Likely unit of observation: one student/respondent per row.
- Obvious key-like column: `ID`.
- Repeated-block expectation: strong wide survey-style families (`A1_*`, `A2_*`, `M1_*`, `M2_*`, `Q*`).
- Structural shape: respondent base table plus repeat-family blocks encoded wide.

### `run_3_financial`
- Likely unit of observation: one transaction per row.
- Obvious key-like column: `Transaction_ID`.
- Repeated-block expectation: none.
- Structural shape: flat transactional table with dirty values and duplicate/export noise.

### `run_4_responseorder`
- Likely unit of observation: one survey response/submission per row.
- Obvious key-like column: `RespondentId`.
- Repeated-block expectation: very strong `RowN` matrix families.
- Structural shape: wide survey export with repeated matrix/question groups.

## Score Table

| Run | Primary Grain Correctness | Dimension Boundary Quality | Family / Repeated-Structure Quality | Conservatism and Calibration | Operational Usefulness | Catastrophic Structural Error | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `run_1_fifa` | 5 | 4 | 5 | 4 | 4 | No | Pass |
| `run_2_statistics` | 5 | 4 | 5 | 4 | 5 | No | Pass |
| `run_3_financial` | 4 | 3 | 5 | 3 | 3 | No | Borderline |
| `run_4_responseorder` | 5 | 3 | 5 | 4 | 4 | No | Pass |

Scoring rubric:
- `5`: strong
- `4`: good with minor issues
- `3`: usable but notable weaknesses
- `2`: poor / risky
- `1`: broken

## Findings

### 1. `run_3_financial`: candidate dimensions are too eager on messy flat data
- Main risk: `Product_Name` is proposed as a dimension key without a supporting `Product_ID`, even though the sample rows show dirty labels such as trailing spaces and generic values like `Coffee `.
- Why it matters: this is the clearest case where the worker moves from structural inference into design suggestion without enough evidence. It does not break the grain choice, but it makes the dimension layer less trustworthy.

### 2. `run_4_responseorder`: dimension suggestions extend beyond clearly defensible entities
- Main risk: `participant` and `debrief` are suggested as dimensions largely because they are stable invariant attributes, not because they are clearly reusable entities.
- Why it matters: family detection is correct, but candidate dimensions are broader than they need to be. This weakens light-contract review by introducing unnecessary table ideas.

### 3. `run_1_fifa`: minor over-structuring on optional dimensions
- Main risk: `Best Position` as `positions_dim` is plausible, but it is less compelling than `Club` or `Nationality` as a separate dimension.
- Why it matters: the worker is directionally correct, but it shows a general tendency to turn stable categories into dimension proposals even when normalization is optional rather than structurally motivated.

### 4. `run_3_financial`: duplicate-noise interpretation is plausible but still assumption-heavy
- Main strength: the worker does not abandon `Transaction_ID` just because the dataset is dirty.
- Main risk: it leans on the assumption that duplicate `Transaction_ID` rows are export noise. That is reasonable from the artifacts, but still needs explicit validation.

### 5. `run_2_statistics` and `run_4_responseorder`: family handling is the strongest part of the grain worker
- Main strength: both survey-style datasets were recognized as wide repeated structures, and the worker correctly rejected fake grains manufactured from question columns.
- Why it matters: this is the most important structural capability for the current workflow, and it generalized well across two different wide survey patterns.

## Per-Run Notes

### `run_1_fifa`
- Primary grain correctness: **5**
  - Strength: `ID` is correctly selected as the semantic player key and URL-like fields are explicitly rejected as fake identifiers.
  - Risk: none at the grain level.
- Dimension boundary quality: **4**
  - Strength: `Club` and `Nationality` are sensible optional dimensions.
  - Risk: `Best Position` is more of a reusable category than a clearly necessary separate entity.
- Family / repeated-structure quality: **5**
  - Strength: no fake families were invented on a flat entity table.
  - Risk: none.
- Conservatism and calibration: **4**
  - Strength: high confidence on `ID` is justified by uniqueness and role evidence.
  - Risk: optional dimensioning is slightly eager.
- Operational usefulness: **4**
  - Strength: review questions are concrete and downstream-safe.
  - Risk: some table suggestions are a little more opinionated than necessary.

### `run_2_statistics`
- Primary grain correctness: **5**
  - Strength: `ID` as respondent grain is correct and repeated question columns are correctly excluded from the key.
  - Risk: none at the grain level.
- Dimension boundary quality: **4**
  - Strength: `Grad_Country`, `Grad_Prov`, and `Mjr1` are reasonable dimension candidates.
  - Risk: grouped summary treatment of additional families is useful but somewhat compressed.
- Family / repeated-structure quality: **5**
  - Strength: detected all major survey blocks and kept the generic `Q*` family appropriately uncertain.
  - Risk: none.
- Conservatism and calibration: **4**
  - Strength: uncertainty is preserved for the generic `Q` family.
  - Risk: it still assumes a normalized repeat index `q` across multiple families without domain confirmation.
- Operational usefulness: **5**
  - Strength: this output is ready for light-contract review with minimal repair.
  - Risk: none material.

### `run_3_financial`
- Primary grain correctness: **4**
  - Strength: `Transaction_ID` is the right default grain despite dirty duplicates.
  - Risk: confidence is appropriately lower, but the noise interpretation still needs validation.
- Dimension boundary quality: **3**
  - Strength: `Customer_ID` and `Payment_Method` are plausible many-to-one structures.
  - Risk: `Product_Name` is too weak as a dimension key in the absence of a product identifier or clearer entity evidence.
- Family / repeated-structure quality: **5**
  - Strength: correctly recognized as a flat transaction table with no repeat-family invention.
  - Risk: none.
- Conservatism and calibration: **3**
  - Strength: the worker did not hallucinate a safer composite key when none existed.
  - Risk: duplicate-export and product-dimension assumptions are a bit too forward.
- Operational usefulness: **3**
  - Strength: usable as a transaction-first light-contract draft.
  - Risk: it needs explicit human validation before using dimension proposals or deduplication policy.

### `run_4_responseorder`
- Primary grain correctness: **5**
  - Strength: `RespondentId` is correctly chosen and the unnamed export index is correctly rejected.
  - Risk: none at the grain level.
- Dimension boundary quality: **3**
  - Strength: `LanguageCode` is a sensible small dimension.
  - Risk: `participant` and `debrief` are weak entity candidates and should probably remain attributes unless later evidence justifies separation.
- Family / repeated-structure quality: **5**
  - Strength: the RowN matrix families are identified strongly and used correctly to reject fake grains.
  - Risk: none.
- Conservatism and calibration: **4**
  - Strength: strong confidence on family structure is justified by the coverage and naming pattern evidence.
  - Risk: dimension expansion is still broader than necessary.
- Operational usefulness: **4**
  - Strength: review questions are targeted and useful for the light-contract stage.
  - Risk: candidate dimensions need pruning.

## Cross-Run Synthesis

- Robust behavior:
  - primary grain selection is strong across all four runs
  - fake keys are rejected reliably
  - repeat-family detection is the strongest capability and generalized well across both survey-like datasets
- Overfit risk:
  - the worker is not overfit to the current survey dataset on grain selection
  - the main generalization weakness is a broader pattern: stable invariant attributes are too readily turned into candidate dimensions
- Most common failure mode:
  - over-eager dimension suggestion on fields that are stable categories but not clearly reusable entities

## Overall Verdict

**Usable but needs prompt tightening.**

The grain worker passes the acceptance criteria:
- no catastrophic structural errors
- 3 of 4 runs are clear `Pass`
- the remaining run is a workable `Borderline`
- flat datasets did not get fake family structure
- survey-like datasets did get family structure recognized
- review questions are specific and operationally useful

The next tightening pass should focus on one issue only:
- make candidate dimensions more conservative, especially on flat transactional data and weakly semantic survey metadata.
