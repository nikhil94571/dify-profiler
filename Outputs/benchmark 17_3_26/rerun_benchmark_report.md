# Rerun Benchmark Report (17/3/26)

## Scope

This report benchmarks the refreshed reruns in:

- `run_1_fifa`
- `run_4_responseorder`

against the 16/3/26 baseline runs. The purpose is not to repeat the full 4-run evaluation. It is to validate the specific hardening changes that were implemented after the first benchmark:

- stricter `type_transform_worker` JSON discipline
- more conservative identifier assignment
- tighter wide-survey pruning for the type-worker bundle

## Executive Summary

The reruns materially improve the two target failure modes.

- `run_1_fifa`: the hard operational failure is fixed. The type worker now emits valid JSON and no longer promotes `Name`, `LongName`, `photoUrl`, or `playerUrl` to identifiers.
- `run_4_responseorder`: the wide-survey bundle is materially tighter. Bundle size drops from `115975` bytes to `59454` bytes, driven mainly by `A2` shrinking from `90` rows to `32`.

Overall verdict:

- the hardening pass is **validated**
- the remaining work is now narrower:
  - residual semantic conservatism in the type worker
  - residual `A4` over-retention on very wide survey exports

## Dataset Notes

### `run_1_fifa`

- Sample file: `fifa_v2_sample_rows.csv`
- Shape from sample: `30` sample rows, `77` columns
- Structure: flat entity-style player profile export
- Expected grain: one row per player, keyed by `ID`
- Main semantic risk: unique-looking descriptive fields and URLs being overtyped as identifiers

### `run_4_responseorder`

- Sample file: `response_order_raw_sample_rows.csv`
- Shape from sample: `30` sample rows, `267` columns
- Structure: wide survey export with matrix/family columns and respondent-level metadata
- Expected grain: one row per respondent response, keyed by `RespondentId`
- Main operational risk: over-retention of family-wide artifacts inflating the type-worker bundle

## Results Table

| Run | Grain Status | Contract Status | Bundle Delta vs 16/3/26 | Type Worker Delta vs 16/3/26 | Verdict |
| --- | --- | --- | --- | --- | --- |
| `run_1_fifa` | Stable | Stable | Slightly smaller (`37839 -> 37064`) | Invalid JSON fixed; identifier overreach fixed | Pass |
| `run_4_responseorder` | Stable | Stable | Much smaller (`115975 -> 59454`) | Output remains valid and more focused | Pass |

## Detailed Findings

### 1. `run_1_fifa` operational reliability is fixed

This was the clearest blocking failure in the original benchmark. The 16/3/26 type output could not be parsed as JSON because it contained an invalid numeric literal. The refreshed 17/3/26 rerun produces valid JSON.

Impact:

- the type-worker stage is now operationally usable on this dataset
- downstream validation no longer fails at basic parsing

Evidence:

- `type_output.json` now parses successfully
- output contains `18` column decisions and `9` review flags

### 2. `run_1_fifa` identifier overreach is materially improved

The refreshed type output no longer classifies obvious descriptive or URL fields as identifiers.

Corrected columns:

- `Name -> nominal_category`
- `LongName -> nominal_category`
- `photoUrl -> nominal_category`
- `playerUrl -> nominal_category`

This is a meaningful semantic improvement. It addresses the main critique from the original run: uniqueness or high distinctness is no longer being treated as sufficient evidence for `identifier`.

What still needs review:

- `photoUrl` and `playerUrl` are no longer mislabeled as identifiers, but `nominal_category` is still a somewhat blunt type for URL-like fields
- `Height`, `Weight`, `Hits`, `Release Clause`, `Value`, and `Wage` are all correctly pushed into reviewed numeric transforms, which is operationally useful

### 3. `run_1_fifa` pruning changed only marginally, which is acceptable

The FIFA dataset was not the main pruning target. The bundle shrank only slightly:

- `37839 -> 37064` bytes

Artifact scope is effectively unchanged:

- `A2`: `18 -> 18`
- `A4`: `16 -> 16`
- `A9`: `18 -> 18`
- `A13`: `16 -> 16`
- `A14`: `17 -> 17`

Interpretation:

- this is acceptable
- the run did not exhibit the wide-survey spillover pattern that motivated the pruning changes
- the slight reduction is consistent with empty-field cleanup rather than structural scope changes

### 4. `run_4_responseorder` pruning improved materially

This was the main pruning regression target, and the rerun shows a clear improvement.

Bundle size:

- `115975 -> 59454` bytes
- reduction: `-56521` bytes

Most important artifact delta:

- `A2`: `90 -> 32`

Unchanged artifact counts:

- `A4 per_column`: `80 -> 80`
- `A9 columns`: `11 -> 11`
- `A13 columns`: `6 -> 6`
- `A14 columns`: `7 -> 7`
- `A16 detected skip rules`: `6 -> 6`

Interpretation:

- the new family-cap pruning is working where it matters most
- the reduction came almost entirely from cutting `A2` family spillover
- `A4` remains broad, so there is still residual dead weight in wide-survey runs

This is still a successful result. The original problem was bundle bloat large enough to distort type-worker efficiency. That is materially reduced here.

### 5. `run_4_responseorder` type-worker scope is tighter and more defensible

The refreshed type output contains `11` column decisions and `4` review flags. It remains valid JSON and is better calibrated than the baseline on at least two important columns.

Meaningful improvements:

- `Unnamed: 0`
  - old: `identifier`
  - new: `mixed_or_ambiguous`
  - now flagged for review with `potential_id_conflict_with_primary_grain`

- `Q12`
  - old: `ordinal_category`
  - new: `nominal_category`
  - now flagged for review instead of being treated as implicitly ordered

- `debrief`
  - old: `categorical_code`
  - new: `nominal_category`
  - more conservative and more defensible for unclear survey metadata

This is the right direction. The worker is using weaker evidence more conservatively.

### 6. `run_4_responseorder` still has residual semantic overreach

Two areas still need tightening:

- `participant` remains an accepted contract dimension and a reviewed `nominal_category`, but its semantics are still unclear
- `Q13Main_cell_groupRow1` and `Q19Main_cell_groupRow1` are still emitted as `ordinal_category`

The ordinal calls may be defensible if these are true matrix-scale responses, but the current evidence packet still does not make the ordering semantics explicit. That means the output is improved, but not yet maximally conservative.

This is no longer a blocking issue. It is a refinement issue.

## Metrics

### `run_1_fifa`

#### Grain

- primary grain: `ID`
- confidence: `0.99`
- candidate dimensions: `3`
- family candidates: `0`

#### Type worker bundle

- size: `37064` bytes
- artifact ids: `A2, A3-T, A3-V, A4, A9, A13, A14, A16`

#### Type worker output

- valid JSON: yes
- column decisions: `18`
- review flags: `9`
- prompt tokens: `17474`
- completion tokens: `7068`
- total price: `$0.0185045`
- latency: `95.064s`

### `run_4_responseorder`

#### Grain

- primary grain: `RespondentId`
- confidence: `0.9`
- candidate dimensions: `2`
- family candidates: `18`

#### Type worker bundle

- size: `59454` bytes
- artifact ids: `A2, A3-T, A3-V, A4, A9, A13, A14, A16`

#### Type worker output

- valid JSON: yes
- column decisions: `11`
- review flags: `4`
- prompt tokens: `23589`
- completion tokens: `6009`
- total price: `$0.0179153`
- latency: `83.328s`

## Final Verdict

### `run_1_fifa`

**Pass**

Reason:

- the hard JSON failure is fixed
- the worst identifier overreach is fixed
- no new regression is visible in grain or contract behavior

### `run_4_responseorder`

**Pass**

Reason:

- the wide-survey type bundle is materially smaller
- the worker output is still valid and more conservative on several ambiguous fields
- remaining issues are refinement issues, not blocking failures

## Overall Conclusion

The 17/3/26 reruns validate the hardening pass.

What is now clearly improved:

- strict JSON reliability on the previously failing FIFA run
- non-primary unique-field handling in the type worker
- wide-survey pruning efficiency on the response-order run

What still remains:

- `A4` is still too broad on very wide survey exports
- some ordinal and survey-metadata semantics are still a little too eager

Recommended next step:

1. accept this hardening pass as successful
2. do one smaller follow-up pass focused on:
   - `A4` pruning discipline
   - stricter ordinal evidence requirements
3. do not reopen grain or light-contract work based on these reruns
