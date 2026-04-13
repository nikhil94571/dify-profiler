# Cost / Latency / Token Efficiency Report

## Scope

This report uses the saved metrics for:
- `grain_output.json` usage
- `type_worker_output.json` usage
- pruned type-worker bundle size from `type_worker_bundle.json`

Notes:
- the light-contract stage has no separate model-usage record in these saved artifacts
- all prices are in USD
- bundle size is measured from the serialized `type_transform_worker_bundle` JSON payload

## Combined Summary Table

| Run | Bundle Bytes | Grain Tokens | Type Tokens | Combined Tokens | Grain Cost | Type Cost | Combined Cost | Grain Latency (s) | Type Latency (s) | Combined Latency (s) | Type JSON Valid | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run_1_fifa` | 36,372 | 18,185 | 23,657 | 41,842 | 0.0706 | 0.0169 | 0.0875 | 36.0 | 96.5 | 132.6 | No | Borderline |
| `run_2_statistics` | 69,076 | 18,302 | 33,108 | 51,410 | 0.0932 | 0.0197 | 0.1129 | 46.0 | 93.9 | 139.9 | Yes | Pass |
| `run_3_financial` | 9,672 | 10,317 | 13,801 | 24,118 | 0.0507 | 0.0106 | 0.0613 | 29.8 | 55.1 | 84.9 | Yes | Pass |
| `run_4_responseorder` | 108,812 | 20,039 | 45,399 | 65,438 | 0.1056 | 0.0201 | 0.1257 | 53.5 | 79.2 | 132.7 | Yes | Borderline |

## Derived Efficiency Table

| Run | Column Decisions | Review Flags | Cost per Decision | Tokens per Decision | Bundle Bytes per Decision |
| --- | --- | --- | --- | --- | --- |
| `run_1_fifa` | 18 | 4 | 0.00486 | 2,324.6 | 2,020.7 |
| `run_2_statistics` | 13 | 7 | 0.00868 | 3,954.6 | 5,313.5 |
| `run_3_financial` | 6 | 3 | 0.01021 | 4,019.7 | 1,612.0 |
| `run_4_responseorder` | 10 | 5 | 0.01257 | 6,543.8 | 10,881.2 |

Interpretation:
- `cost per decision` and `tokens per decision` are rough comparative efficiency metrics, not quality metrics
- `bundle bytes per decision` is a useful proxy for how much context the worker carries for each final column decision

## Findings

### 1. `run_4_responseorder` is the least efficient run by a clear margin
- Main evidence:
  - largest bundle: `108,812` bytes
  - largest combined token count: `65,438`
  - highest total cost: `$0.1257`
  - worst `bundle bytes per decision`: `10,881.2`
  - worst `tokens per decision`: `6,543.8`
- Why it matters: this is not just “a large dataset costs more.” The ratio is poor because the type worker is carrying too much bundle context relative to the number of actual decisions it returns.

### 2. `run_3_financial` is the most efficient run overall
- Main evidence:
  - smallest bundle: `9,672` bytes
  - lowest combined tokens: `24,118`
  - lowest combined cost: `$0.0613`
  - lowest combined latency: `84.9s`
  - best `bundle bytes per decision`: `1,612.0`
- Why it matters: this is the cleanest demonstration that the current architecture can be efficient when the data shape is simple and the pruning is doing real work.

### 3. Grain cost is still the dominant cost center
- Across all four runs:
  - `grain_worker` cost ranges from `$0.0507` to `$0.1056`
  - `type_transform_worker` cost ranges from `$0.0106` to `$0.0201`
- In every run, grain is materially more expensive than type-transform.
- Why it matters: if you later want to reduce total workflow cost, the highest-leverage place is still the grain stage or the number of times it runs, not the type worker alone.

### 4. `run_1_fifa` is relatively cheap, but the efficiency result is misleading because the output failed
- Main evidence:
  - combined cost is modest: `$0.0875`
  - `cost per decision` looks good: `$0.00486`
  - but the type output is invalid JSON
- Why it matters: cheap output is not efficient if it is unusable. This run should not be treated as a real efficiency success.

### 5. `run_2_statistics` is reasonably efficient for a wide survey dataset
- Main evidence:
  - combined cost: `$0.1129`
  - combined tokens: `51,410`
  - bundle size: `69,076` bytes
  - valid output with 13 decisions and 7 review flags
- Why it matters: this looks like an acceptable cost profile for a genuinely complex wide dataset where the worker is doing useful structural interpretation.

### 6. Type-worker latency is consistently high relative to its cost
- Type-worker latencies:
  - `run_1_fifa`: `96.5s`
  - `run_2_statistics`: `93.9s`
  - `run_3_financial`: `55.1s`
  - `run_4_responseorder`: `79.2s`
- Why it matters: the worker is cheap on `GPT-5 mini`, but not fast. If user-facing responsiveness matters, latency may become a constraint before direct dollar cost does.

### 7. The efficiency pattern matches the pruning findings
- Strong alignment:
  - `run_3_financial` had the tightest bundle and the best efficiency profile
  - `run_4_responseorder` had the loosest bundle and the worst efficiency profile
- Why it matters: this confirms the pruning report is not just aesthetic criticism. Bundle discipline is directly driving token and cost efficiency.

## Per-Run Notes

### `run_1_fifa`
- Cost efficiency: **3**
  - Strength: moderate combined cost and token count.
  - Risk: the apparent efficiency is undermined by invalid JSON output.
- Latency efficiency: **2**
  - Strength: grain latency is moderate.
  - Risk: total latency `132.6s` is high for a failed run.
- Token efficiency: **3**
  - Strength: bundle size is moderate.
  - Risk: the tokens spent did not reliably produce a usable result.

### `run_2_statistics`
- Cost efficiency: **4**
  - Strength: the cost is reasonable given the dataset width and the amount of useful output.
  - Risk: still materially more expensive than flat-data cases.
- Latency efficiency: **3**
  - Strength: no major spike relative to complexity.
  - Risk: nearly `140s` combined is still slow.
- Token efficiency: **4**
  - Strength: token spend looks justified by bundle size and output usefulness.
  - Risk: there is still some survey-context overhead.

### `run_3_financial`
- Cost efficiency: **5**
  - Strength: lowest combined cost and strongest cost-to-usefulness ratio.
  - Risk: none material.
- Latency efficiency: **4**
  - Strength: best total latency of the four runs.
  - Risk: still not “fast,” but clearly better than the others.
- Token efficiency: **5**
  - Strength: the tight bundle translates directly into efficient prompt size and a small, focused output.
  - Risk: none material.

### `run_4_responseorder`
- Cost efficiency: **2**
  - Strength: still relatively cheap in absolute dollar terms.
  - Risk: worst cost per useful decision due to bundle over-retention.
- Latency efficiency: **3**
  - Strength: latency is not dramatically worse than the other large runs.
  - Risk: total runtime is still high relative to output volume.
- Token efficiency: **2**
  - Strength: output is valid and usable.
  - Risk: the bundle and prompt are carrying too much context for only 10 final column decisions.

## Cross-Run Synthesis

- Strong behavior:
  - the type worker remains low-cost in absolute terms on `GPT-5 mini`
  - simple flat datasets are already in a good efficiency range
  - bundle size and efficiency move together in the expected direction
- Main bottlenecks:
  - grain remains the largest direct cost center
  - type-worker latency remains high
  - wide-survey pruning quality still drives the worst token-efficiency outcomes
- Most important efficiency failure mode:
  - carrying too much bundle context for too few final decisions
  - this is most obvious in `run_4_responseorder`

## Overall Verdict

**Acceptable cost, but uneven efficiency.**

The current workflow is economically workable:
- all four runs stay close to or below about `$0.13` total for the saved grain + type stages
- the type worker is cheap in direct dollar terms

But the efficiency profile is not yet uniform:
- simple flat datasets are good
- medium survey data is acceptable
- very wide survey data is still inefficient because of bundle bloat
- one failed JSON run means some spend is still being wasted on unusable output

The next efficiency pass should focus on:
- reducing grain-stage cost only if total workflow cost becomes a real issue
- reducing type-worker latency if responsiveness matters
- tightening wide-survey pruning first, because that is the clearest token-efficiency lever
- hardening JSON reliability so cheap runs are still actually usable
