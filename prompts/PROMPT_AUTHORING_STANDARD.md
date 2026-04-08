# System Prompt Authoring Standard

Use this document when creating or editing any `prompts/*system_prompt.md` file.

The goal is to keep worker prompts:
- structurally consistent,
- aligned with validators and schemas,
- explicit about evidence precedence,
- explicit about what the worker is and is not allowed to do,
- rich enough that later workers do not drift into underspecified judgment.

This standard is intentionally stricter for later-stage workers such as:
- `family_worker`
- `table_layout_worker`
- `analysis_layout_worker`
- `canonical_contract_reviewer`

Those workers consume more reviewed context, make more coupled judgments, and therefore need more detailed guidance than early-stage prompts.

## Required Section Skeleton

Every system prompt should contain these sections in a recognizable form:

1. `PROJECT CONTEXT`
2. `ROLE`
3. `WORKFLOW POSITION`
4. `INPUT`
5. `HIGHEST-PRECEDENCE RULE`
6. `DEFINITIONS`
7. `WHAT YOU OWN VS WHAT YOU DO NOT OWN`
8. `ALLOWED OUTPUT ENUMS / HARD FIELD CONSTRAINTS`
9. `ARTIFACT / INPUT SEMANTICS`
10. `DECISION PROCEDURE`
11. `EXAMPLES (positive, negative, and conflict cases)`
12. `OUTPUT SCHEMA`
13. `FINAL OUTPUT CONSTRAINTS`

Exact numbering can vary, but the prompt should clearly include each section.

## Section Expectations

### 1) PROJECT CONTEXT
This section should explain:
- the broader project purpose,
- how the worker fits into the one-dataset restructuring pipeline,
- what has already happened upstream,
- what later stages will consume from this worker.

Do not assume the model understands project-local terms such as:
- light contract
- canonical table layout
- canonical column contract
- analysis mart
- family packet

If a worker depends on a project-local concept, define it.

### 2) ROLE
This section should state:
- what the worker does,
- what kind of judgment it is expected to make,
- what kind of output it is expected to return,
- what kinds of overreach are forbidden.

### 3) WORKFLOW POSITION
This section should explain:
- what stages have already happened,
- what later stages depend on this worker,
- why overreach here is costly.

For later workers, explicitly name the reviewed upstream JSON layers that should already exist.

### 4) INPUT
This section should enumerate:
- every reviewed upstream JSON object,
- every bundled artifact,
- any loop-scoped inputs,
- any skip sentinels or migration-era aliases.

If an input can appear in legacy form, state how to interpret it.

### 5) HIGHEST-PRECEDENCE RULE
This section should:
- define the strict evidence hierarchy,
- state that reviewed evidence outranks raw artifact hints,
- state what must be preserved when conflicts occur,
- give at least one conflict example.

### 6) DEFINITIONS
This section should define project-local or worker-local concepts.

Definitions should be operational, not vague. Good definitions answer:
- what the thing is,
- how the worker should use it,
- what the worker must not confuse it with.

### 7) WHAT YOU OWN VS WHAT YOU DO NOT OWN
This section prevents worker overreach.

It must clearly distinguish:
- allowed decisions,
- forbidden redesigns,
- forbidden inference,
- output responsibilities that belong to later workers.

### 8) ALLOWED OUTPUT ENUMS / HARD FIELD CONSTRAINTS
This section should list all constrained enum surfaces the worker can emit or edit.

When a worker edits a nested reviewed contract, enumerate the editable enum surfaces from the relevant schema or validator.

Also include non-enum hard constraints when they materially affect model behavior:
- uniqueness rules,
- required blank/non-blank rules,
- order preservation rules,
- allowed skip sentinels,
- JSON-pointer rules,
- row coverage rules.

### 9) ARTIFACT / INPUT SEMANTICS
This section should describe every important reviewed JSON input and every bundled artifact the worker receives.

Use this template:

`<input_or_artifact_name>`:
- What it is:
- Why it matters:
- What not to use it for:
- Precedence rank:

This is the section that tells the model where to look for further evidence.

For later workers, do not just list artifacts. Explain how they should be used.

### 10) DECISION PROCEDURE
This section should be a step-by-step judgment workflow.

It should specify:
- where to start,
- how to resolve conflicts,
- when to stay conservative,
- when to set review flags,
- what must be preserved,
- what must be recomputed or cross-checked before output.

### 11) EXAMPLES
Examples should cover judgment families, not every literal dataset permutation.

Every worker should include:
- positive examples,
- negative examples,
- conflict-resolution examples.

Later workers should also include:
- reviewed-evidence-beats-raw-artifact examples,
- “no justified change” or “no justified output” examples where applicable.

Good examples should describe:
- the input evidence pattern,
- the correct judgment,
- the incorrect tempting judgment to avoid when relevant.

### 12) OUTPUT SCHEMA
This section should:
- show the exact top-level JSON shape,
- specify fixed worker names,
- specify required arrays/objects,
- specify any strict path or coverage requirements.

Do not change runtime JSON shape in prompt-only edits unless the validator/schema has also been intentionally changed.

### 13) FINAL OUTPUT CONSTRAINTS
This section is the final defense against format drift.

It should restate:
- one JSON object only,
- no markdown,
- no prose outside the JSON,
- no invented enum values,
- no silent structural redesign.

## Artifact Semantics Rules

For workers that consume multiple evidence sources:
- explicitly identify the primary evidence source,
- explicitly identify supporting-only evidence,
- explicitly identify caution-only evidence,
- explicitly identify evidence that must never override reviewed structure.

Examples:
- use `A16` as first-class structural-validity evidence for missingness questions
- use `A4` as the main missingness magnitude and token layer
- use `A13` and `A14` as supporting context only unless the worker explicitly needs them as primary evidence
- use reviewed worker JSON outputs above raw artifacts

## Example Coverage Rules

The minimum acceptable example coverage is:

- Early workers: at least one positive and one negative example family
- Later workers: at least one positive, one negative, and one conflict-resolution example family

Strong later-worker prompts should cover most major judgment families the worker can make.

Examples should be dataset-agnostic and reusable across fixtures.

## Validator Crosswalk Rules

When a worker emits constrained JSON:
- read the relevant validator or schema before editing the prompt,
- explicitly surface the enums the worker can emit,
- explicitly surface important structural invariants that are easy for the model to violate.

Examples:
- uniqueness of `table_name` or `output_table_name`
- blank-only-when-excluded target fields
- required `wave_column` constraints
- exact skip sentinel values
- order preservation and exhaustive row coverage

## Later-Worker Discipline

Later prompts are at the highest risk of drift because they:
- depend on many upstream artifacts,
- combine structural and semantic evidence,
- make more global decisions,
- are more likely to be underexplained.

For later workers:
- define project-local concepts explicitly,
- define each reviewed input and artifact explicitly,
- define every important decision family explicitly,
- include conflict-resolution examples,
- explicitly state when reviewed evidence outranks raw artifact hints,
- explicitly state when “no change” or “no derived output” is the correct answer.

## Anti-Patterns To Avoid

Do not write prompts that:
- merely list artifacts without explaining how to use them,
- refer to local project objects without defining them,
- assume the model knows validator constraints,
- rely on prose-only descriptions when enums or hard invariants exist,
- include only cheerful best-case examples,
- bury the real evidence hierarchy in passing wording,
- ask the model to “use judgment” without specifying what good judgment means for that worker.

## Required Review Before Merging Prompt Edits

Before merging prompt edits:
- run the prompt audit script,
- check the prompt against the relevant validator/schema,
- confirm the prompt still matches runtime JSON shape,
- confirm examples cover the major decision families,
- confirm later workers document every bundled artifact and reviewed upstream JSON they receive.
