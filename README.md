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
6. eventually generate executable cleaning code and cleaned outputs in one or more target layouts

Today, this repository implements the artifact compiler and artifact serving layers. It does not yet expose a final cleaning-contract endpoint, code-generation endpoint, or execution endpoint for writing cleaned output tables.

## Current Philosophy

The service is designed around:

- deterministic evidence generation
- immutable artifact storage
- deterministic artifact adaptation at view time
- selective LLM consumption through worker-specific bundles

That means the canonical artifact in storage is treated as ground truth, while the LLM-facing response is a compiled view produced on demand through pruning, typed selection policies, and transforms.

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

The new `GET /artifact-bundles` endpoint is the simplest Dify-facing interface for specialist workers.

Example:

```http
GET /artifact-bundles?run_id=<run_id>&mode=grain_worker
```

If `artifact_ids` is omitted and `mode` maps to a named profile, the service infers the worker's artifact list from that profile.

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

## Current Dify Integration Model

The intended Dify pattern is:

1. upload a dataset to `POST /full-bundle`
2. receive a `run_id`
3. request worker-specific bundles from `/artifact-bundles`
4. convert the returned artifact payload into an LLM-safe string inside Dify
5. run specialist workers on those pruned bundles

Post-grain workers can also consume `A16` so they do not misread structurally valid skip-logic nulls as generic low-quality missingness.

For example, the grain worker can call:

```http
GET /artifact-bundles?run_id=<run_id>&mode=grain_worker
```

and receive the profile-defined artifact set for grain reasoning.

This keeps Dify focused on orchestration and LLM adjudication while leaving evidence generation, storage, and compression inside this service.

## Likely Worker Structure

The exact Dify workflow is not encoded in this repository, but the current artifact design strongly suggests a specialist pipeline along these lines:

- grain specialist
- type/transform specialist
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
