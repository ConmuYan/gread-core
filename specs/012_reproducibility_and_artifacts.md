# Reproducibility and Artifacts

## Purpose

Every experiment must be reproducible from config, seed, cache, and checkpoint metadata.

## Required Artifact Layout

```text
artifacts/
├── checkpoints/
│   ├── stage1_detector/
│   └── stage3_reasoner/
├── err_cache/
├── logs/
├── metrics/
├── tables/
├── manifests/
└── predictions/
```

## Experiment Manifest

Every run must write:

```json
{
  "experiment_id": "...",
  "git_commit": "...",
  "config_path": "...",
  "config_hash": "...",
  "dataset": "...",
  "split_hash": "...",
  "seed": 1,
  "base_detector": "...",
  "base_detector_checkpoint": "...",
  "err_cache_hash": "...",
  "contract_version": "gread_v1",
  "created_at": "...",
  "software": {
    "python": "...",
    "torch": "...",
    "torch_geometric": "..."
  }
}
```

## Checkpoint Metadata

Every checkpoint must include:

```text
model_state_dict
optimizer_state_dict if applicable
config
config_hash
seed
dataset
split_hash
git_commit
contract_version
err_cache_hash if applicable
```

## ERR Cache Reproducibility

ERR cache must be keyed by:

```text
prompt_hash
teacher_payload_hash
model_name
prompt_template_version
contract_version
```

Replay mode must not call network.

## Seed Control

All training and sampling must use the configured seed for:

```text
python random
numpy
torch
torch cuda if available
data split
trace selection
```

## README Reproduction Commands

README must include:

```bash
python -m gread_core.cli.train_detector --config ...
python -m gread_core.cli.generate_err --config ...
python -m gread_core.cli.train_reasoner --config ...
python -m gread_core.cli.evaluate --config ...
bash scripts/run_ablations.sh
python scripts/export_results.py
```

## Implementation Files

Expected files:

```text
src/gread_core/experiment/seed.py
src/gread_core/experiment/logger.py
src/gread_core/experiment/registry.py
src/gread_core/training/checkpointing.py
scripts/export_results.py
tests/unit/test_reproducibility_metadata.py
tests/integration/test_llm_cache_replay.py
```

## Required Tests

1. same seed gives same trace selection on fixture.
2. config hash is stable.
3. checkpoint metadata contains required fields.
4. ERR cache replay avoids network.
5. table exporter reads metric JSON files.
6. experiment manifest is written.

## Acceptance Criteria

This module is complete when:

* smoke experiment produces full manifest;
* checkpoint can be loaded from metadata;
* ERR cache can be replayed;
* exported tables are deterministic;
* README reproduction path works.
