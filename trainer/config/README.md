# Config

YAML-driven configuration system with composition support and Pydantic validation.

## Purpose

- Load experiment configs from YAML with `defaults` composition and deep merge
- Validate configs against a typed Pydantic schema (`ExperimentConfig`)
- Produce structured, actionable error messages when validation fails

## How it fits into the pipeline

Both entrypoints ([`offline_trainer.py`](../offline_trainer.py) and [`online_trainer.py`](../online_trainer.py)) call `load_config(path)` → `validate_config(raw)` as the first step. The resulting `ExperimentConfig` drives all subsequent component construction (models, data, trainer, optimizers, losses).

**Inputs:** A YAML file path from the CLI (`--train_config`)
**Outputs:** A validated `ExperimentConfig` Pydantic model

## Key modules

| File | Description |
|------|-------------|
| [`loader.py`](loader.py) | `load_config(path)` — recursive YAML loader. Supports `defaults: [{key: relative_path}]` for config composition. Resolves relative paths against the config file's directory. Deep-merges defaults before applying the current file's overrides |
| [`schemas.py`](schemas.py) | Pydantic models: `ExperimentConfig` (root), `ModelConfig`, `DataConfig`, `TrainConfig`, `ComponentSpec` (`{type, params}`), `OptimizerParams`, `EMAConfig`, `CheckpointConfig`. All use `ConfigDict(extra="allow")` to permit additional fields |
| [`errors.py`](errors.py) | `ConfigError` and `ConfigValidationIssue` — structured error types that format Pydantic validation failures into readable messages with YAML paths |

## Common workflows

### Create a new experiment config

Start from an existing config in [`experiment_training/imitation_learning/`](../../experiment_training/imitation_learning/) and modify the component references. The required top-level keys are:

```yaml
plugins: [...]           # list of module paths to import
model:
  find_unused_parameters: false
  component_config_paths: { ... }
  component_build_args: { ... }
  component_optims: { ... }
data:
  datamodule: { type: "...", params: { ... } }
  batch_size: 60
  num_workers: 12
train:
  trainer: { type: "...", params: { ... } }
  loss: { type: "...", params: { ... } }
  epoch: 100
  save_dir: "~/checkpoints"
  save_every: 5
```

### Use defaults composition

```yaml
defaults:
  - base: ./shared_base.yaml
# overrides here take precedence over base
train:
  epoch: 200
```

## Extension points

- Add new validated fields to `ExperimentConfig` or its sub-models. Fields use Pydantic `Field` with validators for constraints (e.g., `lr > 0`, `max_epochs > 0`)
- The `ComponentSpec` pattern (`{type, params}`) is reused across trainers, losses, optimizers, schedulers, and data modules

## Gotchas / invariants

- `ComponentSpec` uses `extra="allow"` so unknown keys in `params` are preserved and forwarded to the component constructor
- The loader detects circular `defaults` references and raises `ConfigLoadError`
- Relative paths in `defaults` entries are resolved against the directory of the config file that declares them, not the working directory
