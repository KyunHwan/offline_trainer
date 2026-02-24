# Modeling

Model construction adapter that bridges the trainer framework with [policy_constructor](../../policy_constructor/).

## Purpose

- Provide a `ModelFactory` protocol for building PyTorch modules from config
- Implement `PolicyConstructorModelFactory` which calls `model_constructor.build_model()` for each component

## How it fits into the pipeline

Called by `_build_models()` in [`offline_trainer.py`](../offline_trainer.py) and [`online_trainer.py`](../online_trainer.py). Receives the `model.component_config_paths` dict from the experiment config, builds each named model component, and returns them for DDP wrapping and optimizer binding.

**Inputs:** Dict of `{component_name: config_path}` from `model.component_config_paths`
**Outputs:** Dict of `{component_name: nn.Module}` (GraphModel instances from policy_constructor)

## Key modules

| File | Description |
|------|-------------|
| [`factories.py`](factories.py) | `ModelFactory` protocol and `PolicyConstructorModelFactory`. Supports three build modes: (1) single model via `config_path` key, (2) inline config via `config` key, (3) multiple named components via individual keys (the standard experiment pattern) |

## Common workflows

### Build models from an experiment config

The standard pattern uses `component_config_paths` to define multiple named model components:

```yaml
model:
  component_config_paths:
    head_backbone: "experiment_models/vfp_single_expert/exp1/head_backbone.yaml"
    info_embedder: "experiment_models/vfp_single_expert/exp1/info_embedder.yaml"
    action_decoder: "experiment_models/vfp_single_expert/exp1/action_decoder.yaml"
```

Each path points to a [policy_constructor YAML config](../../policy_constructor/model_constructor/config/) that declaratively defines the model architecture.

## Extension points

- Implement a different `ModelFactory` to support alternative model construction backends
- The factory is instantiated in the trainer entrypoints — swap it by modifying `_build_models()`

## Gotchas / invariants

- `PolicyConstructorModelFactory` adds `policy_constructor/` to `sys.path` at import time if `model_constructor` is not already importable. See [`factories.py:12-17`](factories.py)
- Relative config paths in `component_config_paths` are resolved against the project root (parent of `trainer/`). See [`offline_trainer.py:146-153`](../offline_trainer.py)
