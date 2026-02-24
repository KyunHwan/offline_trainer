# components

Registered component implementations for the training framework. Each subdirectory provides pluggable implementations that are discovered via the registry/plugin system.

## Purpose

- Implement concrete data loaders, losses, optimizers, and trainers
- Register each implementation under a string key for YAML-driven instantiation
- Provide production-ready components for imitation learning experiments

## How it fits into the pipeline

Modules here are imported by the plugin loader (`trainer.registry.plugins.load_plugins`) when listed in a config's `plugins` section. Each module's `@REGISTRY.register("key")` decorators run at import time, populating the global registries.

## Structure

| Subdirectory | Description | Registered keys |
|-------------|-------------|-----------------|
| [`dataloader/`](dataloader/) | Dataset factory implementations | `lerobot_dataset_factory`, `episodic_dataset_factory` |
| [`loss/`](loss/) | Loss function factories | `l2_loss`, `sinkhorn_knopp` |
| [`optimizer/`](optimizer/) | Optimizer factory implementations | `adamw_warmup_cosine_decay`, `adamw_cosine_schedule`, `schedule_free_radam` |
| [`trainer/`](trainer/) | Training loop implementations | See [`trainer/`](trainer/) README |

## Common workflows

### Use a component in a config

Reference the registered key in the appropriate config section:

```yaml
plugins:
  - "experiment_training.components.loss.sinkhorn_knopp"

train:
  loss:
    type: "sinkhorn_knopp"
    params:
      p: 1
      lam_state: 0.2
      blur: 0.004
      debias: true
      backend: "tensorized"
      scaling: 0.95
```

### Add a new component

1. Create a Python file in the appropriate subdirectory
2. Import the relevant registry and decorate your class with `@REGISTRY.register("key")`
3. Implement the required protocol from [`trainer/templates/`](../../trainer/templates/)
4. Add the module's import path to your config's `plugins` list
