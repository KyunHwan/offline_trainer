# Registry

Registries map string keys to component classes and are the backbone of configuration-driven instantiation.

## Purpose

- Provide a generic, typed `Registry[T]` with optional base-class enforcement
- Define the four global registries that the training loop uses to discover components
- Load user-defined plugin modules at startup to populate registries

## How it fits into the pipeline

When the training entrypoint loads a config, the `plugins` list is passed to `load_plugins()`, which imports each module. Those modules use `@REGISTRY.register("key")` decorators at module scope, making their components available for lookup by the training loop.

## Key modules

| File | Description |
|------|-------------|
| [`core.py`](core.py) | `Registry[T]` — generic registry with `register(key)` decorator, `get(key)`, `has(key)`, `keys()`, and optional base-class enforcement via `expected_base` |
| [`__init__.py`](__init__.py) | Instantiates the four global registries |
| [`plugins.py`](plugins.py) | `load_plugins(modules)` — imports each module path once via `importlib.import_module()`. Tracks already-loaded modules to prevent double-registration |

## Global registries

Defined in [`__init__.py`](__init__.py):

| Registry | Expected base | Used for |
|----------|---------------|----------|
| `TRAINER_REGISTRY` | `Trainer` protocol | Training loop implementations |
| `DATASET_BUILDER_REGISTRY` | `DatasetFactory` protocol | Dataset construction factories |
| `OPTIMIZER_BUILDER_REGISTRY` | `OptimizerFactory` protocol | Optimizer construction factories |
| `LOSS_BUILDER_REGISTRY` | `LossFactory` protocol | Loss function construction factories |

## Common workflows

### Register a component

```python
from trainer.registry import TRAINER_REGISTRY

@TRAINER_REGISTRY.register("my_trainer")
class MyTrainer:
    def __init__(self, models, optimizers, loss, device): ...
    def train_step(self, data, **kwargs) -> dict: ...
```

### Load plugins from config

```yaml
plugins:
  - "experiment_training.components.trainer.imitation_learning.vfp_single_expert.vfp_single_expert_trainer"
  - "experiment_training.components.dataloader.lerobot_data"
```

These modules are imported by [`plugins.py`](plugins.py), triggering their `@register` decorators.

## Extension points

- Add new global registries in `__init__.py` for new component types
- Any importable Python module can serve as a plugin — just use the registry decorator at module scope

## Gotchas / invariants

- Registering the same key twice raises `KeyError` — each key must be unique within a registry. See [`core.py:28-29`](core.py)
- Base-class enforcement checks `issubclass` for types and `isinstance` for instances. See [`core.py:44-57`](core.py)
- Plugin modules are imported exactly once per process. The `_LOADED_MODULES` set in [`plugins.py`](plugins.py) prevents re-imports
