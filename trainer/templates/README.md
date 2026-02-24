# Templates

Protocol definitions that establish the contracts for all pluggable training components. Every registered component must satisfy one of these interfaces.

## Purpose

- Define the `Trainer`, `DatasetFactory`, `LossFactory`, and `OptimizerFactory` protocols
- Serve as reference skeletons for implementing custom components
- Enable runtime type checking via `@runtime_checkable`

## How it fits into the pipeline

The training entrypoints ([`offline_trainer.py`](../offline_trainer.py), [`online_trainer.py`](../online_trainer.py)) use these protocols as type constraints. After constructing a component from the registry, the entrypoint checks `isinstance(obj, Protocol)` to verify the contract is satisfied.

## Key modules

| File | Protocol | Signature |
|------|----------|-----------|
| [`trainer.py`](trainer.py) | `Trainer` | `__init__(models: ModuleDict, optimizers: dict, loss: Module)` and `train_step(data: dict) -> dict` |
| [`dataset.py`](dataset.py) | `DatasetFactory` | `build(**kwargs) -> dict` — returns `{"dataset": Dataset, "norm_stats": dict}` |
| [`loss.py`](loss.py) | `LossFactory` | `build() -> nn.Module` |
| [`optim.py`](optim.py) | `OptimizerFactory` | `build(params: Iterable[Parameter]) -> torch.optim.Optimizer` |

## Common workflows

### Implement a new trainer

```python
from trainer.registry import TRAINER_REGISTRY

@TRAINER_REGISTRY.register("my_trainer")
class MyTrainer:
    def __init__(self, models, optimizers, loss, device):
        self.models = models
        self.optimizers = optimizers
        self.loss = loss
        self.device = device

    def train_step(self, data, **kwargs) -> dict:
        # Forward pass, backward, optimizer step
        # Return dict of metric names → scalar values for logging
        return {"loss": loss_value.item()}
```

### Implement a new dataset factory

```python
from trainer.registry import DATASET_BUILDER_REGISTRY

@DATASET_BUILDER_REGISTRY.register("my_dataset")
class MyDatasetFactory:
    def build(self, opt_params=None, params=None) -> dict:
        dataset = ...  # build torch.utils.data.Dataset
        stats = {"action": {"mean": [...], "std": [...]}}
        return {"dataset": dataset, "norm_stats": stats}
```

## Extension points

- All protocols use `@runtime_checkable`, so you can verify conformance at runtime with `isinstance()`
- The `Trainer` protocol's `train_step` can accept additional keyword arguments (e.g., `epoch`, `total_epochs`, `iterations`) — the training loop passes these through

## Gotchas / invariants

- The `Trainer.train_step()` return dict is logged directly to WandB. Keys become metric names; values must be scalars or tensors that can be detached to scalars
- `DatasetFactory.build()` receives `opt_params` with keys `local_rank`, `dist_enabled`, `save_dir` from the training loop. See [`offline_trainer.py:232-244`](../offline_trainer.py)
- `OptimizerFactory.build(params)` receives the model's parameters iterator. The factory must return a fully configured optimizer (scheduler integration is the factory's responsibility if needed)
