# trainer

Core package for the config-driven distributed training framework. Provides the training entrypoints, configuration system, component registries, model factory, and shared utilities.

## Purpose

- Load and validate YAML experiment configs via Pydantic schemas
- Discover and register custom components through a plugin system
- Build models from [policy_constructor](../policy_constructor/) configs via a thin adapter
- Run distributed training loops with PyTorch DDP or Ray Train
- Checkpoint models/optimizers and log metrics to Weights & Biases

## How it fits into the pipeline

This package is the framework layer. Experiment-specific implementations (trainers, datasets, losses, optimizers) live in [`experiment_training/`](../experiment_training/) and are loaded as plugins at runtime. Model architecture definitions live in [`experiment_models/`](../experiment_models/) as policy_constructor YAML configs.

```
YAML config → trainer.config.loader → trainer.config.schemas (validation)
           → trainer.registry.plugins (load experiment modules)
           → trainer.modeling.factories (build models from policy_constructor)
           → offline_trainer.train() or online_trainer.train_func()
```

## Key modules

| Module | Description |
|--------|-------------|
| [`offline_trainer.py`](offline_trainer.py) | DDP-based offline training entrypoint. CLI flag: `--train_config`. Defines `train(config_path)` and the epoch-based training loop |
| [`online_trainer.py`](online_trainer.py) | Ray Train online/offline hybrid entrypoint. Defines `train_func(config_path)` for use as a Ray worker function |
| [`config/`](config/) | YAML loader with `defaults` composition, Pydantic schema validation, and structured error reporting |
| [`modeling/`](modeling/) | `PolicyConstructorModelFactory` — adapter that calls `model_constructor.build_model()` per component |
| [`registry/`](registry/) | Generic typed registry (`Registry[T]`) and the four global registries (trainer, dataset, optimizer, loss). Plugin loader for dynamic imports |
| [`templates/`](templates/) | Python `Protocol` definitions for `Trainer`, `DatasetFactory`, `LossFactory`, `OptimizerFactory` — the contracts that all registered components must satisfy |
| [`utils/`](utils/) | `tree_map` for nested structures, `move_to_device`/`cast_dtype` for tensors, `set_global_seed`/`seed_worker` for reproducibility, `select` for dict/tuple access, `instantiate` for filtered kwargs construction |

## Common workflows

### Run an existing experiment

```bash
# Single GPU
python trainer/offline_trainer.py --train_config experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml

# Multi-GPU DDP
torchrun --nproc_per_node=4 trainer/offline_trainer.py --train_config <config.yaml>
```

### Add a new component

1. Create a Python module that implements one of the [protocols](templates/) and registers it via the appropriate registry decorator
2. Add the module's import path to the `plugins` list in your config YAML
3. Reference the registered key in the appropriate config section (`train.trainer.type`, `data.datamodule.type`, etc.)

See [`examples/extensions/`](../examples/extensions/) for concrete examples.

## Extension points

- **Trainer**: implement the `Trainer` protocol, register in `TRAINER_REGISTRY`. Receives `models` (ModuleDict), `optimizers` (dict), `loss` (Module), and `device`
- **Dataset**: implement the `DatasetFactory` protocol, register in `DATASET_BUILDER_REGISTRY`. Return `{"dataset": Dataset, "norm_stats": dict}`
- **Optimizer**: implement the `OptimizerFactory` protocol, register in `OPTIMIZER_BUILDER_REGISTRY`. Build and return a `torch.optim.Optimizer`
- **Loss**: implement the `LossFactory` protocol, register in `LOSS_BUILDER_REGISTRY`. Build and return an `nn.Module`

## Gotchas / invariants

- **DDP wrapping**: Models are wrapped with `DistributedDataParallel` only when `world_size > 1` and the model is not frozen. Frozen models are moved to device but not wrapped. Checkpoints always save unwrapped `.module` state dicts. Defined in [`offline_trainer.py:140-197`](offline_trainer.py)
- **Rank-0-only operations**: WandB logging, checkpoint saving, and stats persistence are gated on `rank == 0`. All ranks must hit barriers together
- **Seed management**: Base seed is shared across ranks for synchronized weight init. After model construction, seed is offset by rank for independent dropout/augmentation randomness. See [`offline_trainer.py:430-441`](offline_trainer.py)
- **SyncBatchNorm**: Automatically applied when BatchNorm layers are detected in DDP mode, before moving to device. See [`offline_trainer.py:182-183`](offline_trainer.py)
- **Normalization in the loop**: Data normalization happens in the training loop, not inside policies. Stats come from the dataset factory. See [`offline_trainer.py:457-461`](offline_trainer.py)
- **Mixed precision**: Forward passes run under `torch.autocast(dtype=torch.bfloat16)`. Data is cast to `float32` before entering the autocast region
