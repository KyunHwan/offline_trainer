# experiment_training

Concrete experiment implementations that plug into the [`trainer/`](../trainer/) framework via the registry system. Contains registered trainers, data loaders, losses, and optimizers for imitation learning experiments, plus the YAML configs that drive them.

## Purpose

- Provide production training components (trainers, data loaders, losses, optimizers) for specific experiments
- Organize training YAML configs by experiment type and version
- Serve as a reference for implementing new experiments

## How it fits into the pipeline

Modules in this package are listed in the `plugins` section of experiment YAML configs. When the training entrypoint calls `load_plugins()`, these modules are imported, triggering their `@register` decorators and making components available to the registry system.

```yaml
plugins:
  - "experiment_training.components.dataloader.lerobot_data"
  - "experiment_training.components.trainer.imitation_learning.vfp_single_expert.vfp_single_expert_trainer"
  - "experiment_training.components.optimizer.adamw_cosine_decay"
  - "experiment_training.components.loss.sinkhorn_knopp"
```

## Structure

```
experiment_training/
├── components/              Registered component implementations
│   ├── dataloader/          Dataset factories (LeRobot, episodic HDF5)
│   ├── loss/                Loss functions (L2, Sinkhorn-Knopp OT, MoE gating)
│   ├── optimizer/           Optimizer factories (AdamW+cosine, OneCycle, schedule-free)
│   └── trainer/             Trainer implementations
│       └── imitation_learning/   IL trainers by experiment type
│
├── imitation_learning/      Training YAML configs
│   ├── vfp_single_expert/   VFP single-expert configs (exp1/, exp2/)
│   ├── cfg_vqvae_flow_matching/
│   ├── naive_flow_matching_policy/
│   ├── variational_flow_matching_policy/
│   └── mutual_information_estimator/
│
└── reinforcement_learning/  RL experiment configs (placeholder)
```

## Key directories

| Directory | Description |
|-----------|-------------|
| [`components/`](components/) | All registered implementations — the code that runs during training |
| [`imitation_learning/`](imitation_learning/) | YAML configs that specify which components to use, model paths, hyperparameters, and training schedule |

## Common workflows

### Run an existing experiment

```bash
python trainer/offline_trainer.py --train_config experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml
```

### Create a new experiment variant

1. Copy an existing experiment config (e.g., `vfp_single_expert/exp1/`) to a new `expN/` directory
2. Modify hyperparameters, model config paths, or component types
3. If using new model architectures, create corresponding configs in [`experiment_models/`](../experiment_models/)

### Add a new experiment type

1. Implement a trainer in `components/trainer/imitation_learning/<name>/`
2. Implement any required custom data loaders, losses, or optimizers in the respective `components/` subdirectories
3. Create model architecture configs in `experiment_models/<name>/`
4. Create a training YAML config in `imitation_learning/<name>/exp1/`

## Extension points

- All components follow the protocol patterns defined in [`trainer/templates/`](../trainer/templates/)
- New experiment types only need to register their components and provide a YAML config — the framework handles the rest

## Gotchas / invariants

- Plugin module paths must be importable from the project root (e.g., `experiment_training.components.dataloader.lerobot_data`)
- Training YAML configs reference model architecture configs in [`experiment_models/`](../experiment_models/) via relative paths that are resolved against the project root
