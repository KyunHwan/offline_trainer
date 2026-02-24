# trainer

Training loop implementations registered in `TRAINER_REGISTRY`. Each trainer defines the forward pass, loss computation, gradient management, and optimizer stepping for a specific experiment type.

## Purpose

- Implement the `Trainer` protocol for specific policy architectures
- Encapsulate the full training step: forward → loss → backward → clip → step
- Manage multi-model orchestration (backbones, encoders, decoders) within a single `train_step()`

## How it fits into the pipeline

The training entrypoint calls `_build_trainer()` which constructs models, optimizers, and loss, then looks up the trainer class by `train.trainer.type`, instantiates it, and verifies it satisfies the `Trainer` protocol. The training loop repeatedly calls `trainer.train_step(data)`.

**Inputs:** `models` (ModuleDict), `optimizers` (dict), `loss` (Module), `device`
**Outputs:** `train_step(data, ...) -> dict` of metric values for logging

## Structure

```
trainer/
├── imitation_learning/           IL trainer implementations
│   ├── vfp_single_expert/        VFP single-expert trainers
│   ├── cfg_vqvae_flow_matching/  CFG-VQVAE flow matching trainer
│   ├── naive_flow_matching_policy/
│   ├── variational_flow_matching_policy/
│   └── mutual_information_estimator/
└── reinforcement_learning/       RL trainer implementations (placeholder)
```

## Registered trainers

| Subdirectory | Registry key | Description |
|-------------|-------------|-------------|
| [`imitation_learning/vfp_single_expert/`](imitation_learning/vfp_single_expert/) | `vfp_single_expert_trainer` | Flow matching with 3-camera visual input, info embedding, and transformer-based action decoder |
| [`imitation_learning/vfp_single_expert/`](imitation_learning/vfp_single_expert/) | `vfp_single_expert_trainer_depth` | Same as above with depth estimation (DA3) as additional visual feature, using L1 loss |
| [`imitation_learning/cfg_vqvae_flow_matching/`](imitation_learning/cfg_vqvae_flow_matching/) | `cfg_vqvae_flow_matching_trainer_kot` | CFG-VQVAE with flow matching and K-OT loss |
| [`imitation_learning/naive_flow_matching_policy/`](imitation_learning/naive_flow_matching_policy/) | `naive_flow_matching_policy_trainer` | Basic flow matching policy trainer |
| [`imitation_learning/variational_flow_matching_policy/`](imitation_learning/variational_flow_matching_policy/) | `variational_flow_matching_policy_trainer` | Variational flow matching with MoE decoder |
| [`imitation_learning/mutual_information_estimator/`](imitation_learning/mutual_information_estimator/) | `mutual_information_estimator_trainer` | Mutual information estimation trainer |

## Common workflows

### Implement a new trainer

1. Create a new subdirectory under `imitation_learning/` (or `reinforcement_learning/`)
2. Implement a class with `__init__(models, optimizers, loss, device)` and `train_step(data, ...) -> dict`
3. Register with `@TRAINER_REGISTRY.register("my_trainer")`
4. Add an `__init__.py` that imports the trainer module
5. Reference the module path in your config's `plugins` list

## Extension points

- The `train_step()` method receives additional kwargs: `epoch`, `total_epochs`, `iterations`
- Trainers manage their own gradient clipping, learning rate logging, and loss decomposition
- Multi-model trainers can selectively freeze/unfreeze models and manage separate optimizer steps
