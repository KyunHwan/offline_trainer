# imitation_learning

Imitation learning trainer implementations. Each subdirectory contains a trainer registered in `TRAINER_REGISTRY` that implements a specific policy learning approach.

## Purpose

- Implement the training step for various imitation learning policy architectures
- Handle multi-camera visual processing, flow matching, and variational methods
- Manage per-component forward passes across backbones, encoders, and decoders

## Trainer implementations

| Directory | Registry key | Policy type |
|-----------|-------------|-------------|
| [`vfp_single_expert/`](vfp_single_expert/) | `vfp_single_expert_trainer` | Single-expert flow matching with 3 cameras (head, left, right). Uses Beta(1.0, 1.5) time sampling, random camera dropout (15%), transformer info embedder + action decoder. Velocity loss (L2 MSE) |
| [`vfp_single_expert/`](vfp_single_expert/) | `vfp_single_expert_trainer_depth` | Same as above with Depth Anything v3 features as a 4th visual input. Uses L1 loss for velocity prediction |
| [`cfg_vqvae_flow_matching/`](cfg_vqvae_flow_matching/) | `cfg_vqvae_flow_matching_trainer_kot` | Classifier-free guidance VQVAE with flow matching and K-OT regularization |
| [`naive_flow_matching_policy/`](naive_flow_matching_policy/) | `naive_flow_matching_policy_trainer` | Basic flow matching policy |
| [`variational_flow_matching_policy/`](variational_flow_matching_policy/) | `variational_flow_matching_policy_trainer` | Variational flow matching with MoE action decoder, VQVAE codebook, posterior/prior |
| [`mutual_information_estimator/`](mutual_information_estimator/) | `mutual_information_estimator_trainer` | Mutual information estimation between state/action representations |

## Common training step pattern

All trainers follow a shared structure:

1. **Extract features** — pass camera images through backbone models
2. **Encode conditioning** — combine visual + proprioceptive features via info embedder
3. **Flow matching** — sample noise and time, interpolate between noise and target action, compute target velocity
4. **Decode** — predict velocity via action decoder
5. **Loss** — compute velocity prediction loss
6. **Update** — zero grad → backward → clip gradients → optimizer step

## How it fits into the pipeline

Trainers receive `models` (nn.ModuleDict of named model components), `optimizers` (dict of per-component optimizers), `loss` (nn.Module or None), and `device`. The training loop calls `train_step(data, epoch, total_epochs, iterations)` and logs the returned metric dict to WandB.
