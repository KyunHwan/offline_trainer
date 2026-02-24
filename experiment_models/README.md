# experiment_models

Model architecture configurations for [policy_constructor](../policy_constructor/). Each YAML file defines a single model component (backbone, encoder, decoder, etc.) that is built into a `GraphModel` (PyTorch `nn.Module`) at training time.

## Purpose

- Define model architectures declaratively in YAML using the policy_constructor config schema
- Organize model configs by experiment type and version
- Provide reusable component configs that can be mixed and matched across experiments

## How it fits into the pipeline

Training YAML configs reference these files via `model.component_config_paths`:

```yaml
model:
  component_config_paths:
    head_backbone: "experiment_models/vfp_single_expert/exp1/head_backbone.yaml"
    info_embedder: "experiment_models/vfp_single_expert/exp1/info_embedder.yaml"
    action_decoder: "experiment_models/vfp_single_expert/exp1/action_decoder.yaml"
```

At training time, [`PolicyConstructorModelFactory`](../trainer/modeling/factories.py) calls `model_constructor.build_model(config_path)` for each entry, producing named `nn.Module` instances that are stored in an `nn.ModuleDict`.

**Inputs:** Relative paths from training YAML configs (resolved against project root)
**Outputs:** `nn.Module` instances via `model_constructor.build_model()`

## Structure

```
experiment_models/
├── vfp_single_expert/
│   ├── exp1/                   Base VFP single-expert (3 cameras)
│   │   ├── head_backbone.yaml      RadioV3 backbone (1024 + 3072 channels)
│   │   ├── left_backbone.yaml      RadioV3 backbone (shared architecture)
│   │   ├── right_backbone.yaml     RadioV3 backbone (shared architecture)
│   │   ├── info_embedder.yaml      4-layer transformer encoder (d=512, 8 heads)
│   │   ├── action_decoder.yaml     7-layer causal transformer decoder (d=512, 8 heads)
│   │   ├── vae_posterior.yaml      5-layer transformer (d=768, 16 heads, CLS tokens)
│   │   ├── vae_prior.yaml          6-layer transformer (d=768, 16 heads)
│   │   └── da3.yaml                Depth Anything v3 model
│   └── exp2/                   VFP + depth estimation (4 visual inputs)
│       ├── head_backbone.yaml
│       ├── left_backbone.yaml
│       ├── right_backbone.yaml
│       ├── info_embedder.yaml      4 visual inputs (includes depth features)
│       ├── action_decoder.yaml     8-layer transformer decoder
│       ├── multimodal_bridge.yaml  Multimodal bridge with 4 visual inputs
│       ├── vae_posterior.yaml
│       ├── vae_prior.yaml
│       └── da3.yaml
│
├── cfg_vqvae_flow_matching/
│   └── exp1/                   CFG-VQVAE + flow matching
│       ├── backbone.yaml
│       ├── info_encoder.yaml
│       ├── action_decoder.yaml
│       ├── vqvae_posterior.yaml
│       ├── vqvae_prior.yaml
│       ├── vqvae_codebook.yaml
│       ├── proprio_projector.yaml
│       └── da3.yaml
│
├── naive_flow_matching_policy/
│   └── exp1/                   Basic flow matching policy
│       ├── backbone.yaml
│       ├── info_embedder.yaml
│       ├── action_decoder.yaml
│       ├── proprio_projector.yaml
│       ├── left_hand_extractor.yaml
│       ├── right_hand_extractor.yaml
│       └── da3.yaml
│
├── variational_flow_matching_policy/
│   └── exp1/                   Variational flow matching with MoE
│       ├── backbone.yaml
│       ├── info_embedder.yaml
│       ├── moe_action_decoder.yaml
│       ├── proprio_projector.yaml
│       ├── vqvae_posterior.yaml
│       ├── vqvae_prior.yaml
│       ├── vqvae_codebook.yaml
│       ├── gate.yaml
│       ├── left_hand_extractor.yaml
│       ├── right_hand_extractor.yaml
│       └── da3.yaml
│
└── mutual_information_estimator/
    └── exp1/                   MI estimation
        ├── action_encoder.yaml
        ├── action_decoder.yaml
        ├── state_resnet34_encoder.yaml
        └── state_resnet34_decoder.yaml
```

## Common workflows

### Create a new experiment variant

1. Copy an existing experiment directory (e.g., `vfp_single_expert/exp1/`) to a new `expN/`
2. Modify architecture parameters (layer count, hidden dimensions, attention heads, etc.)
3. Reference the new paths in a training YAML config in [`experiment_training/imitation_learning/`](../experiment_training/imitation_learning/)

### Add a new experiment type

1. Create a new directory under `experiment_models/` matching your experiment name
2. Define a YAML file for each model component using the [policy_constructor config schema](../policy_constructor/model_constructor/config/)
3. Create a corresponding trainer in [`experiment_training/components/trainer/`](../experiment_training/components/trainer/)
4. Create a training config in [`experiment_training/imitation_learning/`](../experiment_training/imitation_learning/)

## Extension points

- Model architecture complexity is controlled entirely by YAML — no code changes needed for new architectures
- Components can be shared across experiments (e.g., the same `da3.yaml` depth model appears in multiple experiments)
- Per-component `init`/`freeze` flags in the training config control whether each model is initialized from scratch, loaded from a checkpoint, or frozen

## Gotchas / invariants

- Config paths are relative to the project root and resolved at training time by [`_build_models()`](../trainer/offline_trainer.py). Absolute paths are also supported
- Each YAML file defines a single model component — the trainer is responsible for orchestrating the interaction between components
- Model configs use the policy_constructor registry to reference block types (e.g., `vfp_single_action_decoder`, `RadioV3`). These blocks must be registered before model construction
