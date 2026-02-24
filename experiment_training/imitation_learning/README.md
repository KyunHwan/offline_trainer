# imitation_learning

Training YAML configs for imitation learning experiments. Each subdirectory corresponds to a policy architecture, with numbered experiment variants (`exp1/`, `exp2/`, etc.).

## Purpose

- Provide complete, ready-to-run training configurations
- Organize experiments by policy type and version
- Define all hyperparameters: plugins, model paths, data sources, optimizer schedules, loss functions, and training duration

## How it fits into the pipeline

These YAML files are passed directly to the training entrypoint:

```bash
python trainer/offline_trainer.py --train_config experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml
```

The config references:
- **Plugins** in [`experiment_training/components/`](../components/) (loaded via `plugins` list)
- **Model architectures** in [`experiment_models/`](../../experiment_models/) (via `model.component_config_paths`)

## Structure

| Directory | Experiments | Description |
|-----------|------------|-------------|
| [`vfp_single_expert/`](vfp_single_expert/) | `exp1/`, `exp2/` | VFP single-expert flow matching. exp1: 3-camera baseline. exp2: adds depth estimation |
| [`cfg_vqvae_flow_matching/`](cfg_vqvae_flow_matching/) | `exp1/` | Classifier-free guidance VQVAE with flow matching |
| [`naive_flow_matching_policy/`](naive_flow_matching_policy/) | `exp1/`, `exp2/` | Basic flow matching policy |
| [`variational_flow_matching_policy/`](variational_flow_matching_policy/) | `exp1/`, `exp2/` | Variational flow matching with MoE decoder |
| [`mutual_information_estimator/`](mutual_information_estimator/) | `exp1/` | Mutual information estimation |

## Common workflows

### Run an experiment

```bash
# Single GPU
python trainer/offline_trainer.py --train_config experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml

# Multi-GPU
torchrun --nproc_per_node=4 trainer/offline_trainer.py --train_config experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml
```

### Create a new experiment variant

1. Copy an existing `expN/` directory to `expN+1/`
2. Edit hyperparameters (learning rate, batch size, epochs, etc.)
3. Point `model.component_config_paths` to new architecture configs in [`experiment_models/`](../../experiment_models/) if architecture changes are needed

## Gotchas / invariants

- All paths in the config (model configs, save directories) are resolved relative to the project root
- The `plugins` list must include all component modules needed by the config — missing plugins cause `KeyError` at registry lookup time
- `save_dir` should be an absolute path or use `~` expansion. The training loop creates the directory and saves checkpoints as `epoch_{N}/{component}.pt`
