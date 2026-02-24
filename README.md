# Trainer

A config-driven distributed training framework for imitation-learning policies.
Models are composed from YAML via [policy_constructor](policy_constructor/), training components are selected through a registry/plugin system, and the training loop scales from a single GPU (PyTorch DDP) to a multi-node Ray cluster.

---

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                       YAML Experiment Config                        │
 │  (plugins, model, data, train)                                      │
 └───────────────┬──────────────────────────────────────────────────────┘
                 │  load_config()  +  validate_config()
                 ▼
 ┌───────────────────────────┐    ┌──────────────────────────────────┐
 │  trainer/config/loader.py │───▶│ trainer/config/schemas.py        │
 │  YAML with defaults       │    │ Pydantic: ExperimentConfig       │
 │  composition & deep merge │    │  ├─ ModelConfig                  │
 └───────────────────────────┘    │  ├─ DataConfig                   │
                                  │  └─ TrainConfig                  │
                                  └──────────────┬───────────────────┘
                                                 │
                 ┌───────────────────────────────┼───────────────────┐
                 │                               │                   │
                 ▼                               ▼                   ▼
 ┌───────────────────────┐   ┌──────────────────────┐  ┌────────────────────┐
 │  Plugin Loader         │   │  Model Factory        │  │  Registries         │
 │  registry/plugins.py   │   │  modeling/factories.py │  │  TRAINER_REGISTRY   │
 │  importlib.import →    │   │  PolicyConstructor     │  │  DATASET_BUILDER_   │
 │  register components   │   │  ModelFactory          │  │  OPTIMIZER_BUILDER_ │
 └───────────────────────┘   │  → build_model() per   │  │  LOSS_BUILDER_      │
                              │    component config    │  └────────────────────┘
                              └──────────┬─────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  nn.ModuleDict        │
                              │  {name: GraphModel}   │
                              │  per-model freeze /   │
                              │  init / DDP wrap      │
                              └──────────┬────────────┘
                                         │
           ┌─────────────────────────────┼──────────────────────────┐
           │                             │                          │
           ▼                             ▼                          ▼
 ┌──────────────────┐     ┌────────────────────────┐   ┌─────────────────────┐
 │ DatasetFactory    │     │  Trainer (protocol)     │   │ OptimizerFactory    │
 │ → Dataset +       │     │  .train_step(data)      │   │ → Optimizer per     │
 │   norm_stats      │     │  forward / backward /   │   │   model component   │
 └────────┬─────────┘     │  clip / step            │   └─────────────────────┘
          │                └────────────┬───────────┘
          ▼                             │
 ┌──────────────────────┐               │
 │ DataLoader            │               │
 │ DistributedSampler    │               │
 │ (DDP) or              │               │
 │ ray.train.torch       │               │
 │ .prepare_data_loader  │               │
 └────────┬──────────────┘               │
          │                              │
          ▼                              ▼
 ┌────────────────────────────────────────────────────────┐
 │                    Training Loop                        │
 │  normalize(data, stats) → cast_dtype → move_to_device  │
 │  autocast(bfloat16) → trainer.train_step(data)         │
 │  rank-0: wandb.log + _save_checkpoints                 │
 │  barrier → next epoch                                   │
 └────────────────────────────────────────────────────────┘
          │
          │  (online_trainer.py only)
          ▼
 ┌────────────────────────────────────────────────────────┐
 │  Ray Actors                                             │
 │  replay_buffer.sample.remote() → online data            │
 │  policy_state_manager.update_weights.remote()           │
 │  → ray.put(cpu_state_dicts) for inference workers       │
 └────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites

- Python 3.10+ (developed with 3.12)
- CUDA-capable GPU(s) for distributed training

### Environment setup

```bash
# 1. Create virtual environment via uv
bash uv_setup.sh        # installs uv, creates .venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Install dependencies (PyTorch, datasets, vision models, etc.)
bash env_setup.sh
```

Dependencies installed by [`env_setup.sh`](env_setup.sh):
- PyTorch 2.9 (CUDA 13.0), torchvision
- flow_matching, schedulefree, geomloss, einops, timm
- wandb (experiment tracking)
- lerobot, datasets, accelerate (data loading)
- torchcodec, av, ffmpeg (media decoding)
- Depth Anything v3 (editable install from submodule)

### Running training

**Single-GPU (offline):**

```bash
python trainer/offline_trainer.py --train_config <path/to/config.yaml>
```

**Multi-GPU with DDP:**

```bash
torchrun --nproc_per_node=<NUM_GPUS> trainer/offline_trainer.py --train_config <path/to/config.yaml>
```

**Ray-based online/offline hybrid:**

[`trainer/online_trainer.py`](trainer/online_trainer.py) exposes `train_func(config_path)`, which is called as a Ray Train worker function. It expects named Ray actors `"replay_buffer"` and `"policy_state_manager"` to be running in the cluster.

---

## Entrypoints

| Entrypoint | Description | Distributed backend |
|---|---|---|
| [`trainer/offline_trainer.py`](trainer/offline_trainer.py) | Epoch-based offline training. CLI: `--train_config <yaml>` | PyTorch DDP via `torchrun` |
| [`trainer/online_trainer.py`](trainer/online_trainer.py) | Continuous online+offline hybrid training. Called as `train_func(config_path)` inside a Ray Train job | Ray Train |

Both entrypoints follow the same pipeline: load config → validate → load plugins → build models/optimizers/data/trainer → run training loop.

---

## Configuration

Configuration is YAML-driven and validated with Pydantic.

### Config loading

[`trainer/config/loader.py`](trainer/config/loader.py) loads YAML files with support for **`defaults` composition** and **deep merge**. Relative paths in `defaults` entries are resolved against the config file's directory.

```yaml
defaults:
  - base: ./base_config.yaml   # merged first
# keys here override the base
```

### Schema

[`trainer/config/schemas.py`](trainer/config/schemas.py) defines the full schema via `ExperimentConfig`:

```yaml
plugins:                        # list of importable modules that register components
  - "experiment_training.components.dataloader.lerobot_data"
  - "experiment_training.components.trainer.imitation_learning.vfp_single_expert.vfp_single_expert_trainer"

seed: 123                       # base RNG seed (same on all ranks for weight init)

model:
  find_unused_parameters: false # DDP flag for conditional model paths (e.g. MoE)
  component_config_paths:       # maps component name → policy_constructor YAML
    head_backbone: "experiment_models/vfp_single_expert/exp1/head_backbone.yaml"
    info_embedder: "experiment_models/vfp_single_expert/exp1/info_embedder.yaml"
    action_decoder: "experiment_models/vfp_single_expert/exp1/action_decoder.yaml"
  component_build_args:         # per-component flags
    head_backbone: { init: false, freeze: false }
    info_embedder: { init: true, freeze: false }
  component_optims:             # per-component optimizer config
    head_backbone:
      type: "adamw_warmup_cosine_decay"
      params: { peak_lr: 1.0e-4, total_steps: 200000, warmup_steps: 2000, ... }

data:
  datamodule:
    type: "lerobot_dataset_factory"       # registry key
    params: { repo_id: "...", action_horizon: 40, ... }
  batch_size: 60
  num_workers: 12
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

train:
  trainer: { type: "vfp_single_expert_trainer" }
  loss: { type: "sinkhorn_knopp", params: { p: 1, lam_state: 0.2, blur: 0.004, ... } }
  project_name: "my_experiment"
  save_dir: "~/checkpoints"
  save_every: 3                  # save every N epochs
  epoch: 1000
  seed: 0
```

See [`experiment_training/imitation_learning/`](experiment_training/imitation_learning/) for complete working configs.

---

## Project layout

```
├── trainer/                     Core training framework
│   ├── offline_trainer.py       DDP offline training entrypoint
│   ├── online_trainer.py        Ray Train online/offline hybrid entrypoint
│   ├── config/                  YAML loader + Pydantic schemas
│   ├── modeling/                Model factory (policy_constructor adapter)
│   ├── registry/                Registry system + plugin loader
│   ├── templates/               Protocol definitions (Trainer, DatasetFactory, etc.)
│   └── utils/                   Device, tree, seed, selection, import helpers
│
├── experiment_training/         Experiment implementations (plugins)
│   ├── components/              Registered trainers, data loaders, losses, optimizers
│   │   ├── dataloader/          LeRobot + episodic dataset factories
│   │   ├── loss/                L2, Sinkhorn-Knopp OT, MoE gating losses
│   │   ├── optimizer/           AdamW+cosine, AdamW+OneCycle, schedule-free RAdam
│   │   └── trainer/             Imitation learning trainer implementations
│   └── imitation_learning/      Training YAML configs per experiment
│
├── experiment_models/           Model architecture configs (policy_constructor YAML)
│   ├── vfp_single_expert/       VFP single-expert model components
│   ├── cfg_vqvae_flow_matching/ CFG-VQVAE flow matching components
│   ├── naive_flow_matching_policy/
│   ├── variational_flow_matching_policy/
│   └── mutual_information_estimator/
│
├── policy_constructor/          Model construction library (git submodule)
│
├── examples/                    Example configs + extension modules
│   ├── configs/                 Minimal/custom YAML examples
│   └── extensions/              Sample custom trainers and data modules
│
├── tests/                       Pytest suite
├── env_setup.sh                 Dependency installation script
├── uv_setup.sh                  Virtual environment setup
└── pytest.ini                   Pytest configuration
```

---

## Extending the system

All components are registered through the global registries defined in [`trainer/registry/__init__.py`](trainer/registry/__init__.py). To add a new component:

1. Implement the appropriate protocol from [`trainer/templates/`](trainer/templates/)
2. Register it with the corresponding registry decorator
3. Add the module path to the `plugins` list in your config YAML

### Adding a new dataset

Implement the [`DatasetFactory`](trainer/templates/dataset.py) protocol:

```python
from trainer.registry import DATASET_BUILDER_REGISTRY

@DATASET_BUILDER_REGISTRY.register("my_dataset")
class MyDatasetFactory:
    def build(self, opt_params, params) -> dict:
        # Return {"dataset": <torch Dataset>, "norm_stats": <stats dict>}
        ...
```

The factory's `build()` method receives `opt_params` (dict with `local_rank`, `dist_enabled`, `save_dir`) and `params` (from the YAML `data.datamodule.params`). Return a dict with keys `"dataset"` and optionally `"norm_stats"`. Normalization stats, when provided, are serialized to `dataset_stats.pkl` by the training loop (rank 0 only).

### Adding a new trainer

Implement the [`Trainer`](trainer/templates/trainer.py) protocol:

```python
from trainer.registry import TRAINER_REGISTRY

@TRAINER_REGISTRY.register("my_trainer")
class MyTrainer:
    def __init__(self, models, optimizers, loss, device):
        ...
    def train_step(self, data, **kwargs) -> dict:
        # Return dict of loss/metric values for logging
        ...
```

### Adding a new loss

Implement the [`LossFactory`](trainer/templates/loss.py) protocol:

```python
from trainer.registry import LOSS_BUILDER_REGISTRY

@LOSS_BUILDER_REGISTRY.register("my_loss")
class MyLossFactory:
    def build(self) -> nn.Module:
        ...
```

### Adding a new optimizer

Implement the [`OptimizerFactory`](trainer/templates/optim.py) protocol:

```python
from trainer.registry import OPTIMIZER_BUILDER_REGISTRY

@OPTIMIZER_BUILDER_REGISTRY.register("my_optim")
class MyOptimFactory:
    def build(self, params) -> torch.optim.Optimizer:
        ...
```

See [`examples/extensions/`](examples/extensions/) for runnable examples of custom components.

---

## Training data flow and invariants

### Normalization

Normalization is performed **in the training loop** (not inside the policy). Dataset stats are obtained from the dataset factory's returned `norm_stats` dict and applied per-batch before the forward pass:

```python
# trainer/offline_trainer.py:457-461
data['action'] = (data['action'] - stats['action']['mean']) / (stats['action']['std'] + 1e-8)
data['observation.state'] = (data['observation.state'] - stats['observation.state']['mean']) / ...
```

Stats are persisted to `{save_dir}/dataset_stats.pkl` on rank 0 (defined at [`trainer/offline_trainer.py:246-266`](trainer/offline_trainer.py)).

### Dtype and device pipeline

1. Normalization on CPU tensors
2. `cast_dtype(data, torch.float32)` — casts all floating tensors ([`trainer/utils/device.py`](trainer/utils/device.py))
3. `move_to_device(data, device)` — moves tensors to GPU ([`trainer/utils/device.py`](trainer/utils/device.py))
4. `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` — mixed-precision forward pass

### Model construction

Models are built by [`PolicyConstructorModelFactory`](trainer/modeling/factories.py) which calls `model_constructor.build_model()` for each component path in `model.component_config_paths`. The result is an `nn.ModuleDict` of named models, each independently:
- **Initializable** (`component_build_args[name].init`) — applies Xavier/Kaiming weight init
- **Freezable** (`component_build_args[name].freeze`) — disables gradients and sets eval mode
- **Optimizable** (`component_optims[name]`) — gets its own optimizer instance

### DDP behavior

- Models are wrapped with `DistributedDataParallel` only when `world_size > 1` and not frozen
- `SyncBatchNorm` is applied automatically when BatchNorm layers are detected
- `find_unused_parameters` is configurable per experiment (for MoE architectures)
- Checkpoints unwrap DDP `.module` before saving

### Seed management

- All ranks share the same base seed for synchronized weight initialization
- After model construction, seed is offset by rank (`base_seed + rank`) for independent runtime randomness (dropout, data augmentation)
- Dataloader workers are seeded via `seed_worker()` ([`trainer/utils/seed.py`](trainer/utils/seed.py))

### Checkpoint format

```
{save_dir}/
  epoch_{N}/
    {component_name}.pt          # model state dict
    {component_name}_opt.pt      # optimizer state dict
  dataset_stats.pkl              # normalization statistics
```

Defined in [`trainer/offline_trainer.py:328-367`](trainer/offline_trainer.py).

---

## policy_constructor

[`policy_constructor/`](policy_constructor/) is a git submodule that provides a YAML-driven model construction system. It builds `GraphModel` instances (PyTorch `nn.Module`) from declarative config files.

The trainer interfaces with it through [`PolicyConstructorModelFactory`](trainer/modeling/factories.py), which calls `model_constructor.build_model(config_path)` for each component defined in the experiment config.

Model architecture configs live in [`experiment_models/`](experiment_models/), organized by experiment type. Each YAML file defines a single model component (backbone, encoder, decoder, etc.) using the `policy_constructor` config schema.

For details on config authoring, see the [policy_constructor README](policy_constructor/README.md) and its [config documentation](policy_constructor/model_constructor/config/).

---

## Experiment tracking

Training metrics are logged to [Weights & Biases](https://wandb.ai/) (rank 0 only). The project name is derived from `data.datamodule.params.task_name` and the run name from `train.project_name`. Loss and metric dicts returned by `trainer.train_step()` are logged every iteration.

---

## Tests

```bash
pytest
```

Tests cover config validation, registry plugins, checkpoint resume, tree utilities, and a minimal training smoke test. See [`tests/`](tests/) for details.
