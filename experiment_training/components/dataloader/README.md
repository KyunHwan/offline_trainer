# dataloader

Dataset factory implementations for loading training data. Each factory is registered in `DATASET_BUILDER_REGISTRY` and instantiated based on the `data.datamodule.type` config key.

## Purpose

- Load and prepare datasets for training (LeRobot HuggingFace datasets, local HDF5 episodic data)
- Compute and return normalization statistics alongside the dataset
- Apply image augmentations (color jitter, Gaussian blur, rotation)

## How it fits into the pipeline

The training entrypoint calls `_build_dataloader()`, which looks up the factory by `data.datamodule.type` in `DATASET_BUILDER_REGISTRY`, instantiates it with `data.datamodule.params`, and calls `build()`. The returned dict is expected to contain `"dataset"` (a `torch.utils.data.Dataset`) and optionally `"norm_stats"` (a dict of per-key `{mean, std}`).

**Inputs:** Config params from `data.datamodule.params`
**Outputs:** `{"dataset": Dataset, "norm_stats": {key: {"mean": ..., "std": ...}}}`

## Key modules

| File | Registry key | Description |
|------|-------------|-------------|
| [`lerobot_data.py`](lerobot_data.py) | `lerobot_dataset_factory` | Loads datasets from the [LeRobot](https://github.com/huggingface/lerobot) library. Supports configurable action horizons, observation history windows, and `delta_timestamps`. Applies `ColorJitter` + `GaussianBlur` augmentations. Returns dataset + normalization stats from `dataset.meta.stats` |
| [`episodic_data.py`](episodic_data.py) | `episodic_dataset_factory` | Loads episodic demonstration data from local HDF5 files. Supports multiple camera streams, image compression/decompression, configurable action chunks, temporal delays, and image downsampling. Computes normalization stats from the data |
| [`utils/`](utils/) | — | Shared data utilities |

### utils/

| File | Description |
|------|-------------|
| [`utils/config_loader.py`](utils/config_loader.py) | `ConfigLoader` — loads JSON task configurations defining observation/action structure. Parses field dimensions (position, orientation, sliced fields) and camera names from config |
| [`utils/utils.py`](utils/utils.py) | Helpers: `find_all_hdf5()` for directory scanning, `validate_hdf5_file()` for structure checking, `compute_norm_stats()` for mean/std computation, `get_episode_len()` for cached episode lengths |

## Common workflows

### Use LeRobot data

```yaml
plugins:
  - "experiment_training.components.dataloader.lerobot_data"

data:
  datamodule:
    type: "lerobot_dataset_factory"
    params:
      repo_id: "joon001001/igris-b-pnp-v4.1"
      root: "~/.cache/huggingface/lerobot"
      local_files_only: false
      HZ: 20
      action_horizon: 40
      obs_proprio_history: 40
      obs_images_history: 1
```

Key params for `lerobot_dataset_factory`:
- `repo_id` — HuggingFace dataset repository ID
- `root` — local cache directory (used when `local_files_only: true`)
- `HZ` — data frequency in Hz (default: 20)
- `action_horizon` — number of future action steps to include
- `obs_proprio_history` — number of proprioceptive observation history steps
- `obs_images_history` — number of image observation history steps

### Use episodic HDF5 data

```yaml
plugins:
  - "experiment_training.components.dataloader.episodic_data"

data:
  datamodule:
    type: "episodic_dataset_factory"
    params:
      dataset_dir: "/path/to/hdf5s"
      task_config: "/path/to/task_config.json"
      camera_names: ["cam_head", "cam_left", "cam_right"]
      chunk_size: 40
```

## Extension points

- Implement a new `DatasetFactory` to support other data formats (e.g., RoboSet, RLDS, custom formats)
- Add new image augmentation pipelines in the factory's `build()` method
- Normalization stats format must follow `{key: {"mean": list/tensor, "std": list/tensor}}` for compatibility with the training loop

## Gotchas / invariants

- Normalization stats are returned as plain Python lists or tensors from the dataset. The training loop converts them to tensors via `tree_map(map_list_to_torch, stats)` before use
- When `local_files_only: true`, `lerobot_data.py` sets `HF_HUB_OFFLINE=1` and expects the dataset to be cached at `root`. See [`lerobot_data.py:59-74`](lerobot_data.py)
- The episodic data loader computes normalization stats from up to 100k samples for efficiency. See [`utils/utils.py`](utils/utils.py)
- `delta_timestamps` in LeRobot are computed from `HZ`, `action_horizon`, and observation history params. They define the temporal windows for each data key
