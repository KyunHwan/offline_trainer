# Utils

Shared utilities used across the training stack.

## Purpose

- Traverse and transform nested data structures (dicts, lists, tuples, dataclasses)
- Move tensors between devices and cast dtypes
- Seed all RNG sources for reproducible training
- Safely select values from structured objects
- Dynamically instantiate classes with filtered kwargs

## How it fits into the pipeline

These utilities are used throughout the training loop. `tree_map` transforms batch data, `move_to_device`/`cast_dtype` handle the CPU→GPU transfer, `set_global_seed` ensures reproducibility across ranks, and `instantiate` bridges config params to constructor calls.

## Key modules

| File | Description |
|------|-------------|
| [`tree.py`](tree.py) | `tree_map(fn, tree)` — recursively applies `fn` to leaves in nested structures (dict, list, tuple, namedtuple, dataclass). `tree_flatten(tree)` collects all leaves into a flat list. `is_tensor_leaf` predicate for tensor-specific traversal |
| [`device.py`](device.py) | `select_device(requested)` — resolves `"auto"` to CUDA/CPU. `move_to_device(batch, device)` — moves all tensors in nested structure to device. `cast_dtype(batch, dtype)` — casts floating-point tensors to a given dtype. Both use `tree_map` internally |
| [`seed.py`](seed.py) | `set_global_seed(seed, deterministic)` — seeds Python, NumPy, and PyTorch RNGs. Sets `PYTHONHASHSEED`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and optionally enables deterministic mode. `seed_worker(worker_id)` — seeds dataloader workers based on `torch.initial_seed()` |
| [`selection.py`](selection.py) | `select(obj, key)` — retrieves a value by string key (dict), integer index (tuple/list), or attribute name. Returns `obj` unchanged when `key` is `None` |
| [`import_utils.py`](import_utils.py) | `import_module(path)` — thin wrapper around `importlib.import_module`. `instantiate(obj, params, **extra)` — calls a class/function with merged `params` and `extra` kwargs, filtering to only accepted parameters via signature inspection |

## Common workflows

### Transform a batch dict

```python
from trainer.utils.tree import tree_map

# Convert all list-valued stats to tensors
stats_tensors = tree_map(lambda lst: torch.tensor(lst), stats)
```

### Move data to GPU

```python
from trainer.utils.device import move_to_device, cast_dtype

data = cast_dtype(data, torch.float32)
data = move_to_device(data, torch.device("cuda:0"))
```

## Extension points

- `tree_map` accepts an optional `is_leaf` predicate to customize what counts as a leaf node
- `instantiate` passes `**kwargs` through if the target accepts `**kwargs`; otherwise it filters to only named parameters

## Gotchas / invariants

- `move_to_device` is a no-op for tensors already on the target device (avoids unnecessary copies)
- `cast_dtype` only affects floating-point tensors — integer tensors are left unchanged
- `set_global_seed` optionally enables `torch.use_deterministic_algorithms(True)` when `deterministic=True`, which may raise errors for operations without deterministic implementations
- `seed_worker` is designed for use as the `worker_init_fn` parameter of `DataLoader`
