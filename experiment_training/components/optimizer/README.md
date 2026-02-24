# optimizer

Optimizer factory implementations registered in `OPTIMIZER_BUILDER_REGISTRY`. Each factory's `build(params)` method receives model parameters and returns a configured `torch.optim.Optimizer`.

## Purpose

- Provide optimizer + learning rate schedule combinations for training
- Integrate scheduler stepping inside the optimizer (no separate scheduler step needed in the training loop)
- Support checkpoint save/load of both optimizer and scheduler state

## How it fits into the pipeline

The training entrypoint calls `_build_optimizers()` which iterates over `model.component_optims` — each model component gets its own optimizer. The factory is looked up by type key, instantiated with the params, and `build(model.parameters())` is called.

**Inputs:** `component_optims[name].type` and `.params` from config, plus model parameters
**Outputs:** `torch.optim.Optimizer` (with integrated scheduler)

## Key modules

| File | Registry key | Description |
|------|-------------|-------------|
| [`adamw_cosine_decay.py`](adamw_cosine_decay.py) | `adamw_warmup_cosine_decay` | `AdamWWithWarmupCosine` — AdamW with a linear warmup → cosine decay schedule. Scheduler steps inside `optimizer.step()`. Saves/loads scheduler state within the optimizer state dict |
| [`adamw_onecyclelr.py`](adamw_onecyclelr.py) | `adamw_cosine_schedule` | `AdamWWithOneCycle` — AdamW with PyTorch's `OneCycleLR` schedule integrated. Same step-inside-step pattern |
| [`schedule_free_radam.py`](schedule_free_radam.py) | `schedule_free_radam` | Factory wrapping `schedulefree.RAdamScheduleFree` from the [schedulefree](https://github.com/facebookresearch/schedule_free) library |

## Common workflows

### Use AdamW with warmup cosine decay

```yaml
model:
  component_optims:
    info_embedder:
      type: "adamw_warmup_cosine_decay"
      params:
        peak_lr: 1.0e-4
        start_lr: 1.0e-6
        end_lr: 1.0e-6
        total_steps: 200000
        warmup_steps: 2000
        betas: [0.9, 0.999]
        eps: 1.0e-8
        weight_decay: 0.01
```

Key params for `adamw_warmup_cosine_decay`:
- `peak_lr` — maximum learning rate (reached at end of warmup)
- `start_lr` — learning rate at step 0
- `end_lr` — learning rate at the final step
- `total_steps` — total number of optimizer updates
- `warmup_steps` — steps for linear warmup from `start_lr` to `peak_lr`

### Use AdamW with OneCycle

```yaml
model:
  component_optims:
    backbone:
      type: "adamw_cosine_schedule"
      params:
        lr: 1.0e-4
        max_lr: 1.0e-3
        total_steps: 100000
        pct_start: 0.1
```

## Extension points

- Add new optimizer factories by implementing the `OptimizerFactory` protocol
- The pattern of integrating schedulers inside the optimizer's `step()` keeps the training loop simple — it only calls `optimizer.step()`

## Gotchas / invariants

- Schedulers are stepped inside `optimizer.step()`, so the training loop must **not** step a scheduler separately. See [`adamw_cosine_decay.py:453-457`](adamw_cosine_decay.py)
- Optimizer state dicts include scheduler state under the `"scheduler"` key. `load_state_dict` restores both. See [`adamw_cosine_decay.py:459-470`](adamw_cosine_decay.py)
- One optimizer is created per model component listed in `component_optims`. Frozen models (no trainable params) are skipped. See [`offline_trainer.py:199-222`](../../../trainer/offline_trainer.py)
- `peak_lr` must be > 0, and `start_lr`/`end_lr` must be <= `peak_lr`. Validation is performed at construction time
