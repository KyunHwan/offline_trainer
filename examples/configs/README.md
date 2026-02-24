# Example Configs

YAML configuration files demonstrating common configuration patterns for the training framework.

## Purpose

- Provide ready-to-run example configs for testing and learning
- Demonstrate config features: minimal setup, custom components via plugins, optimizer/scheduler swapping

## Configs

| File | Description |
|------|-------------|
| [`minimal.yaml`](minimal.yaml) | Minimal configuration — uses a policy_constructor sequential MLP model with default components. Runs 2 steps for 1 epoch |
| [`swap_optim_sched.yaml`](swap_optim_sched.yaml) | Shows how to swap the optimizer (SGD with momentum) and scheduler (StepLR) via config keys |
| [`custom_trainer_custom_data.yaml`](custom_trainer_custom_data.yaml) | Loads plugin modules from [`extensions/`](../extensions/) to use custom trainer and data module types |

## How it fits into the pipeline

These configs are loaded by [`trainer/config/loader.py`](../../trainer/config/loader.py) and validated against the Pydantic schema in [`trainer/config/schemas.py`](../../trainer/config/schemas.py).

## Common workflows

### Run a minimal example

```bash
python trainer/offline_trainer.py --train_config examples/configs/minimal.yaml
```

### Use custom components

```bash
python trainer/offline_trainer.py --train_config examples/configs/custom_trainer_custom_data.yaml
```

## Gotchas / invariants

- `model.config_path` in these examples points to policy_constructor YAML configs (e.g., `policy_constructor/configs/examples/sequential_mlp.yaml`). Paths are resolved relative to the config file's directory or the project root
- The `defaults` composition feature of the config loader can be used to layer configs (see [`trainer/config/loader.py`](../../trainer/config/loader.py))
