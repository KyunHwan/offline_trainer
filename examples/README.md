# Examples

Runnable examples that demonstrate the core training flow and the plugin system.

## Purpose

- Provide minimal working examples for first-time users
- Demonstrate how to create and register custom components
- Show different configuration patterns (minimal, custom components, optimizer swapping)

## How it fits into the pipeline

These examples use the same training framework as production experiments but with lightweight components (random data, small models) for quick iteration.

## Contents

| Path | Description |
|------|-------------|
| [`run_from_python.py`](run_from_python.py) | Minimal entrypoint — calls `train()` from [`trainer/offline_trainer.py`](../trainer/offline_trainer.py) with the minimal config |
| [`configs/`](configs/) | Example YAML configs demonstrating different configuration patterns |
| [`extensions/`](extensions/) | Importable modules that register custom components via the plugin system |

## Quick start

```bash
python examples/run_from_python.py
```

This runs a minimal training loop with a small sequential MLP model and random data for 2 steps.
