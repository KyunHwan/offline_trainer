# Tests

Pytest-based test suite covering configuration validation, registry plugins, checkpointing, and training.

## Purpose

- Verify config schema validation and error formatting
- Test registry plugin loading and component registration
- Validate checkpoint save/resume behavior
- Run a minimal end-to-end training smoke test

## How it fits into the pipeline

Tests import from the `trainer` package and exercise the public API. The [`conftest.py`](conftest.py) adds both the project root and `policy_constructor/` to `sys.path`.

## Test files

| File | Coverage |
|------|----------|
| [`test_minimal_train_smoke.py`](test_minimal_train_smoke.py) | End-to-end: creates a config, calls `train()`, verifies it completes without error |
| [`test_config_errors.py`](test_config_errors.py) | Config validation: verifies Pydantic errors include correct YAML paths and messages (e.g., `lr must be > 0`) |
| [`test_registry_plugins.py`](test_registry_plugins.py) | Registry: tests plugin loading, duplicate key detection, and base-class enforcement |
| [`test_checkpoint_resume.py`](test_checkpoint_resume.py) | Checkpointing: verifies save/load roundtrip for model and optimizer state dicts |
| [`test_tree_device_transfer.py`](test_tree_device_transfer.py) | Utils: tests `tree_map`, `move_to_device`, `cast_dtype` on nested structures |

## Running tests

```bash
pytest
```

Test paths are configured in [`pytest.ini`](../pytest.ini) (`testpaths = tests`).

## Gotchas / invariants

- Tests require `policy_constructor/` to be available (it's a git submodule). Run `git submodule update --init` if tests fail with import errors
- The smoke test creates a temporary config and runs 2 training steps with random data — it does not require a GPU
