# Extension Examples

Sample modules demonstrating how to register custom components into the global registries.

## Purpose

- Show the plugin pattern for adding custom trainers, data modules, losses, optimizers, and other components
- Provide copy-paste starting points for new component implementations

## How it fits into the pipeline

These modules are referenced in config YAML `plugins` lists. When the training entrypoint imports them, their `@REGISTRY.register()` decorators execute, making the components available for lookup.

```yaml
plugins:
  - "examples.extensions.custom_trainer"
  - "examples.extensions.custom_data"
```

## Files

| File | Registry | Key | Description |
|------|----------|-----|-------------|
| [`custom_trainer.py`](custom_trainer.py) | `TRAINER_REGISTRY` | `custom_trainer_v1` | Thin wrapper around the default trainer, demonstrating the registration pattern |
| [`custom_data.py`](custom_data.py) | `DATAMODULE_REGISTRY` | `custom_data_v1` | Custom data module that generates random regression data |

## Common workflows

### Create a new custom component

1. Create a Python file in any importable location
2. Import the appropriate registry from `trainer.registry`
3. Decorate your class with `@REGISTRY.register("my_key")`
4. Implement the required protocol from [`trainer/templates/`](../../trainer/templates/)
5. Add the module path to your config's `plugins` list

See the root [README](../../README.md#extending-the-system) for protocol signatures and detailed guidance.
