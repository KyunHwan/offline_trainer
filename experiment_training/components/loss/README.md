# loss

Loss function implementations registered in `LOSS_BUILDER_REGISTRY`. Each follows the factory pattern: the registered class is a factory whose `build()` method returns an `nn.Module`.

## Purpose

- Provide loss functions for training imitation learning policies
- Support standard losses (L2/MSE) and specialized losses (Sinkhorn-Knopp optimal transport)
- Provide utility loss functions for mixture-of-experts architectures

## How it fits into the pipeline

The training entrypoint calls `_build_loss()` which looks up the factory by `train.loss.type`, instantiates it with `train.loss.params`, and calls `build()`. The resulting `nn.Module` is passed to the trainer constructor. Whether and how the loss is used depends on the trainer implementation.

**Inputs:** `train.loss.type` and `train.loss.params` from config
**Outputs:** `nn.Module` loss function

## Key modules

| File | Registry key | Description |
|------|-------------|-------------|
| [`l2.py`](l2.py) | `l2_loss` | Factory wrapping `torch.nn.MSELoss` with configurable reduction |
| [`sinkhorn_knopp.py`](sinkhorn_knopp.py) | `sinkhorn_knopp` | Sinkhorn-Knopp optimal transport loss using `geomloss.SamplesLoss`. Computes OT distance between predicted and target (action, state) pairs with configurable state weighting |
| [`moe_gating_loss.py`](moe_gating_loss.py) | *(not registered)* | Utility functions for MoE auxiliary losses: `router_z_loss` (z-loss for router logits) and `switch_load_balancing_loss` (load balancing across experts) |

## Common workflows

### Use Sinkhorn-Knopp OT loss

```yaml
train:
  loss:
    type: "sinkhorn_knopp"
    params:
      p: 1
      lam_state: 0.2       # weight for state distance relative to action distance
      blur: 0.004           # Sinkhorn entropic regularization
      debias: true
      backend: "tensorized" # or "online" for KeOps
      scaling: 0.95
```

The `KOTSinkhornLoss.forward()` expects `(pred_action, target_action, state_pred, state_target)`. It concatenates action and scaled state features into a single vector and computes the OT distance. See [`sinkhorn_knopp.py:57-69`](sinkhorn_knopp.py).

### Use L2 loss

```yaml
train:
  loss:
    type: "l2_loss"
    params:
      reduction: "mean"
```

## Extension points

- Add new loss factories by implementing the `LossFactory` protocol and registering in `LOSS_BUILDER_REGISTRY`
- MoE gating losses in [`moe_gating_loss.py`](moe_gating_loss.py) are standalone functions — call them directly from trainer implementations that use mixture-of-experts

## Gotchas / invariants

- The loss factory `build()` method returns an `nn.Module`, which is moved to the training device by `_build_loss()`. See [`offline_trainer.py:129-138`](../../../trainer/offline_trainer.py)
- `KOTSinkhornLoss` uses `math.sqrt(lam_state)` to scale state features so that the squared Euclidean distance in the concatenated space equals `||a-a'||^2 + lam_state * ||s-s'||^2`
- The `sinkhorn_knopp` factory requires the `geomloss` package (installed by [`env_setup.sh`](../../../env_setup.sh))
