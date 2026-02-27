"""DSRL trainer: wires OpenPiBatchedWrapper, Q-function, and Noise Latent Actor.

Data flow:
  1. Frozen shared RadioV3 backbone processes cam_head, cam_left, cam_right
     → (B, 1024, H', W') each (features detached from grad graph).
  2. Frozen DA3 produces depth features from cam_head → rearranged to (B, 1024, Hd, Wd).
  3. q_function_processor receives {head, left, right, head_depth, proprio, action_gt}
     → flattened vector → Q_Function → Q(s, a_gt) [critic update].
  4. noise_processor receives {head, left, right, head_depth, proprio}
     → flattened vector → Noise_Latent_Actor → ε.
  5. Actor update: Q(s, ε) evaluated directly — gradient flows noise_actor ← ε ← Q.
     NOTE: OpenPiBatchedWrapper.forward() detaches tensors via numpy round-trip,
     so gradients CANNOT flow back through guided_action = openpi(obs, noise=ε).
     ε is therefore used as the action proxy for the actor loss.
  6. OpenPI guided action openpi(obs, noise=ε) is logged for monitoring only.

Register via plugins in the experiment YAML:
  - "experiment_training.components.trainer.reinforcement_learning.dsrl_openpi.dsrl_openpi_trainer"
"""
from __future__ import annotations

from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.trainer.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("dsrl_openpi_trainer")
class DSRLOpenPITrainer(nn.Module):
    """Actor-critic trainer combining OpenPI (diffusion backbone) with DSRL blocks.

    Models expected in ``models`` dict:
      - ``openpi_model``:         GraphModel wrapping OpenPiBatchedWrapper
      - ``backbone``:             GraphModel wrapping RadioV3 (shared for all cameras)
      - ``da3``:                  GraphModel wrapping DepthAnything3Bridge (frozen)
      - ``q_function_processor``: GraphModel wrapping QFunctionImgDepthProprioProcessor
      - ``q_function``:           GraphModel wrapping Q_Function
      - ``noise_processor``:      GraphModel wrapping NoiseActorImgDepthProprioProcessor
      - ``noise_actor``:          GraphModel wrapping Noise_Latent_Actor

    Loss weights are class-level constants; subclass or set after construction to override.
    """

    LAMBDA_Q: float = 0.5
    LAMBDA_ACTOR: float = 0.1

    def __init__(
        self,
        *,
        models: nn.ModuleDict,
        optimizers: dict[str, torch.optim.Optimizer],
        loss: nn.Module | None,
        device: torch.device,
    ):
        super().__init__()
        self.models = models
        self.optimizers = optimizers
        self.loss = loss  # not used — losses are computed inline
        self.device = device

    # ------------------------------------------------------------------
    # Module accessors (unwrap DDP to reach GraphModel.graph_modules)
    # ------------------------------------------------------------------

    def _unwrap(self, key: str) -> nn.Module:
        m = self.models[key]
        return m.module if isinstance(m, DDP) else m

    def _openpi(self):
        return self._unwrap("openpi_model").graph_modules["openpi_model"]

    def _backbone(self):
        # Shared RadioV3 — called once per camera in forward()
        return self._unwrap("backbone").graph_modules["radiov3"]

    def _q_proc(self):
        return self._unwrap("q_function_processor").graph_modules["q_proc"]

    def _q_fn(self):
        return self._unwrap("q_function").graph_modules["q_fn"]

    def _noise_proc(self):
        return self._unwrap("noise_processor").graph_modules["noise_proc"]

    def _noise_actor(self):
        return self._unwrap("noise_actor").graph_modules["actor"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_depth(self, image: torch.Tensor) -> torch.Tensor:
        """Run DA3 on head camera and rearrange to (B, D, Hp, Wp).

        DA3 returns (B, N_layers, Hp, Wp, feature_dim).
        We export one layer and transpose to channel-first format.
        """
        da3 = self._unwrap("da3").graph_modules["da3"]
        raw = da3(image=image, export_feat_layers=[18])  # (B, 1, Hp, Wp, 1024)
        return einops.rearrange(raw[:, 0], "b h w d -> b d h w")  # (B, 1024, Hp, Wp)

    def _build_openpi_obs(self, data: dict) -> dict[str, Any]:
        """Build the raw observation dict expected by OpenPiBatchedWrapper."""
        return {
            "proprio": data["observation.state"],
            "head":    data["observation.images.cam_head"],
            "left":    data["observation.images.cam_left"],
            "right":   data["observation.images.cam_right"],
        }

    def _build_processor_data(
        self,
        data: dict,
        head_feats: torch.Tensor,
        left_feats: torch.Tensor,
        right_feats: torch.Tensor,
        depth_feats: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assemble the dict consumed by both processor modules.

        The processors expect ``proprio`` as (B, obs_history, obs_dim).
        LeRobot datasets with obs_proprio_history > 1 already deliver this shape;
        when history=1 the tensor arrives as (B, obs_dim) and is unsqueezed.
        """
        proprio = data["observation.state"]
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)  # (B, 1, obs_dim)

        d = {
            "head":       head_feats,
            "left":       left_feats,
            "right":      right_feats,
            "head_depth": depth_feats,
            "proprio":    proprio,
        }
        if action is not None:
            d["action"] = action
        return d

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data: dict[str, Any],
        epoch: int,
        total_epochs: int,
        iterations: int,
    ) -> dict[str, Any]:
        loss_dict: dict[str, Any] = {}

        # 1. ── Perception: frozen backbone, one call per camera ─────────────
        # Backbone and DA3 are frozen; wrap in no_grad for memory efficiency.
        with torch.no_grad():
            backbone = self._backbone()
            head_feats, _  = backbone(data["observation.images.cam_head"])
            left_feats, _  = backbone(data["observation.images.cam_left"])
            right_feats, _ = backbone(data["observation.images.cam_right"])
            # each: (B, 1024, H', W')

            # 2. Depth features from DA3 (frozen)
            depth_feats = self._extract_depth(data["observation.images.cam_head"])
            # (B, 1024, Hd, Wd)

        # Detach so downstream trainable modules start a fresh computation graph.
        head_feats  = head_feats.detach()
        left_feats  = left_feats.detach()
        right_feats = right_feats.detach()
        depth_feats = depth_feats.detach()

        # 3. ── Q-function: evaluate ground-truth dataset actions ──────────
        q_proc_data = self._build_processor_data(
            data, head_feats, left_feats, right_feats, depth_feats,
            action=data["action"],  # (B, T_a, a_dim)
        )
        q_flat  = self._q_proc()(q_proc_data)   # (B, flat_q_dim)
        q_value = self._q_fn()(q_flat)           # (B, 1)

        # Q-function loss.
        # Replace the placeholder target with your RL objective:
        # e.g., CQL conservative penalty, IQL expectile, or reward-based TD.
        q_target = torch.zeros_like(q_value)     # ← substitute with RL target
        L_q = F.mse_loss(q_value, q_target)

        loss_dict["q_loss"]  = L_q
        loss_dict["q_value"] = q_value.mean().detach().item()

        # 4. ── Noise actor: generate action candidates from observation ────
        noise_proc_data = self._build_processor_data(
            data, head_feats, left_feats, right_feats, depth_feats,
            # no action — noise actor conditions only on state
        )
        noise_flat = self._noise_proc()(noise_proc_data)  # (B, flat_n_dim)
        noise_eps  = self._noise_actor()(noise_flat)       # (B, T_a, a_dim)
        # noise_eps is tanh-squashed to [-1, 1]; shape matches action dimensions.

        # 5. ── Actor loss: evaluate Q(s, ε) directly ──────────────────────
        # OpenPiBatchedWrapper.forward() internally detaches tensors via a
        # numpy round-trip, breaking gradient flow back to the noise actor.
        # Instead, we evaluate Q directly on noise_eps as the action candidate.
        # Gradient path: L_actor → Q(s, ε) → q_proc(ε) → ε → noise_actor ✓
        #
        # Optionally, record the OpenPI-guided action for monitoring only:
        obs_raw = self._build_openpi_obs(data)
        with torch.no_grad():
            guided_action = self._openpi().forward(obs_raw, noise=noise_eps)
            # (B, T_a, a_dim) — for logging/monitoring only

        q_actor_proc_data = self._build_processor_data(
            data, head_feats, left_feats, right_feats, depth_feats,
            action=noise_eps,  # use noise_eps directly — differentiable path
        )
        q_actor_flat  = self._q_proc()(q_actor_proc_data)  # (B, flat_q_dim)
        q_actor_value = self._q_fn()(q_actor_flat)          # (B, 1)

        L_actor = -q_actor_value.mean()  # minimise -Q  ⟺  maximise Q
        loss_dict["actor_loss"]  = L_actor
        loss_dict["q_actor_val"] = q_actor_value.mean().detach().item()

        # 6. ── Monitoring: log guided-action Q value (no gradient) ───────
        with torch.no_grad():
            q_guided_proc = self._build_processor_data(
                data, head_feats, left_feats, right_feats, depth_feats,
                action=guided_action,
            )
            q_guided_val = self._q_fn()(self._q_proc()(q_guided_proc))
        loss_dict["q_guided_val"] = q_guided_val.mean().detach().item()

        # 7. ── Combined loss ───────────────────────────────────────────────
        loss_dict["total"] = (
            self.LAMBDA_Q     * L_q
            + self.LAMBDA_ACTOR * L_actor
        )
        return loss_dict

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(
        self,
        data: dict[str, Any],
        epoch: int,
        total_epochs: int,
        iterations: int,
    ) -> dict[str, Any]:
        self._ready_train()
        self._zero_grad()
        loss = self.forward(data, epoch, total_epochs, iterations)
        self._backward(loss)
        detached = self._clip_get_grad_norm(loss, clip_val=1.0)
        self._step()
        detached = self._detached_loss(detached)
        detached = self._get_lr(detached)
        return detached

    # ------------------------------------------------------------------
    # Boilerplate (identical pattern to existing trainers in the project)
    # ------------------------------------------------------------------

    def _ready_train(self):
        for key in self.optimizers:
            self.models[key].train()
            if hasattr(self.optimizers[key], "train"):
                self.optimizers[key].train()

    def _zero_grad(self):
        for key in self.optimizers:
            self.optimizers[key].zero_grad(set_to_none=True)

    def _backward(self, loss: dict[str, Any]):
        for v in loss.values():
            if isinstance(v, torch.Tensor):
                v.backward()

    def _step(self):
        for key in self.optimizers:
            self.optimizers[key].step()

    def _detached_loss(self, loss: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.detach().item() if isinstance(v, torch.Tensor) else v
            for k, v in loss.items()
        }

    def _clip_get_grad_norm(
        self, loss: dict[str, Any], clip_val: float = float("inf")
    ) -> dict[str, Any]:
        for name in self.models:
            if name in self.optimizers:
                loss[f"{name} grad_norm"] = (
                    torch.nn.utils.clip_grad_norm_(
                        self.models[name].parameters(), max_norm=clip_val
                    )
                    .detach()
                    .item()
                )
        return loss

    def _get_lr(self, loss: dict[str, Any]) -> dict[str, Any]:
        for name in self.models:
            if name in self.optimizers:
                loss[f"{name} lr"] = self.optimizers[name].param_groups[0]["lr"]
        return loss
