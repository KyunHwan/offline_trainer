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
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data: dict[str, Any],
        stats: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Return 
            - Action Q Loss
            - Noise Q Loss
            - Noise Q Value
        """
        # ------------------------------------------------------------------
        # actor critic
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # noise latent actor critic
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # action critic step
        # ------------------------------------------------------------------

        
        return loss_dict

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(
        self,
        data: dict[str, Any],
        stats: dict[str, Any]
    ) -> dict[str, Any]:
        self._ready_train()
        self._zero_grad()
        loss = self.forward(data, stats)
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
