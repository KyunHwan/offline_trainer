"""Trainer plugin for OpenPiBatchedWrapper.

Bridges LeRobot DataLoader output to the wrapper's expected input format and
handles loss computation, backward pass, and optimizer step.

Register via YAML plugins:
  - "experiment_training.components.trainer.imitation_learning.openpi_batched.openpi_batched_trainer"
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from trainer.trainer.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("openpi_batched_trainer")
class OpenPiBatchedTrainer(nn.Module):
    """Trainer for OpenPiBatchedWrapper.

    Converts LeRobot-format data batches into the observation dict expected by
    ``OpenPiBatchedWrapper``, calls ``compute_loss()``, and performs the
    optimiser step.
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
        self.loss = loss  # Not used — loss is computed inside the wrapper
        self.device = device

    # ------------------------------------------------------------------
    # Forward / train_step
    # ------------------------------------------------------------------

    def forward(
        self,
        data: dict[str, Any],
        epoch: int,
        total_epochs: int,
        iterations: int,
    ) -> dict[str, Any]:
        loss_dict: dict[str, Any] = {}

        # ---- Build observation dict from LeRobot batch ----
        obs: dict[str, Any] = {
            "proprio": data["observation.state"],  # (B, state_dim)
            "head": data["observation.images.cam_head"],  # (B, 3, H, W)
            "left": data["observation.images.cam_left"],  # (B, 3, H, W)
            "right": data["observation.images.cam_right"],  # (B, 3, H, W)
        }
        if "prompt" in data:
            obs["prompt"] = data["prompt"]

        actions = data["action"]  # (B, action_horizon, action_dim)

        # ---- Get the OpenPiBatchedWrapper module ----
        openpi_wrapper = self._get_openpi_wrapper()

        # ---- Compute per-element loss ----
        loss_per_elem = openpi_wrapper.compute_loss(obs, actions)

        # ---- Aggregate ----
        total_loss = loss_per_elem.mean()

        loss_dict["total"] = total_loss
        loss_dict["mse_per_element"] = total_loss.detach().clone().item()

        return loss_dict

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
        detached_loss = self._clip_get_grad_norm(loss=loss, clip_val=1.0)
        self._step()
        detached_loss = self._detached_loss(detached_loss)
        detached_loss = self._get_lr(detached_loss)
        return detached_loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_openpi_wrapper(self):
        """Retrieve the OpenPiBatchedWrapper from the GraphModel."""
        graph_model = self.models["main"]
        # Unwrap DDP if needed
        model = graph_model.module if isinstance(graph_model, DDP) else graph_model
        return model.graph_modules["openpi_model"]

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
        detached = {}
        for k, v in loss.items():
            if isinstance(v, torch.Tensor):
                detached[k] = v.detach().item()
            else:
                detached[k] = v
        return detached

    def _clip_get_grad_norm(
        self, loss: dict[str, Any], clip_val: float = float("inf")
    ) -> dict[str, Any]:
        for model_name in self.models:
            if model_name in self.optimizers:
                loss[f"{model_name} grad_norm"] = (
                    torch.nn.utils.clip_grad_norm_(
                        self.models[model_name].parameters(),
                        max_norm=clip_val,
                    )
                    .detach()
                    .item()
                )
        return loss

    def _get_lr(self, loss: dict[str, Any]) -> dict[str, Any]:
        for model_name in self.models:
            if model_name in self.optimizers:
                loss[f"{model_name} lr"] = self.optimizers[
                    model_name
                ].param_groups[0]["lr"]
        return loss
