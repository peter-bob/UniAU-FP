"""
ema.py — Exponential Moving Average for model weights
======================================================
Maintains a shadow copy of model parameters that is a running
exponential average of the training weights. Use the EMA copy
for validation/inference to get smoother, better-generalizing
predictions.

Usage in training loop:
    ema = EMA(model, decay=0.999)
    for epoch in ...:
        train(model, ...)        # normal training updates model
        ema.update()             # update shadow weights

        ema.apply_shadow()       # swap to EMA weights
        val_rmse = test(model, ...)  # evaluate with EMA weights
        ema.restore()            # swap back to training weights
"""

import torch
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.

    shadow_param = decay * shadow_param + (1 - decay) * model_param

    Args:
        model: the model to track
        decay: EMA decay factor (0.999 is typical; higher = slower update)
        warmup_steps: number of update() calls before EMA starts
                      (use 0 to start immediately)
    """

    def __init__(self, model, decay=0.999, warmup_steps=0):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Store shadow parameters (deep copy of initial weights)
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self):
        """Ramp up decay during warmup for stability."""
        if self.step_count < self.warmup_steps:
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        return self.decay

    def update(self):
        """Call after each optimizer.step() to update shadow weights."""
        self.step_count += 1
        decay = self._get_decay()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def apply_shadow(self):
        """Swap model weights with shadow weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}