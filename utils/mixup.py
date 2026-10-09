"""
mixup.py — Mixup data augmentation for regression tasks
========================================================
Mixes pairs of training samples and their labels linearly:
    x_mixed = lambda * x_i + (1 - lambda) * x_j
    y_mixed = lambda * y_i + (1 - lambda) * y_j

where lambda ~ Beta(alpha, alpha).

This is one of the most effective regularizers for deep learning,
and works naturally with regression (no label smoothing tricks needed).

Usage:
    mixup = MixupRegression(alpha=0.2)

    for images, labels in dataloader:
        images, labels = mixup(images, labels)  # mixed
        logits = model(images, ...)
        loss = criterion(logits, labels)         # standard loss on mixed targets
"""

import torch


class MixupRegression:
    """
    Mixup augmentation for regression.

    Args:
        alpha: Beta distribution parameter. Higher = more mixing.
               0.2 is a good default; 0.4 for stronger regularization.
        prob:  Probability of applying mixup to a given batch.
               1.0 = always mix (default).
    """

    def __init__(self, alpha=0.2, prob=1.0):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images, labels):
        """
        Args:
            images: (B, C, H, W) input images
            labels: (B, num_classes) regression targets
        Returns:
            mixed_images, mixed_labels
        """
        if self.alpha <= 0 or torch.rand(1).item() > self.prob:
            return images, labels

        batch_size = images.shape[0]

        # Sample mixing coefficient from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # Random permutation for pairing
        index = torch.randperm(batch_size, device=images.device)

        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels


class CutMixRegression:
    """
    CutMix augmentation for regression — cuts a rectangular patch from one
    image and pastes it onto another, mixing labels proportionally to area.

    Args:
        alpha: Beta distribution parameter for area ratio
        prob:  Probability of applying CutMix
    """

    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images, labels):
        if self.alpha <= 0 or torch.rand(1).item() > self.prob:
            return images, labels

        B, C, H, W = images.shape
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        index = torch.randperm(B, device=images.device)

        # Generate random bounding box
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cy = torch.randint(0, H, (1,)).item()
        cx = torch.randint(0, W, (1,)).item()

        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual area
        actual_lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        mixed_labels = actual_lam * labels + (1 - actual_lam) * labels[index]

        return mixed_images, mixed_labels