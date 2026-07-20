"""Vision encoder wrapper.

Returns *patch* features (not just a pooled CLS), because the grounding head needs
per-location tokens to build spatial attention maps.

Output contract for encode():
  patch_tokens: [B, N, D]   N = H'*W' patches
  grid_hw:      (H', W')     so tokens can be reshaped back to a heatmap
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, name: str = "biomedclip", out_dim: int = 512, freeze: bool = False):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.model, self.native_dim, self.grid = self._build(name)
        self.proj = nn.Identity() if self.native_dim == out_dim else nn.Linear(self.native_dim, out_dim)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def _build(self, name: str):
        if name == "biomedclip":
            try:
                import open_clip

                model, _, _ = open_clip.create_model_and_transforms(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
                visual = model.visual
                # trunk.forward_features yields 768-dim PATCH tokens (the 512-dim
                # CLIP embedding is post-projection on the pooled CLS only).
                return visual, 768, (14, 14)
            except Exception as exc:  # noqa: BLE001
                # TODO(theia): first run downloads weights; needs network + auth.
                print(f"[backbone] BiomedCLIP unavailable ({exc}); falling back to timm ViT")
                name = "timm_vit_b16"
        if name in ("timm_vit_b16", "open_clip_vit_b16"):
            import timm

            model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            return model, 768, (14, 14)
        raise ValueError(f"unknown vision_encoder: {name}")

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Get patch tokens [B, N, native_dim] from whichever backbone is loaded."""
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)  # timm: [B, 1+N, D] or [B, N, D]
            if feats.dim() == 3 and feats.shape[1] == self.grid[0] * self.grid[1] + 1:
                feats = feats[:, 1:, :]  # drop CLS
            return feats
        # open_clip visual trunk path
        feats = self.model.trunk.forward_features(x)
        if feats.dim() == 3 and feats.shape[1] > self.grid[0] * self.grid[1]:
            feats = feats[:, 1:, :]
        return feats

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """images: [B, S, 1, H, W] -> mean over slices -> patch tokens [B, N, D]."""
        b, s, c, h, w = images.shape
        x = images.view(b * s, c, h, w)
        if c == 1:
            x = x.repeat(1, 3, 1, 1)  # 3-channel stem
        tokens = self._forward_tokens(x)                 # [B*S, N, native]
        tokens = self.proj(tokens)                       # [B*S, N, D]
        n, d = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(b, s, n, d).mean(dim=1)     # slice-mean -> [B, N, D]
        return tokens, self.grid
