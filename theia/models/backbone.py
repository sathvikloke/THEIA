"""Vision encoder wrapper.

Returns *patch* features (not just a pooled CLS), because the grounding head needs
per-location tokens to build spatial attention maps.

Output contract for encode():
  patch_tokens: [B, N, D]   N = H'*W' patches
  grid_hw:      (H', W')     so tokens can be reshaped back to a heatmap

Two corrections vs the first version:

1. Input normalisation. `window_hu` hands us [0,1] and the old code repeated it
   to 3 channels and fed it straight in. Every backbone here was pretrained on
   normalised input — timm's vit_base_patch16_224 declares mean=std=(0.5,0.5,0.5),
   i.e. it expects [-1,1] — so the encoder was seeing half its expected dynamic
   range. No crash, just quietly worse features. The backbone's own declared
   statistics are now applied.
2. Padded slices. `collate` pads short stacks with zeros, and the old slice-mean
   averaged those zeros into the patch tokens, diluting features for any patient
   with fewer slices than the batch max. The mean is now masked.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Fallback only; real values are read off the loaded backbone when available.
_DEFAULT_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_STD = (0.5, 0.5, 0.5)


def _normalization_from_transform(preprocess) -> tuple[tuple, tuple]:
    """Pull mean/std out of an open_clip preprocessing pipeline.

    open_clip hands back a torchvision Compose whose last op is a Normalize; read
    the statistics from it rather than assuming CLIP's, since BiomedCLIP does not
    necessarily share them.
    """
    for op in reversed(getattr(preprocess, "transforms", []) or []):
        mean, std = getattr(op, "mean", None), getattr(op, "std", None)
        if mean is not None and std is not None:
            return tuple(mean), tuple(std)
    print("[backbone] no Normalize found in preprocess; using (0.5,)*3")
    return _DEFAULT_MEAN, _DEFAULT_STD


class VisionEncoder(nn.Module):
    def __init__(self, name: str = "biomedclip", out_dim: int = 512, freeze: bool = False):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.model, self.native_dim, self.grid, mean, std = self._build(name)
        self.proj = nn.Identity() if self.native_dim == out_dim else nn.Linear(self.native_dim, out_dim)
        self.register_buffer("pix_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pix_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def _build(self, name: str):
        if name == "biomedclip":
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
                visual = model.visual
                mean, std = _normalization_from_transform(preprocess)
                # trunk.forward_features yields 768-dim PATCH tokens (the 512-dim
                # CLIP embedding is post-projection on the pooled CLS only).
                return visual, 768, (14, 14), mean, std
            except Exception as exc:  # noqa: BLE001
                # First run downloads weights; needs network access.
                print(f"[backbone] BiomedCLIP unavailable ({exc}); falling back to timm ViT")
                name = "timm_vit_b16"
        if name in ("timm_vit_b16", "open_clip_vit_b16"):
            import timm

            model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            cfg = getattr(model, "pretrained_cfg", None) or {}
            mean = tuple(cfg.get("mean", _DEFAULT_MEAN))
            std = tuple(cfg.get("std", _DEFAULT_STD))
            return model, 768, (14, 14), mean, std
        raise ValueError(f"unknown vision_encoder: {name}")

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Get patch tokens [B, N, native_dim] from whichever backbone is loaded."""
        n_patches = self.grid[0] * self.grid[1]
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)  # timm: [B, 1+N, D] or [B, N, D]
        else:
            feats = self.model.trunk.forward_features(x)  # open_clip visual trunk
        if feats.dim() == 3 and feats.shape[1] > n_patches:
            feats = feats[:, -n_patches:, :]  # drop CLS / register tokens
        return feats

    def encode(self, images: torch.Tensor,
               slice_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[int, int]]:
        """images: [B, S, 1, H, W] -> masked mean over slices -> patch tokens [B, N, D].

        slice_mask: [B, S] with 1 for real slices and 0 for collate padding.
        """
        b, s, c, h, w = images.shape
        x = images.view(b * s, c, h, w)
        if c == 1:
            x = x.repeat(1, 3, 1, 1)  # 3-channel stem
        x = (x - self.pix_mean.to(x.dtype)) / self.pix_std.to(x.dtype)
        tokens = self._forward_tokens(x)                 # [B*S, N, native]
        tokens = self.proj(tokens)                       # [B*S, N, D]
        n, d = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(b, s, n, d)
        if slice_mask is None:
            return tokens.mean(dim=1), self.grid
        m = slice_mask.to(tokens.device, tokens.dtype).view(b, s, 1, 1)
        pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return pooled, self.grid
