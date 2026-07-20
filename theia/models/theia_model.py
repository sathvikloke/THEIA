"""THEIA: vision encoder + three heads, one joint forward.

forward() returns everything the loss needs:
  logits:    {gene: [B,2]}
  attn_maps: [B, Q, H', W']   grounding, supervised against the ROI mask
  gen_loss:  scalar (teacher-forced rationale loss), or None at inference

Design note: the classifier pools from the *grounded* region embeddings, not a raw
CLS token, so the prediction and the localization share a representation — the
model cannot point one place and decide from another.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import VisionEncoder
from .heads import ClassifierHead, GenerationHead, GroundingHead


class Theia(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.genes = [g.lower() for g in cfg.data.target_genes]
        self.vision = VisionEncoder(m.vision_encoder, m.vision_dim, m.freeze_vision)
        self.grounding = GroundingHead(m.vision_dim, m.grounding_tokens)
        self.classifier = ClassifierHead(m.vision_dim, cfg.data.target_genes,
                                         m.classifier_hidden, m.dropout)
        self.generation = GenerationHead(m.vision_dim, m.lm_name, m.lm_max_new_tokens)
        if cfg.lora.enabled:
            self._apply_lora(cfg.lora)

    def _apply_lora(self, lora_cfg) -> None:
        try:
            from peft import LoraConfig, get_peft_model

            peft_cfg = LoraConfig(
                r=lora_cfg.r, lora_alpha=lora_cfg.alpha, lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules), bias="none",
            )
            self.generation.lm = get_peft_model(self.generation.lm, peft_cfg)
        except Exception as exc:  # noqa: BLE001
            # TODO(theia): target_modules must match the chosen LM's attn names.
            print(f"[theia] LoRA not applied ({exc}); training full LM instead")

    def forward(self, batch, device, generate: bool = False) -> dict:
        images = batch["images"].to(device)
        patch_tokens, grid = self.vision.encode(images)
        g = self.grounding(patch_tokens, grid)
        pooled = self.grounding.pooled_from_regions(g["region_emb"])
        logits = self.classifier(pooled)

        out = dict(logits=logits, attn_maps=g["attn_maps"], region_emb=g["region_emb"])
        if generate:
            out["text"] = self.generation.generate(g["region_emb"], device)
        else:
            out["gen_loss"] = self.generation(g["region_emb"], batch["report"], device)
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
