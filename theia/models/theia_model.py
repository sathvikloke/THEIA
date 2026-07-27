"""THEIA: vision encoder + three heads, one joint forward.

forward() returns everything the loss needs:
  logits:    {gene: [B,2]}
  attn_maps: [B, Q, H', W']   grounding, supervised against the ROI mask
  gen_loss:  scalar (teacher-forced rationale loss), or None at inference

Design note: the classifier pools from the *grounded* region embeddings, not a raw
CLS token, so the prediction and the localization share a representation.

Honest scope of that claim: the grounding loss supervises the element-wise *max*
over the Q region queries, while the classifier consumes their *mean*. A query
that attends off-tumor still feeds the prediction and is never penalised, as long
as some other query covers the tumor. `pooling` below makes the two consistent
when set to "max"; the default stays "mean" so existing behaviour is unchanged,
but the README no longer claims the coupling is airtight.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import VisionEncoder
from .heads import ClassifierHead, GenerationHead, GroundingHead


class LoRAUnavailable(RuntimeError):
    """Raised when LoRA is requested but cannot be applied."""


class Theia(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.genes = [g.lower() for g in cfg.data.target_genes]
        self.pooling = getattr(m, "region_pooling", "mean")
        self.vision = VisionEncoder(m.vision_encoder, m.vision_dim, m.freeze_vision)
        self.grounding = GroundingHead(m.vision_dim, m.grounding_tokens)
        self.classifier = ClassifierHead(m.vision_dim, cfg.data.target_genes,
                                         m.classifier_hidden, m.dropout)
        self.generation = GenerationHead(m.vision_dim, m.lm_name, m.lm_max_new_tokens)
        self.lora_applied = False
        if cfg.lora.enabled:
            self._apply_lora(cfg.lora)

    def _apply_lora(self, lora_cfg) -> None:
        """Apply LoRA to the LM, or fail loudly.

        This used to swallow every exception and print a note. That is dangerous
        rather than forgiving: a run that silently trains the full LM produces a
        checkpoint whose state_dict keys do not match a model rebuilt with LoRA
        active, so `build_cases.py` fails at load time with an opaque key error
        long after the 12-hour training run finished. Fail at construction, where
        the message is actionable, unless the config explicitly allows a fallback.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise LoRAUnavailable(
                "lora.enabled is true but `peft` is not installed. "
                "Install it (pip install peft) or set lora.enabled: false."
            ) from exc

        peft_cfg = LoraConfig(
            r=lora_cfg.r, lora_alpha=lora_cfg.alpha, lora_dropout=lora_cfg.dropout,
            target_modules=list(lora_cfg.target_modules), bias="none",
        )
        try:
            self.generation.lm = get_peft_model(self.generation.lm, peft_cfg)
        except Exception as exc:  # noqa: BLE001
            raise LoRAUnavailable(
                f"LoRA could not be applied to '{type(self.generation.lm).__name__}' "
                f"with target_modules={list(lora_cfg.target_modules)}: {exc}. "
                "Set lora.target_modules to this LM's attention projection names, "
                "or set lora.enabled: false."
            ) from exc
        self.lora_applied = True

    def forward(self, batch, device, generate: bool = False) -> dict:
        images = batch["images"].to(device)
        slice_mask = batch.get("slice_mask")
        if slice_mask is not None:
            slice_mask = slice_mask.to(device)
        patch_tokens, grid = self.vision.encode(images, slice_mask)
        g = self.grounding(patch_tokens, grid)
        pooled = self.grounding.pooled_from_regions(g["region_emb"], self.pooling)
        logits = self.classifier(pooled)

        out = dict(logits=logits, attn_maps=g["attn_maps"], region_emb=g["region_emb"])
        if generate:
            out["text"] = self.generation.generate(g["region_emb"], device)
        else:
            out["gen_loss"] = self.generation(g["region_emb"], batch["report"], device)
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
