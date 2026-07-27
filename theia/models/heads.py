"""The three heads.

ClassifierHead  : pooled visual feature -> per-gene logits.
GroundingHead   : learned region queries cross-attend to patch tokens, producing
                  (a) attention maps over the patch grid, one per query, and
                  (b) grounded region embeddings the generator can consume.
GenerationHead  : a causal LM conditioned on a visual prefix (the region embeddings)
                  that decodes the rationale text.

The grounding attention maps are THE differentiator vs Glio-LLaMA-Vision: they let
each generated claim point back at pixels, and they are supervised against the
tumor ROI mask so "where the model looks" is anchored to real anatomy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    def __init__(self, dim: int, genes: list[str], hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.genes = genes
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({g.lower(): nn.Linear(hidden, 2) for g in genes})

    def forward(self, pooled: torch.Tensor) -> dict:
        h = self.trunk(pooled)
        return {g: head(h) for g, head in self.heads.items()}


class GroundingHead(nn.Module):
    """Region queries attend over patch tokens -> attention maps + region embeddings."""

    def __init__(self, dim: int, n_queries: int = 8, n_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens: torch.Tensor, grid_hw: tuple[int, int]) -> dict:
        b = patch_tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)          # [B, Q, D]
        region_emb, attn_w = self.attn(q, patch_tokens, patch_tokens,
                                       need_weights=True, average_attn_weights=True)
        region_emb = self.norm(region_emb)                       # [B, Q, D]
        h, w = grid_hw
        maps = attn_w.view(b, -1, h, w)                          # [B, Q, H', W']
        return dict(region_emb=region_emb, attn_maps=maps)

    @staticmethod
    def pooled_from_regions(region_emb: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Pool the Q region embeddings into one vector for the classifier.

        "mean" is the original behaviour. "max" matches what the grounding loss
        supervises (an element-wise max over queries), which is the only setting
        under which the "cannot decide from one place and point at another"
        claim actually holds.
        """
        if mode == "max":
            return region_emb.amax(dim=1)                        # [B, D]
        if mode == "mean":
            return region_emb.mean(dim=1)                        # [B, D]
        raise ValueError(f"unknown region_pooling: {mode!r} (expected 'mean' or 'max')")


class GenerationHead(nn.Module):
    """Causal LM decoder with a projected visual prefix.

    The region embeddings are projected into the LM embedding space and prepended
    as soft prompt tokens; the LM then decodes the rationale. LoRA is applied to
    the LM in theia_model, not here.
    """

    def __init__(self, dim: int, lm_name: str, max_new_tokens: int = 96):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if self.tokenizer.pad_token is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback is None:
                raise ValueError(
                    f"{lm_name} has no pad/eos/unk token; cannot batch the generation head."
                )
            self.tokenizer.pad_token = fallback
        # Labels mask the padding, so right-padding is correct for the teacher-forced
        # forward; generate() runs off inputs_embeds and never sees a pad.
        self.tokenizer.padding_side = "right"
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.max_new_tokens = max_new_tokens
        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.visual_proj = nn.Linear(dim, lm_dim)

    def _prefix(self, region_emb: torch.Tensor) -> torch.Tensor:
        return self.visual_proj(region_emb)                      # [B, Q, lm_dim]

    def forward(self, region_emb: torch.Tensor, reports: list[str], device) -> torch.Tensor:
        prefix = self._prefix(region_emb)
        tok = self.tokenizer(reports, return_tensors="pt", padding=True,
                             truncation=True, max_length=128).to(device)
        text_emb = self.lm.get_input_embeddings()(tok.input_ids)
        inputs_emb = torch.cat([prefix, text_emb], dim=1)
        q = prefix.shape[1]
        prefix_mask = torch.ones(prefix.shape[:2], device=device, dtype=tok.attention_mask.dtype)
        attn_mask = torch.cat([prefix_mask, tok.attention_mask], dim=1)
        # visual-prefix positions are masked out of the loss with -100
        ignore = torch.full((prefix.shape[0], q), -100, device=device, dtype=torch.long)
        labels = torch.cat([ignore, tok.input_ids.masked_fill(tok.attention_mask == 0, -100)], dim=1)
        out = self.lm(inputs_embeds=inputs_emb, attention_mask=attn_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, region_emb: torch.Tensor, device) -> list[str]:
        prefix = self._prefix(region_emb)
        attn = torch.ones(prefix.shape[:2], device=device)
        out = self.lm.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)
