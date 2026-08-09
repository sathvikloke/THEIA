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

import math

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
        region_emb, attn_w = self._attend(q, patch_tokens)
        region_emb = self.norm(region_emb)                       # [B, Q, D]
        h, w = grid_hw
        maps = attn_w.view(b, -1, h, w)                          # [B, Q, H', W']
        return dict(region_emb=region_emb, attn_maps=maps)

    def _attend(self, q: torch.Tensor, kv: torch.Tensor):
        """Multi-head attention computed explicitly, using self.attn's parameters.

        Mathematically identical to
            self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        and it deliberately keeps the nn.MultiheadAttention submodule so the
        parameter names — and therefore every existing checkpoint — are unchanged.

        Written out because this head needs the attention WEIGHTS, not just the
        output. Asking nn.MultiheadAttention for them forces its non-fused math
        path, and that path's backward is the leading suspect for the recurring
        NaN: 141 parameters, all in the unfrozen ViT blocks feeding this
        attention, go non-finite in the backward while every loss in the forward
        stays finite. Computing it here keeps the graph explicit and the softmax
        under our control.

        The softmax is taken in float32 regardless of the incoming dtype: it is
        the one step where a large logit spread turns into inf/NaN, and under
        autocast it would otherwise run in a narrower type.
        """
        b, n_q, d = q.shape
        n_k = kv.shape[1]
        heads = self.attn.num_heads
        dh = d // heads

        qw, kw, vw = self.attn.in_proj_weight.chunk(3, dim=0)
        qb, kb, vb = self.attn.in_proj_bias.chunk(3, dim=0)
        Q = F.linear(q, qw, qb).view(b, n_q, heads, dh).transpose(1, 2)
        K = F.linear(kv, kw, kb).view(b, n_k, heads, dh).transpose(1, 2)
        V = F.linear(kv, vw, vb).view(b, n_k, heads, dh).transpose(1, 2)

        logits = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)       # [B,H,Q,K]
        attn = torch.softmax(logits.float(), dim=-1).to(V.dtype)
        out = (attn @ V).transpose(1, 2).reshape(b, n_q, d)
        out = self.attn.out_proj(out)
        return out, attn.mean(dim=1)                             # average over heads

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
        # Whatever this tokenizer actually prepends to a sequence — read it off
        # rather than assuming bos_token_id. For BioGPT the two disagree:
        # bos_token is <s> (0) but the tokenizer prepends </s> (2). Guessing
        # wrong here puts generation in a state training never saw.
        probe = self.tokenizer("x", return_tensors="pt").input_ids[0]
        self.seq_start_id = int(probe[0])
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
        """Decode a rationale from the visual prefix.

        The sequence-start token is appended to the prefix, and it is not
        optional. BioGPT's tokenizer prepends `</s>` to every sequence, and
        `</s>` is ALSO its eos_token. Training therefore sees

            [visual prefix] + [</s>] + rationale tokens

        so at inference the model's first prediction after the prefix is exactly
        `</s>` — which is correct behaviour, and which `generate()` reads as
        "stop". The result was a single EOS token and an empty string for every
        patient. Nothing raised: the reader-study builder happily wrote 31
        blank rationales, and the arm that exists to be judged on its
        explanation had no explanation in it.

        Feeding the start token explicitly puts inference in the same state
        training left the model in, so decoding continues into real text.
        """
        prefix = self._prefix(region_emb)
        b = prefix.shape[0]
        start = torch.full((b, 1), self.seq_start_id, device=device, dtype=torch.long)
        start_emb = self.lm.get_input_embeddings()(start)
        inputs_emb = torch.cat([prefix, start_emb], dim=1)
        attn = torch.ones(inputs_emb.shape[:2], device=device)
        out = self.lm.generate(
            inputs_embeds=inputs_emb,
            attention_mask=attn,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=8,          # a one-token reply is not a rationale
            do_sample=False,
            # Greedy decoding on a short, templated corpus loops: measured
            # "attachment to pleura, attachment to pleura, attachment to
            # pleura". Readers would mark that down for a decoding artefact
            # rather than for anything the model got wrong about the image.
            no_repeat_ngram_size=4,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return [t.strip() for t in
                self.tokenizer.batch_decode(out, skip_special_tokens=True)]
