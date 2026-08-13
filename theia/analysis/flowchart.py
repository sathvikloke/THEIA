"""CLAIM item 14: the participant flow diagram.

Drawn from `theia.analysis.cohort.flow()` rather than typed, so the boxes cannot
drift from the pipeline. Every exclusion carries its count and its reason; a flow
diagram whose arrows do not sum is the first thing a reviewer checks.

Run: python -m theia.analysis.flowchart
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw(stages: list[dict], out: str, title: str | None = None) -> None:
    n = len(stages)
    fig, ax = plt.subplots(figsize=(7.6, 1.62 * n), dpi=200)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.62 * n); ax.axis("off")

    box_w, box_h, x0 = 4.6, 0.86, 0.4
    for i, st in enumerate(stages):
        y = 1.62 * n - 1.2 - i * 1.62
        primary = st["stage"].startswith("PRIMARY")
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y), box_w, box_h, boxstyle="round,pad=0.06",
            linewidth=1.8 if primary else 1.1,
            edgecolor="#08519c" if primary else "0.25",
            facecolor="#deebf7" if primary else "white"))
        label = st["stage"].replace("PRIMARY ANALYSIS SET: ", "")
        ax.text(x0 + box_w / 2, y + box_h * 0.62, label,
                ha="center", va="center", fontsize=8.4,
                fontweight="bold" if primary else "normal", wrap=True)
        ax.text(x0 + box_w / 2, y + box_h * 0.22, f"n = {st['n']}",
                ha="center", va="center", fontsize=9.6, fontweight="bold",
                color="#08519c" if primary else "0.15")
        if primary:
            ax.text(x0 + box_w / 2, y - 0.16, "PRIMARY ANALYSIS SET",
                    ha="center", va="center", fontsize=6.6, color="#08519c",
                    fontweight="bold")

        if i + 1 < n:
            ax.annotate("", xy=(x0 + box_w / 2, y - 0.30),
                        xytext=(x0 + box_w / 2, y - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.2))
            nxt = stages[i + 1]
            if nxt.get("lost"):
                ex_y = y - 1.62 + box_h + 0.30
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0 + box_w + 0.55, ex_y), 4.2, 0.80,
                    boxstyle="round,pad=0.05", linewidth=0.9,
                    edgecolor="0.55", facecolor="#f7f7f7"))
                ax.annotate("", xy=(x0 + box_w + 0.52, ex_y + 0.40),
                            xytext=(x0 + box_w / 2, ex_y + 0.40),
                            arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.0))
                reason = nxt["reason"] or ""
                if len(reason) > 190:
                    reason = reason[:187] + "..."
                ax.text(x0 + box_w + 0.75, ex_y + 0.58, f"excluded n = {nxt['lost']}",
                        fontsize=7.6, fontweight="bold", color="0.25", va="center")
                ax.text(x0 + box_w + 0.75, ex_y + 0.26, _wrap(reason, 52),
                        fontsize=6.3, color="0.35", va="center", linespacing=1.35)

    if title:
        ax.set_title(title, fontsize=9.5, pad=10)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[flow] wrote {out}")


def _wrap(s: str, width: int) -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:4])


def main() -> None:
    from theia.analysis.cohort import flow
    from theia.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="figures/fig1_flow.png")
    a = ap.parse_args()
    cfg = load_config(a.config)
    stages = flow(cfg)
    draw(stages, a.out,
         "NSCLC-Radiogenomics — participant flow (CLAIM item 14)")
    total_lost = sum(s.get("lost") or 0 for s in stages)
    assert stages[0]["n"] - total_lost == stages[-1]["n"], "flow does not reconcile"
    print(f"[flow] reconciles: {stages[0]['n']} - {total_lost} = {stages[-1]['n']}")


if __name__ == "__main__":
    main()
