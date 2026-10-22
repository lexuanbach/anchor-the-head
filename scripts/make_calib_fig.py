"""Novel-class vs overall ECE across calibrators for ProtoMask-Set (REAL values, read directly
from tables/tab_real_calib.tex, 5-split N=347 protocol). Two-series design is mandatory:
the visible global-temperature OVERALL bar (3.5) prevents misreading the figure as a clean diffusion win.
Renders figures/calib_ece.png. matplotlib only."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["Uncalib.", "Global $T$", "Per-class $T$", "Vector/Platt", "Diffusion\n(ours)"]
novel = [12.8, 6.6, 27.9, 34.0, 5.2]
novel_sd = [3.0, 3.1, 3.5, 4.4, 1.7]
overall = [14.6, 3.5, 12.6, 16.2, 5.9]
overall_sd = [0.0, 0.6, 2.8, 0.7, 0.0]

x = np.arange(len(methods)); w = 0.38
fig, ax = plt.subplots(figsize=(5.0, 3.1))
b1 = ax.bar(x - w/2, novel, w, yerr=novel_sd, capsize=3, color="#c0392b", label="Novel-class ECE")
b2 = ax.bar(x + w/2, overall, w, yerr=overall_sd, capsize=3, color="#5dade2", label="Overall ECE")
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel("ECE (lower is better)", fontsize=9)
ax.set_ylim(0, 41)
ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.set_title("Only class-symmetric diffusion calibrates held-out classes", fontsize=9.5)
# annotation enforcing the two-sided story (global-T trades overall for novel).
# Keep it in the open left/top region so it does not collide with the tall
# per-class/vector bars.
ax.annotate("best overall (3.5)\nbut 6.6 on novel", xy=(1+w/2, 3.5), xytext=(0.9, 25),
            fontsize=7.5, ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
            arrowprops=dict(arrowstyle="->", lw=0.7, connectionstyle="arc3,rad=-0.22"))
for rect, v, sd in zip(b1, novel, novel_sd):
    ax.text(rect.get_x()+rect.get_width()/2, v+sd+0.7, f"{v:.1f}", ha="center", fontsize=7)
plt.tight_layout()
import os
for outdir in ("figures",):
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "calib_ece.png"), dpi=220, bbox_inches="tight")
print("wrote figures/calib_ece.png")
