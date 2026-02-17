import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")

norankselection = Path("results/example_log/ablation-non-iid-10/alternating-training-warm20-double-rank-int-rank24-norankselection_2025-11-24_13-16-54/exp_log.txt").read_text(encoding="utf-8", errors="ignore")
noniid10 = Path("results/example_log/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-noniid-10_2025-12-07_04-21-46/exp_log.txt").read_text(encoding="utf-8", errors="ignore")
nosvd = Path("results/example_log/ablation-non-iid-10/alternating-training-warm20-double-rank-int-rank24-nosvd_2025-11-24_06-13-19/exp_log.txt").read_text(encoding="utf-8", errors="ignore")
noalternating = Path("results/example_log/ablation-non-iid-10/alternating-training-warm20-double-rank-int-rank24-noalternating_2025-11-24_09-44-22/exp_log.txt").read_text(encoding="utf-8", errors="ignore")


def extract_round_accuracy(text: str, max_round=200):
    pattern = re.compile(
        r"Round:\s*(\d+)\s*/\s*\d+.*?\{\s*'accuracy'\s*:\s*([0-9]*\.?[0-9]+)\s*\}",
        flags=re.DOTALL
    )
    pairs = [(int(r), float(a)) for r, a in pattern.findall(text)]
    pairs = [(r, a) for r, a in pairs if r <= max_round]
    pairs.sort(key=lambda x: x[0])
    return pairs

def moving_average_variable(y, window=5):
    y = np.asarray(y, dtype=float)
    n = len(y)
    half = window // 2
    out = np.empty(n, dtype=float)

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        out[i] = y[left:right].mean()

    return out

def unpack(pairs):
    xs = [r*20.56/60 for r, _ in pairs]
    ys = [a * 100 for _, a in pairs]  # convert to %
    ys = moving_average_variable(ys, window=5)
    return xs, ys

norankselection_pairs = extract_round_accuracy(norankselection)
cifar_n10_pairs = extract_round_accuracy(noniid10)
nosvd_pairs = extract_round_accuracy(nosvd)
noalternating_pairs = extract_round_accuracy(noalternating)

x_1, y_1 = unpack(norankselection_pairs)
x_2, y_2 = unpack(cifar_n10_pairs)
x_3, y_3 = unpack(nosvd_pairs)
x_4, y_4 = unpack(noalternating_pairs)

plt.figure(figsize=(10, 8))  # square plot

plt.plot(x_2, y_2,  linewidth=5, markersize=3, label="HRALoRA")
plt.plot(x_1, y_1,  linewidth=5, markersize=3, label="No Rank Alloc")
plt.plot(x_3, y_3,  linewidth=5, markersize=3, label="No SVD")
plt.plot(x_4, y_4,  linewidth=5, markersize=3, label="Non Alternating")

plt.xlabel("Training time (min)", fontsize=30, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.legend(fontsize=30)
plt.xlim(-2,70)
plt.grid(True, linewidth=1.5)

plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, "cifar_train_plot.pdf"), format="pdf", bbox_inches="tight")
plt.show()