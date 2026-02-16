import os
import re
from pathlib import Path
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")

iid = Path("results/example_log/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int_2025-12-07_00-45-38/exp_log.txt").read_text(encoding="utf-8", errors="ignore")
noniid10 = Path("results/example_log/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-noniid-10_2025-12-07_04-21-46/exp_log.txt").read_text(encoding="utf-8", errors="ignore")
noniid20 = Path("results/example_log/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-noniid-20_2025-12-07_07-54-51/exp_log.txt").read_text(encoding="utf-8", errors="ignore")

def extract_round_accuracy(text: str, max_round=200):
    pattern = re.compile(
        r"Round:\s*(\d+)\s*/\s*\d+.*?\{\s*'accuracy'\s*:\s*([0-9]*\.?[0-9]+)\s*\}",
        flags=re.DOTALL
    )
    pairs = [(int(r), float(a)) for r, a in pattern.findall(text)]
    pairs = [(r, a) for r, a in pairs if r <= max_round]
    pairs.sort(key=lambda x: x[0])
    return pairs

def unpack(pairs):
    xs = [r for r, _ in pairs]
    ys = [a * 100 for _, a in pairs]  # convert to %
    return xs, ys

cifar_iid_pairs = extract_round_accuracy(iid)
cifar_n10_pairs = extract_round_accuracy(noniid10)
cifar_n20_pairs = extract_round_accuracy(noniid20)

x_iid, y_iid = unpack(cifar_iid_pairs)
x_n10, y_n10 = unpack(cifar_n10_pairs)
x_n20, y_n20 = unpack(cifar_n20_pairs)

plt.figure(figsize=(10, 8))  # square plot

plt.plot(x_iid, y_iid, marker='o', linewidth=3, markersize=8, label="IID")
plt.plot(x_n20, y_n20, marker='s', linewidth=3, markersize=8, label="Non-IID 20")
plt.plot(x_n10, y_n10, marker='^', linewidth=3, markersize=8, label="Non-IID 10")

plt.xlabel("Round of Training", fontsize=30, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold',rotation=25)
plt.yticks(fontsize=30, fontweight='bold')
plt.legend(fontsize=30)
plt.grid(True, linewidth=1.5)

plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, "cifar_train_plot.pdf"), format="pdf", bbox_inches="tight")
plt.show()