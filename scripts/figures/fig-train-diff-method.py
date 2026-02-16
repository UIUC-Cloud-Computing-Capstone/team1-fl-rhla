import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths relative to project root (script at scripts/figures/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG = _PROJECT_ROOT / "log"

def _read_log(rel_path: str) -> str:
    return (_LOG / rel_path).read_text(encoding="utf-8", errors="ignore")

_BASE = "cifar100/facebook/deit-small-patch16-224/ffm_fedavg/experiments/cifar100_vit_lora"

# non-iid 10
# FEDIT = _read_log(f"{_BASE}/FedIT/FedIT-noniid-pat_10_dir_2025-11-28_15-04-28/exp_log.txt")
# FLEXLORA  = _read_log(f"log/{_BASE}/FlexLoRA/FlexLoRA-noniid-pat_10_dir_2025-12-17_10-33-08/exp_log.txt")
# HETLORA = _read_log(f"{_BASE}/HetLoRA/HetLoRA-noniid-pat_10_dir_2025-12-16_05-10-20/exp_log.txt")
# LEGEND = _read_log(f"{_BASE}/LEGEND/LEGEND-noniid-10_2025-11-28_22-12-22/exp_log.txt")
# FEDHELLO = _read_log(f"{_BASE}/FedHello/FedHello_noniid-pat_10_dir-noprior-s50-e50_2025-11-28_11-29-31/exp_log.txt")
# HRALORA = _read_log(f"{_BASE}/Ours/alternating-training-warm20-double-rank-int-noniid-10_2025-12-07_04-21-46/exp_log.txt")

# iid
# FEDIT = _read_log(f"{_BASE}/FedIT/FedIT-iid_2025-11-26_11-29-11/exp_log.txt")
# FLEXLORA  = _read_log(f"log/{_BASE}/FlexLoRA/FlexLoRA-iid_2025-12-17_08-42-35/exp_log.txt")
# HETLORA = _read_log(f"{_BASE}/HetLoRA/HetLoRA-iid_2025-12-16_01-32-48/exp_log.txt")
# LEGEND = _read_log(f"{_BASE}/LEGEND/LEGEND-iid_2025-11-26_18-51-24/exp_log.txt")
# FEDHELLO = _read_log(f"{_BASE}/FedHello/FedHello_rank-24iid-noprior-s50-e50_2025-11-26_07-52-20/exp_log.txt")
# HRALORA = _read_log(f"{_BASE}/Ours/alternating-training-warm20-double-rank-int-2025-12-07_00-45-38/exp_log.txt")

# non-iid 20
FEDIT = _read_log(f"{_BASE}/FedIT/FedIT-noniid-pat_20_dir_2025-11-25_13-32-44/exp_log.txt")
FLEXLORA  = _read_log(f"{_BASE}/FlexLoRA/FlexLoRA-noniid-pat_20_dir_2025-12-17_20-03-54/exp_log.txt")  # under log/log/
HETLORA = _read_log(f"{_BASE}/HetLoRA/HetLoRA-noniid-pat_20_dir_2025-12-17_00-26-22/exp_log.txt")
LEGEND = _read_log(f"{_BASE}/LEGEND/LEGEND-noniid-20_2025-11-25_20-38-24/exp_log.txt")
FEDHELLO = _read_log(f"{_BASE}/FedHello/FedHello_noniid-pat_20_dir-noprior-s50-e50_2025-11-25_10-00-05/exp_log.txt")
HRALORA = _read_log(f"{_BASE}/Ours/alternating-training-warm20-double-rank-int-noniid-20_2025-12-07_07-54-51/exp_log.txt")



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

def unpack(pairs, length=200 ,name = 'None'):
    scale = 20
    if name== 'FedIT':
        scale = 20.57
    elif name == 'FlexLoRA':
        scale = 22.315
    elif name == 'HetLoRA':
        scale = 21.323
    elif name == 'LEGEND':
        scale = 19.797
    elif name == 'Fed-HeLLo':
        scale = 20.155
    elif name == 'HRALoRA':
        scale = 20.56
    xs = [r*scale/60 for r, _ in pairs][:length]
    ys = [a * 100 for _, a in pairs][:length]  # convert to %
    ys = moving_average_variable(ys, window=5)
    return xs, ys

# norankselection_pairs = extract_round_accuracy(norankselection)
# cifar_n10_pairs = extract_round_accuracy(noniid10)
# nosvd_pairs = extract_round_accuracy(nosvd)
# noalternating_pairs = extract_round_accuracy(noalternating)

# x_1, y_1 = unpack(norankselection_pairs)
# x_2, y_2 = unpack(cifar_n10_pairs)
# x_3, y_3 = unpack(nosvd_pairs)
# x_4, y_4 = unpack(noalternating_pairs)

plt.figure(figsize=(10, 8))  # square plot

pairs_by_method = {
    "FedIT": extract_round_accuracy(FEDIT),
    "FlexLoRA": extract_round_accuracy(FLEXLORA),
    "HetLoRA": extract_round_accuracy(HETLORA),
    "LEGEND": extract_round_accuracy(LEGEND),
    "Fed-HeLLo": extract_round_accuracy(FEDHELLO),
    "HRALoRA": extract_round_accuracy(HRALORA),
}

# --- Plot ---
plt.figure(figsize=(10, 8))

for name, pairs in pairs_by_method.items():
    x, y = unpack(pairs,length = 200, name = name)
    plt.plot(x, y, linewidth=5, label=name)  # no markers

# plt.plot(x_1, y_1,  linewidth=5, markersize=3, label="No Rank Alloc")
# plt.plot(x_2, y_2,  linewidth=5, markersize=3, label="HRALoRA")
# plt.plot(x_3, y_3,  linewidth=5, markersize=3, label="No SVD")
# plt.plot(x_4, y_4,  linewidth=5, markersize=3, label="Non Alternating")

plt.xlabel("Training time (min)", fontsize=30, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.legend(fontsize=30)
plt.xlim(-2,70)
plt.grid(True, linewidth=1.5)

plt.tight_layout()
plt.savefig("cifar_train_plot-diff-method.pdf", format="pdf", bbox_inches="tight")
plt.show()