import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ["no-rank selection", "no-alternating", "no-svd"]

accuracy_diff = np.array([3.02, 4.52, 1.12])
micro_f1_diff = np.array([0.0048, 0.0173, 0.0182])

x = np.arange(len(labels))
bar_width = 0.35

plt.rcParams.update({
    'font.size': 16,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

# Square figure
fig, ax1 = plt.subplots(figsize=(10,5))

# Left Y-axis (Accuracy — LIGHT BLUE)
light_blue = "#7DBEFF"

bars1 = ax1.bar(
    x - bar_width/2,
    accuracy_diff,
    width=bar_width,
    color=light_blue,
    label="Accuracy Diff (%)"
)

ax1.set_ylabel("Accuracy Diff (%)",
               fontsize=20, fontweight='bold', color=light_blue)
ax1.tick_params(axis='y', labelsize=16, width=2, colors=light_blue)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=16, fontweight='bold', rotation=15)

# Right Y-axis (Micro F1 — RED)
ax2 = ax1.twinx()  # <--- ONLY ONE twin axis

bars2 = ax2.bar(
    x + bar_width/2,
    micro_f1_diff,
    width=bar_width,
    color='red',
    alpha=0.85,
    label="Micro F1 Diff"
)

ax2.set_ylabel("Micro F1 Diff",
                fontsize=20, fontweight='bold', color='red')
ax2.tick_params(axis='y', labelsize=16, width=2, colors='red')

# Title & Grid
#plt.title("Ablation Study — Accuracy vs. Micro F1 Difference",
#          fontsize=20, fontweight='bold')

ax1.grid(axis='y', linewidth=1.3)

plt.tight_layout()

# Save as scalable PDF
plt.savefig("ablation_dual_axis.pdf", format="pdf", bbox_inches="tight")

plt.show()