import matplotlib.pyplot as plt
import numpy as np

# Methods
methods = [
    "FedLR", "Straggler", "Exclusive", 
    "FFA-LoRA", "LEGEND", "Fed-HeLLo", "HRALoRA"
]

# Ratio data (3 rows × 8 cols)
ratio = np.array([
    [0.00001225224248, 0.00001859085648, 0.00002320183648, 
     0.00002262311455, 0.00001257727957, 0.00001234849864, 0.00002473783322],

    [0.00001083465175, 0.00001577193649, 0.00001780192057, 
     0.00001660856295, 0.00001144606863, 0.00001115550562, 0.00002299355492],

    [0.000008705348809, 0.00001287841797, 0.00001411155418, 
     0.00001154490554, 0.000009954656302, 0.000009170586918, 0.00002012337123]
])

x = np.arange(len(methods))  # 0..7

# Width of each bar
bar_width = 0.25

# Colors for IID, non-IID 20, non-IID 10
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, ax = plt.subplots(figsize=(14, 8))

# IID bars
ax.bar(x - bar_width, ratio[0], width=bar_width, 
       color=colors[0], edgecolor="black", linewidth=2, label="IID")

# non-IID 20 bars
ax.bar(x, ratio[1], width=bar_width, 
       color=colors[1], edgecolor="black", linewidth=2, label="Non-IID 20")

# non-IID 10 bars
ax.bar(x + bar_width, ratio[2], width=bar_width, 
       color=colors[2], edgecolor="black", linewidth=2, label="Non-IID 10")

# X-axis labels (method names)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=30, fontweight="bold")

# Axis labels
ax.set_ylabel("Accuracy / Parameter Ratio", fontsize=30, fontweight="bold")
#ax.set_xlabel("Method", fontsize=20, fontweight="bold")
plt.yticks(fontsize=30, fontweight='bold')
# Thick borders
for spine in ax.spines.values():
    spine.set_linewidth(3)

ax.tick_params(width=3)

# Grid
ax.grid(axis="y", linewidth=1.3, alpha=0.6)

# Legend on upper left
ax.legend(fontsize=25, frameon=False, loc="upper left")

ax = plt.gca()  # get current axis
ax.yaxis.get_offset_text().set_fontsize(20)   # <-- change 16 to what you want

plt.tight_layout()
plt.savefig("ratio_grouped_histogram.pdf", format="pdf", bbox_inches="tight")
plt.show()
