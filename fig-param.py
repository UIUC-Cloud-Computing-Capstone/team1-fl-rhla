import matplotlib.pyplot as plt
import numpy as np

# ===== Data =====
methods = [
    "FedLR", "Straggler", "Exclusive", "LoKr",
    "FFA-LoRA", "LEGEND", "Fed-HeLLo", "HRALoRA"
]

accuracy = np.array([
    84.01, 82.24, 82.11, 66.90,
    77.56, 84.50, 84.67, 84.81
])

num_params = np.array([
    6_856_704, 4_423_680, 3_538_944, 299_520,
    3_428_352, 6_718_464, 6_856_704, 3_428_352, 6_856_704
])

# Marker shapes (9 unique)
markers = ["X", "X", "X", "X", "X", "X", "X", "X", "X"]

# 9 distinct colors
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"
]

# ===== Style =====
plt.rcParams.update({
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, ax = plt.subplots(figsize=(10, 8))

# Offsets for labels
dx = 80_000    # x offset for side labels
dy_up = 0.5    # y offset upward
dy_down = 0.9  # y offset downward

# ===== Plot points and labels =====
for i, name in enumerate(methods):
    x = num_params[i]
    y = accuracy[i]

    # Plot marker
    ax.scatter(
        x, y,
        marker=markers[i],
        s=320,                 # size
        color=colors[i],
        edgecolors="black",
        linewidths=3           # thick outline
    )

    # Custom label placement by method
    if name == "FedLR":
        # Under the marker, centered
        ax.text(
            x,
            y - dy_down,
            name,
            fontsize=20,
            fontweight="bold",
            ha="center",
            va="top"
        )
    elif name == "Straggler":
        # Down-right
        ax.text(
            x + dx,
            y - dy_down,
            name,
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="top"
        )
    elif name == "LEGEND" :
        # Top-left
        ax.text(
            x + dx,
            y + dy_up,
            name,
            fontsize=20,
            fontweight="bold",
            ha="right",
            va="bottom"
        )
    else:
        # Default: top-right
        ax.text(
            x + dx,
            y + dy_up,
            name,
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

# ===== Axes, limits, grid =====
ax.set_xlabel("Number of Trainable Parameters", fontsize=30, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=30, fontweight="bold")
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')

# Show x-axis up to 8e6
ax.set_xlim(0, 9_000_000)
ax.set_ylim(65, 87)
# Thick axis spines
for spine in ax.spines.values():
    spine.set_linewidth(3)


# Thick ticks
ax.tick_params(axis="both", width=3)

ax.grid(True, linewidth=1.5, alpha=0.6)

ax = plt.gca()
offset = ax.xaxis.get_offset_text()

offset.set_fontsize(14)          # increase size if needed
offset.set_x(1.08)                # shift horizontally (default often ~0)
offset.set_y(10)              # shift downward or upward


plt.tight_layout()
plt.savefig("accuracy_vs_params_custom_labels.pdf", format="pdf", bbox_inches="tight")
plt.show()
