import matplotlib.pyplot as plt

# Data
ranks = [8, 16, 24, 32]
cifar_iid = [81.25, 84.01, 84.81, 85.15]
cifar_n20 = [71.02, 77.64, 78.83, 78.99]
cifar_n10 = [55.4, 64.13, 68.99, 70.11]

# Plot

plt.figure(figsize=(10,8))   # square plot

plt.plot(ranks, cifar_iid, marker='o', linewidth=3, markersize=13, label="IID")
plt.plot(ranks, cifar_n20, marker='s', linewidth=3, markersize=13, label="Non-IID 20")
plt.plot(ranks, cifar_n10, marker='^', linewidth=3, markersize=13, label="Non-IID 10")

plt.xlabel("Rank", fontsize=30, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
#plt.title("Accuracy vs Rank for CIFAR Settings", fontsize=18, fontweight='bold')
plt.legend(fontsize=30)
plt.grid(True, linewidth=1.5)

plt.tight_layout()

plt.savefig("cifar_rank_plot.pdf", format="pdf", bbox_inches="tight")
plt.show()
