import re

log_path = "/home/youye/team1-fl-rhla/log/cifar100/google/vit-base-patch16-224-in21k/ffm_fedavg/experiments/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-no-rank-vary_2025-12-07_00-45-38/exp_log.txt"

with open(log_path, "r", encoding="utf-8") as f:
    log_text = f.read()

# Regex patterns
round_re = re.compile(r"Round:\s+(\d+)/")
layer_re = re.compile(
    r"fim score:\s*\[([^\]]*)\]\s*,\s*rank list:\s*\[([^\]]*)\]"
)

fim_by_round = {}     # round -> fim list
rank_by_round = {}    # round -> rank list

current_round = None
seen_in_round = False

for line in log_text.splitlines():

    # Detect round
    m_round = round_re.search(line)
    if m_round:
        current_round = int(m_round.group(1))
        seen_in_round = False
        continue

    # Detect fim score + rank list
    m_layer = layer_re.search(line)
    if (
        m_layer 
        and current_round is not None
        and current_round >= 20              # <-- SKIP ROUND 0 and 1
        and not seen_in_round               # <-- only once per round
    ):
        fim_str, rank_str = m_layer.groups()

        fim_scores = [float(x.strip()) for x in fim_str.split(",") if x.strip()]
        rank_list  = [int(x.strip())   for x in rank_str.split(",") if x.strip()]

        fim_by_round[current_round]  = fim_scores
        rank_by_round[current_round] = rank_list

        seen_in_round = True  # do not store repeated lines for same round

# Convert to 2D list (round order)
rounds = sorted(fim_by_round.keys())
fim_2d  = [fim_by_round[r]  for r in rounds]
rank_2d = [rank_by_round[r] for r in rounds]

print("Rounds included:", rounds)
print("FIM shape:",  len(fim_2d), "x", len(fim_2d[0]))
print("RANK shape:", len(rank_2d), "x", len(rank_2d[0]))

import numpy as np
print(f'mean fim score {np.mean(fim_2d,axis=0)}')
print(f'std fim score {np.std(fim_2d,axis=0)}')


print(f'mean rank {np.mean(rank_2d,axis=0)}')
print(f'std rank {np.std(rank_2d,axis=0)}')

#%% show layer as training goes

import matplotlib.pyplot as plt
import numpy as np

# fim_2d should be a list of lists:
# shape = [num_rounds][num_layers]
# convert to numpy for convenience
# fim = np.array(fim_2d)

# num_rounds, num_layers = fim.shape

# plt.figure(figsize=(12, 7))

# #Plot each layer (each column)
# for layer in range(num_layers):
#     plt.plot(
#         range(num_rounds),
#         fim[:, layer],
#         marker='o',
#         linewidth=2,
#         markersize=6,
#         label=f"Layer {layer}"
#     )

# plt.xlabel("Round", fontsize=16, fontweight="bold")
# plt.ylabel("FIM Score", fontsize=16, fontweight="bold")
# plt.title("FIM Evolution per Layer", fontsize=18, fontweight="bold")

# plt.grid(True, linestyle='--', alpha=0.4)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)

# plt.tight_layout()
# plt.show()

#%% show fim score for layer1 and 7
# rounds
# layer_1 = [row[1] for row in fim_2d]
# layer_7 = [row[7] for row in fim_2d]
# layer_5 = [row[5] for row in fim_2d]
# # Plot

# layer_1.pop(-1)
# layer_7.pop(-1)
# layer_5.pop(-1)
# rounds.pop(-1)
# plt.figure(figsize=(12,6))   # square plot

# plt.plot(rounds, layer_1, marker='o', linewidth=3, markersize=13, label="Layer-1")
# plt.plot(rounds, layer_5, marker='s', linewidth=3, markersize=13, label="Layer-5")
# plt.plot(rounds, layer_7, marker='s', linewidth=3, markersize=13, label="Layer-7")

# plt.xlabel("Round of Training", fontsize=30, fontweight='bold')
# plt.ylabel("FIM Score", fontsize=30, fontweight='bold')
# plt.xticks(fontsize=30, fontweight='bold')
# plt.yticks(fontsize=30, fontweight='bold')
# #plt.title("Accuracy vs Rank for CIFAR Settings", fontsize=18, fontweight='bold')
# plt.legend(fontsize=30)
# plt.grid(True, linewidth=1.5)

# plt.tight_layout()

# plt.savefig("layer-fim.pdf", format="pdf", bbox_inches="tight")
# plt.show()

#%% layer rank

rounds
layer_1 = [row[1] for row in rank_2d]
layer_7 = [row[7] for row in rank_2d]
layer_5 = [row[5] for row in rank_2d]
# Plot

layer_1.pop(-1)
layer_7.pop(-1)
layer_5.pop(-1)
rounds.pop(-1)
plt.figure(figsize=(12,6))   # square plot

plt.plot(rounds, layer_1, marker='o', linewidth=3, markersize=13, label="Layer-1")
plt.plot(rounds, layer_5, marker='s', linewidth=3, markersize=13, label="Layer-5")
plt.plot(rounds, layer_7, marker='s', linewidth=3, markersize=13, label="Layer-7")

plt.xlabel("Round of Training", fontsize=30, fontweight='bold')
plt.ylabel("Allocated Rank", fontsize=30, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
#plt.title("Accuracy vs Rank for CIFAR Settings", fontsize=18, fontweight='bold')
plt.legend(fontsize=30)
plt.grid(True, linewidth=1.5)

plt.tight_layout()

plt.savefig("layer-rank.pdf", format="pdf", bbox_inches="tight")
plt.show()

# #%% fim and rank bar
# import matplotlib.pyplot as plt
# import numpy as np

# # --- Compute mean/std ---
# fim_mean = np.mean(fim_2d, axis=0)
# fim_std  = np.std(fim_2d, axis=0)

# rank_mean = np.mean(rank_2d, axis=0)
# rank_std  = np.std(rank_2d, axis=0)

# num_layers = len(fim_mean)
# layers = np.arange(num_layers)

# # --------------------------------------------------------------------
# # 🔵 1) FIM Mean ± Std Plot
# # --------------------------------------------------------------------
# plt.figure(figsize=(12, 6))

# plt.errorbar(
#     layers,
#     fim_mean,
#     yerr=fim_std,
#     fmt='o',
#     capsize=6,
#     elinewidth=2,
#     markeredgewidth=2,
#     markersize=10,
#     color='blue',
#     ecolor='black',
#     label='Mean ± Std'
# )

# plt.plot(
#     layers,
#     fim_mean,
#     color='blue',
#     linewidth=3
# )

# plt.xticks(layers, [f"{l}" for l in layers], fontsize=30)
# plt.yticks(fontsize=30)
# plt.xlabel("Layer", fontsize=30, fontweight="bold")
# plt.ylabel("FIM Score", fontsize=30, fontweight="bold")
# #plt.title("FIM Mean and Std per Layer", fontsize=18, fontweight="bold")


# plt.grid(True, linestyle="--", alpha=0.3)
# plt.tight_layout()


# plt.savefig("fim_mean_std.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# # --------------------------------------------------------------------
# # 🔵 2) Rank Mean ± Std Plot
# # --------------------------------------------------------------------
# plt.figure(figsize=(12, 6))

# plt.errorbar(
#     layers,
#     rank_mean,
#     yerr=rank_std,
#     fmt='o',
#     capsize=6,
#     elinewidth=2,
#     markeredgewidth=2,
#     markersize=10,
#     color='red',
#     ecolor='black',
#     label='Mean ± Std'
# )

# plt.plot(
#     layers,
#     rank_mean,
#     color='red',
#     linewidth=3
# )


# plt.xticks(layers, [f"{l}" for l in layers], fontsize=30)
# plt.yticks(fontsize=30)
# plt.xlabel("Layer", fontsize=30, fontweight="bold")
# plt.ylabel("Allocated Rank", fontsize=30, fontweight="bold")
# #plt.title("Rank Mean and Std per Layer", fontsize=18, fontweight="bold")

# plt.grid(True, linestyle="--", alpha=0.3)
# plt.tight_layout()


# plt.savefig("rank_mean_std.pdf", format="pdf", bbox_inches="tight")
# plt.show()
