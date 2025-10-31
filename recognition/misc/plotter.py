import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data ---
epochs = list(range(1, 21))

dp_02  = [0.7478, 0.6394, 0.6279, 0.6221, 0.5821, 0.561, 0.5729, 0.5604, 0.5176, 0.5277, 0.5564, 0.5105, 0.4928, 0.5251, 0.5116, 0.4793, 0.4863, 0.5135, 0.483, 0.4602]
dp_015  = [0.7632,0.6573,0.6509,0.6437,0.6017,0.5745,0.5815,0.5747,0.5336,0.5351,0.564,0.5187,0.4974,0.5259,0.5127,0.4793,0.482,0.5061,0.4775,0.4551]
dp_01 = [0.7272, 0.6449, 0.6489, 0.6253, 0.5919, 0.5622, 0.5747, 0.5629, 0.5244, 0.523, 0.5448, 0.5153, 0.4906, 0.5218, 0.5137, 0.475, 0.4809, 0.5064, 0.4691, 0.4467]
dp_0075 = [0.7633, 0.6527, 0.6413, 0.6323, 0.5966, 0.5704, 0.5848, 0.5623, 0.5365, 0.5379, 0.5531, 0.5269, 0.5028, 0.524, 0.5132, 0.48, 0.4855, 0.5146, 0.4772, 0.4578]
dp_005 = [0.7182, 0.6233, 0.6212, 0.6203, 0.5837, 0.5686, 0.5817, 0.5643, 0.5237, 0.5278, 0.5544, 0.5227, 0.4953, 0.5193, 0.5276, 0.4772, 0.4884, 0.5127, 0.4792, 0.4587]


# --- 2. Style setup ---
sns.set_theme(style="darkgrid")     # dark grid aesthetic
plt.style.use("dark_background")    # full dark background
colors = sns.color_palette("mako", 6)  # smooth dark-cool gradient

# --- 3. Plot ---
plt.figure(figsize=(10,6))
plt.plot(epochs, dp_02,  label="0.2",  lw=2.5, color=colors[0])
plt.plot(epochs, dp_015,  label="0.15",  lw=2.5, color=colors[1])
plt.plot(epochs, dp_01, label="0.1",lw=2.5, color=colors[2])
plt.plot(epochs, dp_0075, label="0.075",lw=2.5, color=colors[3])
plt.plot(epochs, dp_005, label="0.005",lw=2.5, color=colors[4])


# --- 4. Decorate ---
plt.title("Test Loss vs Epochs for Varying DropPath Rates", fontsize=15, weight="bold", color="w", pad=15)
plt.xlabel("Epoch", fontsize=13, color="w")
plt.ylabel("Test Loss", fontsize=13, color="w")
plt.legend(title="Learning Rate", fontsize=10, facecolor="#222222", edgecolor="#444", labelcolor="white", title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Optional: subtle glow effect (looks amazing in dark mode)
for spine in plt.gca().spines.values():
    spine.set_color("#444")

plt.savefig("sexy_plot.png", dpi=300, bbox_inches="tight", facecolor="#111111")
plt.show()
