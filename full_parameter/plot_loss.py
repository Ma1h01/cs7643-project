import csv
import matplotlib.pyplot as plt

csv_path = "/home/hice1/xkong43/scratch/outputs/full_ft_sd15/loss_curve.csv"
png_path = "/home/hice1/xkong43/scratch/outputs/full_ft_sd15/loss_curve.png"
png_smooth_path = "/home/hice1/xkong43/scratch/outputs/full_ft_sd15/loss_curve_smoothed.png"

steps = []
losses = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        losses.append(float(row["loss"]))

plt.figure(figsize=(8, 5))
plt.plot(steps, losses)
plt.xlabel("Training step")
plt.ylabel("Step loss")
plt.title("Full fine-tuning training loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(png_path, dpi=200)

window = 50
smoothed = []
for i in range(len(losses)):
    left = max(0, i - window + 1)
    smoothed.append(sum(losses[left:i+1]) / (i - left + 1))

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, alpha=0.25, label="raw loss")
plt.plot(steps, smoothed, linewidth=2, label=f"moving average ({window})")
plt.xlabel("Training step")
plt.ylabel("Step loss")
plt.title("Full fine-tuning training loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(png_smooth_path, dpi=200)

print(f"Saved {png_path}")
print(f"Saved {png_smooth_path}")
