import re
import csv

log_path = "/home/hice1/xkong43/scratch/outputs/logs/sd_fullft-4501168.err"
csv_path = "/home/hice1/xkong43/scratch/outputs/full_ft_sd15/loss_curve.csv"

pattern_step = re.compile(r"(\d+)/(\d+)")
pattern_loss = re.compile(r"step_loss=([0-9.eE+-]+)")

rows = []

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "step_loss=" not in line:
            continue

        step_match = pattern_step.search(line)
        loss_match = pattern_loss.search(line)

        if step_match and loss_match:
            step = int(step_match.group(1))
            total_steps = int(step_match.group(2))
            loss = float(loss_match.group(1))
            rows.append((step, total_steps, loss))

# remove duplicate steps, keep latest occurrence
dedup = {}
for step, total_steps, loss in rows:
    dedup[step] = (step, total_steps, loss)

final_rows = [dedup[k] for k in sorted(dedup.keys())]

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "total_steps", "loss"])
    writer.writerows(final_rows)

print(f"Saved {len(final_rows)} rows to {csv_path}")
