import os
import csv
from collections import defaultdict
from datetime import datetime

experiments_dir = "experiments"

# {step: {tokenizer: value}}
data = defaultdict(dict)

# Iterate through experiments
for experiment in os.listdir(experiments_dir):
    experiment_path = os.path.join(experiments_dir, experiment)
    if not os.path.isdir(experiment_path) or experiment == "testing":
        continue

    ratios_path = os.path.join(experiment_path, "ratios.txt")
    if not os.path.exists(ratios_path):
        print(f"Skipping {experiment}: no ratios.txt found")
        continue

    with open(ratios_path, "r") as f:
        for line in f:
            step, value = line.strip().split(",")
            data[int(step)][experiment_path] = float(value)

# Get headers
all_steps = sorted(data.keys())
all_dirs = sorted({dirname for step in data.values() for dirname in step})

# Write to CSV
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f"compiled_ratios_{timestamp}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step"] + all_dirs)
    
    for step in all_steps:
        row = [step]
        for dirname in all_dirs:
            row.append(data[step].get(dirname, ""))  # empty if missing
        writer.writerow(row)

print(f"All ratios written to compiled_ratios.csv on {timestamp}")
