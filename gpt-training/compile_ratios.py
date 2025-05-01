import os
import csv
from collections import defaultdict
from datetime import datetime
from results_visualization import visualize_results

experiments_dir = "experiments"

# {step: {tokenizer: entropy_value}}
entropy_data = defaultdict(dict)

# Iterate through experiments
for experiment in os.listdir(experiments_dir):
    experiment_path = os.path.join(experiments_dir, experiment)
    if not os.path.isdir(experiment_path) or experiment in ["testing", "model_trial1"]:
        continue

    ratios_path = os.path.join(experiment_path, "ratios.txt")
    if not os.path.exists(ratios_path):
        print(f"Skipping {experiment}: no ratios.txt found")
        continue

    with open(ratios_path, "r") as f:
        for line in f:
            step, value = line.strip().split(",")
            step = int(step)
            entropy = float(value)
            entropy_data[step][experiment_path] = entropy

# Get headers
all_steps = sorted(entropy_data.keys())
all_dirs = sorted({dirname for step in entropy_data.values() for dirname in step})

# Create timestamp
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Write CSVs
def write_csv(path, data):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + all_dirs)
        
        for step in all_steps:
            row = [step]
            for dirname in all_dirs:
                row.append(data[step].get(dirname, ""))
            writer.writerow(row)

entropy_csv_path = f"compiled_entropy_ratios.csv"
write_csv(entropy_csv_path, entropy_data)

print(f"Entropy and perplexity ratios written to:")
print(f" - {entropy_csv_path}")

visualize_results(entropy_csv_path)
