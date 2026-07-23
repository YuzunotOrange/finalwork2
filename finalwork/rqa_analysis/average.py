import os
import re
import numpy as  np

directory_path = "./rqa2"

metrics = {
    "Determinism (DET)": [],
    "Laminarity (LAM)": [],
    "Entropy vertical lines (V_entr)": [],
}

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        with open(os.path.join(directory_path, filename), "r") as file:
            content = file.read()
            for key in metrics:
                match = re.search(rf"{re.escape(key)}:\s+([0-9eE\+\-\.nan]+)", content)
                if match:
                    value = match.group(1)
                    try:
                        value = float(value)
                        if not np.isnan(value):
                            metrics[key].append(value)
                    except ValueError:
                        continue
print("=== 平均値 ===")
for key, values in metrics.items():
    if values:
        print(f"{key}: {np.mean(values):.6f}")
    else:
        print(f"{key}: No vaild data found.")