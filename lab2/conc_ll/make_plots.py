import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Input and output folder paths
results_folder = "results"
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Parsing the result files
def parse_results(folder):
    data = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".out"):
            sync_type = file_name.split(".")[0]
            with open(os.path.join(folder, file_name), 'r') as file:
                sync_type_value = None
                for line in file:
                    if line.startswith("MT_CONF"):
                        sync_type_value = line.strip().split("=")[1]
                    elif line.startswith("Nthreads"):
                        parts = line.strip().split()
                        nthreads = int(parts[1])
                        list_size = int(parts[5])
                        workload = parts[7]
                        throughput = float(parts[9])
                        data.append({
                            "SyncType": sync_type,
                            "SyncConfig": sync_type_value,
                            "Threads": nthreads,
                            "ListSize": list_size,
                            "Workload": workload,
                            "Throughput": throughput
                        })
    return pd.DataFrame(data)

# Load data
data = parse_results(results_folder)

# Define custom color palette
custom_palette = ["#348ABD", "#A60628", "#7A68A6", "#467821", "#D55E00", "#CC79A7"]

# Generate plots for each sync type and list size
for (sync_type, list_size), group_data in data.groupby(["SyncType", "ListSize"]):
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=group_data,
        x="Threads",
        y="Throughput",
        hue="Workload",
        palette=custom_palette[:group_data['Workload'].nunique()]
    )
    plt.title(f"Throughput vs Number of Threads for {sync_type} (List Size: {list_size})", fontsize=16)
    plt.xlabel("Number of Threads", fontsize=14)
    plt.ylabel("Throughput (Kops/sec)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Workload", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    # Save plot
    output_path = os.path.join(plots_folder, f"{sync_type}_list_{list_size}_throughput_vs_threads.svg")
    plt.savefig(output_path, format="svg")
    plt.close()

print(f"Plots saved to {plots_folder}.")
