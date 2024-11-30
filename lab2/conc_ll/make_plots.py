# import os
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Input and output folder paths
# results_folder = "results"
# plots_folder = "plots"
# os.makedirs(plots_folder, exist_ok=True)

# # Parsing the result files
# def parse_results(folder):
#     data = []
#     for file_name in os.listdir(folder):
#         if file_name.endswith(".out"):
#             sync_type = file_name.split(".")[0]
#             with open(os.path.join(folder, file_name), 'r') as file:
#                 sync_type_value = None
#                 for line in file:
#                     if line.startswith("MT_CONF"):
#                         sync_type_value = line.strip().split("=")[1]
#                     elif line.startswith("Nthreads"):
#                         parts = line.strip().split()
#                         nthreads = int(parts[1])
#                         list_size = int(parts[5])
#                         workload = parts[7]
#                         throughput = float(parts[9])
#                         data.append({
#                             "SyncType": sync_type,
#                             "SyncConfig": sync_type_value,
#                             "Threads": nthreads,
#                             "ListSize": list_size,
#                             "Workload": workload,
#                             "Throughput": throughput
#                         })
#     return pd.DataFrame(data)

# # Load data
# data = parse_results(results_folder)

# # Define custom color palette
# custom_palette = ["#348ABD", "#A60628", "#7A68A6", "#467821", "#D55E00", "#CC79A7"]

# # Generate plots for each sync type and list size
# for (sync_type, list_size), group_data in data.groupby(["SyncType", "ListSize"]):
#     plt.figure(figsize=(12, 8))
#     sns.barplot(
#         data=group_data,
#         x="Threads",
#         y="Throughput",
#         hue="Workload",
#         palette=custom_palette[:group_data['Workload'].nunique()]
#     )
#     plt.title(f"Throughput vs Number of Threads for {sync_type} (List Size: {list_size})", fontsize=16)
#     plt.xlabel("Number of Threads", fontsize=14)
#     plt.ylabel("Throughput (Kops/sec)", fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.legend(title="Workload", fontsize=12, title_fontsize=14)
#     plt.tight_layout()
#     # Save plot
#     output_path = os.path.join(plots_folder, f"{sync_type}_list_{list_size}_throughput_vs_threads.svg")
#     plt.savefig(output_path, format="svg")
#     plt.close()

# print(f"Plots saved to {plots_folder}.")

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
    ax = sns.barplot(
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

    # Adjust the y-axis upper limit to create space above the bars for the text labels
    y_max = ax.get_ylim()[1]  # Get the current maximum y limit
    ax.set_ylim(0, y_max * 1.05)  # Increase the upper limit by 10% for more space

    # Add the exact y-axis value (throughput) on top of each bar, rotated vertically
    for p in ax.patches:
        if (p.get_width() == 0):
            continue
        x_position = p.get_x() + p.get_width() / 2  # Get the center of the bar
        y_position = p.get_height() + 0.8  # Move label further up to avoid overlap
        ax.text(
            x_position,  # X position
            y_position,  # Y position
            f'{p.get_height():.2f}',  # The throughput value, formatted
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=10,  # Font size of the annotation
            rotation=90  # Rotate the label vertically
        )

    # Save plot
    output_path = os.path.join(plots_folder, f"{sync_type}_list_{list_size}_throughput_vs_threads.svg")
    plt.savefig(output_path, format="svg")
    plt.close()

print(f"Plots saved to {plots_folder}.")
