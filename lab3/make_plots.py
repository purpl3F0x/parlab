import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re


COORDINATES=32
SIZE=1024
CLUSTERS=64
METRICS_FILE = f"silver1-V100_Sz-{SIZE}_Coo-{COORDINATES}_Cl-{CLUSTERS}.csv"
SEQUENTIAL_FILE = f"Sz-{SIZE}_Coo-{COORDINATES}_Cl-{CLUSTERS}.csv"

# Define folders and file paths
results_folder = "results"
plots_folder = "plots"
plots_config_subfolder = '_'.join(re.findall(r'\d+', SEQUENTIAL_FILE))
os.makedirs(os.path.join(plots_folder, plots_config_subfolder), exist_ok=True)

gpu_metrics_file = os.path.join(results_folder, METRICS_FILE)
sequential_metrics_file = os.path.join(results_folder, SEQUENTIAL_FILE)

# Load the GPU metrics data
gpu_data = pd.read_csv(gpu_metrics_file)
gpu_data = gpu_data[gpu_data['Implementation'] != "Sequential"]  # Remove Sequential from GPU data
gpu_data = gpu_data[['Implementation', 'blockSize', 'avg_cpu_time', 'avg_gpu_time', 'transfers_time']]

# Load the Sequential metrics data
seq_data = pd.read_csv(sequential_metrics_file)
seq_time = seq_data.loc[seq_data['Implementation'] == "Sequential", 'av_loop_t'].values[0]

# Prepare the data for the stacked barplot (time breakdown)
gpu_data['total_time'] = gpu_data['avg_cpu_time'] + gpu_data['avg_gpu_time'] + gpu_data['transfers_time']

# Prepare the speedup plot
gpu_data['speedup'] = seq_time / gpu_data['total_time']

# Set the plot style
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12

custom_palette = ["#348ABD", "#A60628", "#7A68A6"]

# Plot the time breakdown and speedup for each implementation
for impl in gpu_data['Implementation'].unique():
    # Filter data by the implementation
    impl_data = gpu_data[gpu_data['Implementation'] == impl]

    # Create the stacked time breakdown plot for the implementation (with Sequential included)
    plt.figure(figsize=(12, 8))

    # Plot the Sequential data as individual bars (not stacked)
    seq_data_filtered = seq_data[seq_data['Implementation'] == "Sequential"]
    sns.barplot(
        data=seq_data_filtered,
        x='Implementation',
        y='av_loop_t',
        color="#467821",  # Sequential color (distinct)
        label="Sequential",
        ci=None
    )

    # Plot each component separately to stack them for non-sequential implementations
    sns.barplot(
        data=impl_data,
        x='blockSize',
        y='avg_cpu_time',
        color="#348ABD",  # CPU time color
        label="CPU Time"
    )

    sns.barplot(
        data=impl_data,
        x='blockSize',
        y='avg_gpu_time',
        bottom=impl_data['avg_cpu_time'],  # Stack on top of CPU time
        color="#A60628",  # GPU time color
        label="GPU Time"
    )

    sns.barplot(
        data=impl_data,
        x='blockSize',
        y='transfers_time',
        bottom=impl_data['avg_cpu_time'] + impl_data['avg_gpu_time'],  # Stack on top of both CPU and GPU times
        color="#7A68A6",  # Transfers time color
        label="Transfers Time"
    )

    # Set plot details for with Sequential
    plt.title(f"Execution Time by Block Size ({impl} Implementation - With Sequential)", fontsize=16)
    plt.suptitle("Configuration: {" + str(SIZE) + "-" + str(COORDINATES) + "-" + str(CLUSTERS) + "}", fontsize=12, y=0.95)
    plt.xlabel("Block Size", fontsize=14)
    plt.ylabel("Time (ms)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add the legend with the correct title and position it to the right of the plot
    plt.legend(title="Component", fontsize=12, title_fontsize=14, loc="upper left", bbox_to_anchor=(1, 1))

    # Save the plot as an SVG file (with Sequential)
    plt.savefig(os.path.join(plots_folder, plots_config_subfolder, f"time_breakdown_{impl}_with_sequential.svg"), format="svg", bbox_inches="tight")
    plt.close()

    # Create the stacked time breakdown plot for the implementation (without Sequential)
    plt.figure(figsize=(12, 8))

    # Plot each component separately to stack them for non-sequential implementations
    sns.barplot(
        data=impl_data,
        x='blockSize',
        y='avg_cpu_time',
        color="#348ABD",  # CPU time color
        label="CPU Time"
    )

    sns.barplot(
        data=impl_data,
        x='blockSize',
        y='avg_gpu_time',
        bottom=impl_data['avg_cpu_time'],  # Stack on top of CPU time
        color="#A60628",  # GPU time color
        label="GPU Time"
    )

    sns.barplot(
        data=impl_data,
        x='blockSize',
        y='transfers_time',
        bottom=impl_data['avg_cpu_time'] + impl_data['avg_gpu_time'],  # Stack on top of both CPU and GPU times
        color="#7A68A6",  # Transfers time color
        label="Transfers Time"
    )

    # Set plot details for without Sequential
    plt.title(f"Execution Time by Block Size ({impl} Implementation - Without Sequential)", fontsize=16)
    plt.suptitle("Configuration: {" + str(SIZE) + "-" + str(COORDINATES) + "-" + str(CLUSTERS) + "}", fontsize=12, y=0.95)
    plt.xlabel("Block Size", fontsize=14)
    plt.ylabel("Time (ms)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add the legend with the correct title and position it to the right of the plot
    plt.legend(title="Component", fontsize=12, title_fontsize=14, loc="upper left", bbox_to_anchor=(1, 1))

    # Save the plot as an SVG file (without Sequential)
    plt.savefig(os.path.join(plots_folder, plots_config_subfolder, f"time_breakdown_{impl}_without_sequential.svg"), format="svg", bbox_inches="tight")
    plt.close()

    # Create the speedup plot for the implementation
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=impl_data,
        x='blockSize',
        y='speedup',
        color="#348ABD"
    )
    plt.title(f"Speedup by Block Size ({impl} Implementation)", fontsize=16)
    plt.suptitle("Configuration: {" + str(SIZE) + "-" + str(COORDINATES) + "-" + str(CLUSTERS) + "}", fontsize=12, y=0.95)
    plt.xlabel("Block Size", fontsize=14)
    plt.ylabel("Speedup", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add the legend for speedup
    # plt.legend(["Speedup"], loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=12)

    # Save the plot as an SVG file
    plt.savefig(os.path.join(plots_folder, plots_config_subfolder, f"speedup_{impl}.svg"), format="svg", bbox_inches="tight")
    plt.close()

print(f"Plots saved in the '{plots_folder}' folder.")
