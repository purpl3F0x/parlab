import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# Define file paths
COORDINATES=32
SIZE=1024
CLUSTERS=64
METRICS_FILE = f"silver1-V100_Sz-{SIZE}_Coo-{COORDINATES}_Cl-{CLUSTERS}.csv"
SEQUENTIAL_FILE = f"Sz-{SIZE}_Coo-{COORDINATES}_Cl-{CLUSTERS}.csv"

results_folder = "results"
plots_folder = "plots"
plots_config_subfolder = '_'.join(re.findall(r'\d+', SEQUENTIAL_FILE))
os.makedirs(os.path.join(plots_folder, plots_config_subfolder), exist_ok=True)

silver1_file = os.path.join(results_folder, METRICS_FILE)
sequential_file = os.path.join(results_folder, SEQUENTIAL_FILE)

# Load data
silver1_data = pd.read_csv(silver1_file)
sequential_data = pd.read_csv(sequential_file)

# Extract sequential time
seq_time = sequential_data.loc[sequential_data['Implementation'] == "Sequential", 'av_loop_t'].values[0]

# Calculate total time and speedup
silver1_data['total_time'] = (
    silver1_data['avg_cpu_time'] + 
    silver1_data['avg_gpu_time'] + 
    silver1_data['transfers_time']
)
silver1_data['speedup'] = seq_time / silver1_data['total_time']

# Set plot style
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12

# Custom colors for the implementations
custom_palette = ["#467821", "#348ABD", "#A60628", "#7A68A6"]


# Get implementations as lowercase string
implementations = '_'.join(silver1_data['Implementation'].str.lower().unique())

# Create a combined speedup plot for all implementations (side by side)
plt.figure(figsize=(12, 8))

# Create a bar plot with bars for each implementation side by side
sns.barplot(
    data=silver1_data,
    x='blockSize',
    y='speedup',
    hue='Implementation',  # Use 'Implementation' to separate the bars
    palette=custom_palette
)

# Set plot details
plt.title("Speedup by Block Size for Different Implementations", fontsize=16)
plt.suptitle("Configuration: {" + str(SIZE) + "-" + str(COORDINATES) + "-" + str(CLUSTERS) + "}", fontsize=12, y=0.95)
plt.xlabel("Block Size", fontsize=14)
plt.ylabel("Speedup", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add the legend to differentiate implementations
plt.legend(title="Implementation", fontsize=12, title_fontsize=14, loc="upper left", bbox_to_anchor=(1, 1))

# Save the combined plot as an SVG file
plt.savefig(os.path.join(plots_folder, plots_config_subfolder,  f"combined_speedup_{implementations}.svg"), format="svg", bbox_inches="tight")
plt.close()

print(f"Combined speedup plot saved in the '{plots_folder}' folder.")
