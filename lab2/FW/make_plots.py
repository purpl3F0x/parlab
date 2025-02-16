import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the custom color palette
colors = ["#348ABD", "#A60628", "#7A68A6", "#467821", "#D55E00", "#CC79A7"]

# Set the style of the plots
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12

# Define folders for metrics and plots
METRICS_FOLDER = "results"
PLOTS_FOLDER = "plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Load data and strip whitespace from column names
fw_seq = pd.read_csv(os.path.join(METRICS_FOLDER, "fw_seq.csv")).rename(
    columns=lambda x: x.strip()
)
fw_sr = pd.read_csv(os.path.join(METRICS_FOLDER, "fw_sr.csv")).rename(
    columns=lambda x: x.strip()
)
fw_tiled = pd.read_csv(os.path.join(METRICS_FOLDER, "fw_tiled_on_roids.csv")).rename(
    columns=lambda x: x.strip()
)

# Plot 1: Sequential Floyd-Warshall Edition (single bar color)
# plt.figure(figsize=(14, 7))
sns.barplot(
    data=fw_seq, x="BATCH_SIZE", y="TIME", color=colors[0]
)  # Apply first color in palette
plt.title("Sequential Floyd-Warshall Edition", fontsize=16)  # Increase title font size
plt.xlabel("Batch Size", fontsize=14)  # Increase xlabel font size
plt.ylabel("Time (s)", fontsize=14)  # Increase ylabel font size
plt.tick_params(axis="both", which="major", labelsize=12)  # Increase tick label size
plt.savefig(os.path.join(PLOTS_FOLDER, "Sequential_Floyd_Warshall.svg"), format="svg")
plt.close()
exit()
# Plot 2: Recursive Floyd-Warshall Edition
for batch_size in fw_sr["BATCH_SIZE"].unique():
    plt.figure(figsize=(14, 7))
    subset = fw_sr[
        fw_sr["BATCH_SIZE"] == batch_size
    ].copy()  # Use copy to avoid the warning
    subset.loc[:, "Thread_Label"] = subset["THREADS"].apply(lambda x: f"{x}")

    # Get the sequential time for this batch size from fw_seq
    seq_time = fw_seq[fw_seq["BATCH_SIZE"] == batch_size]["TIME"].values[0]

    # Add the sequential time as a reference (Thread == 1)
    seq_data = pd.DataFrame(
        {
            "Thread_Label": ["Sequential"],
            "TIME": [seq_time],
            "TILE_SIZE": [
                "None"
            ],  # Set a string placeholder for sequential's TILE_SIZE
        }
    )

    # Append the sequential data to the subset
    subset = pd.concat([seq_data, subset], ignore_index=True)

    # Use colors from palette for hue
    sns.barplot(
        data=subset, x="Thread_Label", y="TIME", hue="TILE_SIZE", palette=colors
    )

    plt.title(
        f"Recursive Floyd-Warshall Edition (Batch Size {batch_size})", fontsize=16
    )  # Increase title font size
    plt.xlabel("Threads", fontsize=14)  # Increase xlabel font size
    plt.ylabel("Time (s)", fontsize=14)  # Increase ylabel font size
    plt.legend(
        title="Tile Size", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12
    )  # Legend font size
    plt.tick_params(
        axis="both", which="major", labelsize=12
    )  # Increase tick label size
    plt.savefig(
        os.path.join(PLOTS_FOLDER, f"Recursive_Floyd_Warshall_Batch_{batch_size}.svg"),
        format="svg",
    )
    plt.close()


# Plot 3: Tiled Floyd-Warshall Edition
for batch_size in fw_tiled["BATCH_SIZE"].unique():
    plt.figure(figsize=(14, 7))
    subset = fw_tiled[
        fw_tiled["BATCH_SIZE"] == batch_size
    ].copy()  # Use copy to avoid the warning
    subset.loc[:, "Thread_Label"] = subset["THREADS"].apply(lambda x: f"{x} Threads")

    # Get the sequential time for this batch size from fw_seq
    seq_time = fw_seq[fw_seq["BATCH_SIZE"] == batch_size]["TIME"].values[0]

    # Add the sequential time as a reference (Thread == 1)
    seq_data = pd.DataFrame(
        {
            "Thread_Label": ["Sequential"],
            "TIME": [seq_time],
            "TILE_SIZE": [
                "None"
            ],  # Set a string placeholder for sequential's TILE_SIZE
        }
    )

    # Append the sequential data to the subset
    subset = pd.concat([seq_data, subset], ignore_index=True)

    # Use colors from palette for hue
    g = sns.barplot(
        data=subset, x="Thread_Label", y="TIME", hue="TILE_SIZE", palette=colors
    )
    g.bar_label(g.containers[0], fontsize=12);
    g.bar_label(g.containers[1], fontsize=12, rotation=90);
    g.bar_label(g.containers[2], fontsize=12, rotation=90);
    g.bar_label(g.containers[3], fontsize=12, rotation=90);
    g.bar_label(g.containers[4], fontsize=12, rotation=90);





plt.title(
    f"Very-Very Fast Tiled Floyd-Warshall Edition (Batch Size {batch_size})", fontsize=16
)  # Increase title font size
plt.xlabel("Threads", fontsize=14)  # Increase xlabel font size
plt.ylabel("Time (s)", fontsize=14)  # Increase ylabel font size
plt.legend(
    title="Tile Size", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12
)  # Legend font size
plt.tick_params(
    axis="both", which="major", labelsize=12
)  # Increase tick label size
plt.savefig(
    os.path.join(PLOTS_FOLDER, f"Tiled_Floyd_Warshall_Batch_{batch_size}.svg"),
    format="svg",
)
plt.close()
