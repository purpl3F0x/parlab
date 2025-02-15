import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# Define file paths
RESULTS_FOLDER = "results"
PLOTS_FOLDER = "plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Filenames to read
FILES = [
    "jacobi_heat_transfer_mpi.out",
    "gauss_heat_transfer_mpi.out",
    "redblack_mpi_heat_transfer_mpi.out",
]

# Read data from files
def read_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(
                r"([\w-]+) X=(\d+), Y=(\d+), Workers=(\d+), Px=(\d+), Py=(\d+), Iter=(\d+), ComputationTime=([\d.]+), TotalTime=([\d.]+), midpoint=([\d.]+)",
                line,
            )
            if match:
                (
                    method,
                    x,
                    y,
                    workers,
                    px,
                    py,
                    iterations,
                    comp_time,
                    total_time,
                    midpoint,
                ) = match.groups()
                data.append(
                    {
                        "Method": method,
                        "X": int(x),
                        "Y": int(y),
                        "Workers": int(workers),
                        "ComputationTime": float(comp_time),
                        "TotalTime": float(total_time),
                    }
                )
    return data

# Process all files
data = []
for file in FILES:
    file_path = os.path.join(RESULTS_FOLDER, file)
    data.extend(read_data(file_path))

df = pd.DataFrame(data)

# Set plot style
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12

# Filter for relevant MPI workers
df = df[df["Workers"].isin([8, 16, 32, 64])]

# Unique grid sizes and worker counts
grid_sizes = df[["X", "Y"]].drop_duplicates().values
worker_counts = df["Workers"].unique()

# Define colors for computation time and total time
COMPUTATION_COLOR = "#348ABD"  # Blue
TOTAL_COLOR = "#A60628"        # Red

# Generate stacked bar plots for each grid size and worker count
for x, y in grid_sizes:
    for workers in worker_counts:
        subset = df[(df["X"] == x) & (df["Y"] == y) & (df["Workers"] == workers)]
        
        plt.figure(figsize=(10, 6))
        
        # Plot total time bars
        sns.barplot(
            data=subset,
            x="Method",
            y="TotalTime",
            color=TOTAL_COLOR,
            edgecolor="black",
            label="Total Time"
        )
        
        # Plot computation time bars
        for i, row in subset.iterrows():
            plt.bar(
                row["Method"], 
                row["ComputationTime"], 
                color=COMPUTATION_COLOR,
                alpha=1,
                label="Computation Time" if i == 0 else ""  # Add label only once
            )
        
        # Manually create legend handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COMPUTATION_COLOR, label="Computation Time"),
            Patch(facecolor=TOTAL_COLOR, label="Total Time"),
        ]
        
        # Add legend
        plt.legend(handles=legend_elements, title="Time Type", loc="upper right")
        
        plt.title(f"Computation vs Total Time for Grid {x}x{y} with {workers} Workers", fontsize=16)
        plt.xlabel("Method", fontsize=14)
        plt.ylabel("Time (s)", fontsize=14)
        plt.ylim(0, df[(df["X"] == x) & (df["Y"] == y)]["TotalTime"].max() * 1.1)
        
        plot_filename = os.path.join(PLOTS_FOLDER, f"stacked_bar_{x}x{y}_{workers}workers.svg")
        plt.savefig(plot_filename, format="svg", bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_filename}")