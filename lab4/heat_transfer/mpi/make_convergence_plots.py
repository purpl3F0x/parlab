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
    "jacobi_heat_transfer_mpi_CONV.out",
    "redblack_mpi_heat_transfer_mpi_CONV.out",
    "gauss_heat_transfer_mpi_CONV.out",
]


# Read data from files
def read_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(
                r"([\w-]+) X=(\d+), Y=(\d+), Workers=(\d+), Px=(\d+), Py=(\d+), Iter=(\d+), ComputationTime=([\d.]+), TotalTime=([\d.]+), ConvergenceTime=([\d.]+), midpoint=([\d.]+)",
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
                    conv_time,
                    midpoint,
                ) = match.groups()
                # Adjust times for Jacobi method
                if method == "Jacobi":
                    method = "Jacobi / 10"
                    comp_time = float(comp_time) / 10
                    total_time = float(total_time) / 10
                    conv_time = float(conv_time) / 10
                else:
                    comp_time = float(comp_time)
                    total_time = float(total_time)
                    conv_time = float(conv_time)
                data.append(
                    {
                        "Method": method,
                        "X": int(x),
                        "Y": int(y),
                        "Workers": int(workers),
                        "ComputationTime": comp_time,
                        "TotalTime": total_time,
                        "ConvergenceTime": conv_time,
                    }
                )
    return data


# Process all files
data = []
for file in FILES:
    file_path = os.path.join(RESULTS_FOLDER, file)
    data.extend(read_data(file_path))

df = pd.DataFrame(data)

# Compute average times for each method
avg_df = (
    df.groupby("Method")
    .agg({"ComputationTime": "mean", "TotalTime": "mean", "ConvergenceTime": "mean"})
    .reset_index()
)

# Set plot style
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12

# Define colors for computation time, convergence time, and total time
COMPUTATION_COLOR = "#348ABD"  # Blue
CONVERGENCE_COLOR = "#7A68A6"  # Purple
TOTAL_COLOR = "#A60628"  # Red

# Generate stacked bar plot
plt.figure()

# Plot total time bars
sns.barplot(
    data=avg_df,
    x="Method",
    y="TotalTime",
    color=TOTAL_COLOR,
    edgecolor="black",
    label="Total Time",
)

# Plot convergence time bars
for i, row in avg_df.iterrows():
    plt.bar(
        row["Method"],
        row["ConvergenceTime"],
        color=CONVERGENCE_COLOR,
        alpha=1,
        label="Convergence Time" if i == 0 else "",  # Add label only once
    )

# Plot computation time bars
for i, row in avg_df.iterrows():
    plt.bar(
        row["Method"],
        row["ComputationTime"],
        color=COMPUTATION_COLOR,
        alpha=1,
        label="Computation Time" if i == 0 else "",  # Add label only once
    )

# Manually create legend handles
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=COMPUTATION_COLOR, label="Computation Time"),
    Patch(facecolor=CONVERGENCE_COLOR, label="Convergence Time"),
    Patch(facecolor=TOTAL_COLOR, label="Total Time"),
]

# Add legend
plt.legend(handles=legend_elements, title="Time Type", loc="best")

plt.title("Time Comparison (64 MPI Workers)", fontsize=16)
plt.xlabel("Method", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.ylim(0, avg_df["TotalTime"].max() * 1.1)

plot_filename = os.path.join(PLOTS_FOLDER, "stacked_bar_average_times.svg")
plt.savefig(plot_filename, format="svg", bbox_inches="tight")
plt.close()
print(f"Saved: {plot_filename}")
