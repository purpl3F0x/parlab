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
                        "Px": int(px),
                        "Py": int(py),
                        "Iterations": int(iterations),
                        "ComputationTime": float(comp_time),
                        "TotalTime": float(total_time),
                        "Midpoint": float(midpoint),
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
custom_palette = {
    "Gauss-Seidel": "#467821",
    "Jacobi": "#348ABD",
    "Red-Black": "#A60628",
}

# Unique grid sizes
grid_sizes = df[["X", "Y"]].drop_duplicates().values

# Generate speedup plots for each grid size
for x, y in grid_sizes:
    subset = df[(df["X"] == x) & (df["Y"] == y)]

    # Compute speedup using the TotalTime of Workers=1 as baseline for each method
    baseline_times = (
        subset[subset["Workers"] == 1].set_index("Method")["TotalTime"].to_dict()
    )
    subset["Speedup"] = subset.apply(
        lambda row: baseline_times[row["Method"]] / row["TotalTime"], axis=1
    )

    plt.figure()
    sns.lineplot(
        data=subset,
        x="Workers",
        y="Speedup",
        hue="Method",
        marker="o",
        palette=custom_palette,
    )

    plt.xticks(
        sorted(subset["Workers"].unique()),
        labels=sorted(subset["Workers"].unique()),
        fontsize=12,
    )
    plt.yticks(fontsize=12)

    plt.title(f"Speedup for Grid {x}x{y}", fontsize=16)
    plt.xlabel("MPI Workers", fontsize=14)
    plt.ylabel("Speedup", fontsize=14)
    plt.legend(title="Method", fontsize=12, title_fontsize=14)

    plot_filename = os.path.join(PLOTS_FOLDER, f"speedup_{x}x{y}.svg")
    plt.savefig(plot_filename, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_filename}")
