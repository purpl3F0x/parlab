import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re

# get to the directory where the file is located
import os

os.chdir(os.path.dirname(__file__))


# Read the file
with open("./kmeans.out", "r") as file:
    data = file.read()

# Regular expression pattern to extract the data
pattern = r"nodes =\s*([A-Za-z0-9]+), nloops =\s*(\d+), total =\s*([\d.]+)s, per loop =\s*([\d.]+)s"

# Parse the data using re.findall
matches = re.findall(pattern, data)

# Create a DataFrame from the parsed data
df = pd.DataFrame(matches, columns=["nodes", "nloops", "total", "per_loop"])

# Convert numeric columns to appropriate data types
df = df.astype({"nodes": str, "nloops": int, "total": float, "per_loop": float})

# Caclulate speedup
df["speedup"] = df["total"][0] / df["total"]

print(df)


# Setup seaborn
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12
bmh_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# Plot runtimes in barplot

ax = sns.barplot(x="nodes", y="total", data=df)
ax.set_title("MPI K-Means Runtime")
ax.set_xlabel("# Nodes")
ax.set_ylabel("Runtime (s)")
plt.savefig("kmeans_runtime.svg")
plt.close()

# Plot speedup in lineplot
ax = sns.lineplot(x="nodes", y="speedup", data=df, markers=True)
ax.set_title("MPI K-Means Speedup")
ax.set_xlabel("# Nodes")
ax.set_ylabel("Speedup")
plt.savefig("kmeans_speedup.svg")
plt.close()

############################
# Compare with omp speedup #
############################


# Regex to extract details from each line
pattern = re.compile(
    r"nthreads =\s+(?P<threads>(\d+)|sequential), nloops =\s+(?P<loops>\d+), total =\s+(?P<total_time>\d+\.\d+)s, per loop =\s+(?P<loop_time>\d+\.\d+)s.*"
)

data = []
# Read results
with open("../../../lab2/kmeans/results/reduction_ft.out", "r") as f:
    data = f.read()

# Regular expression pattern to extract the data
pattern = r"nthreads =\s*(\d+), nloops =\s*(\d+), total =\s*([\d.]+)s, per loop =\s*([\d.]+)s, numa =\s*(.+)"

# Parse the data using re.findall
matches = re.findall(pattern, data)

# Create a DataFrame from the parsed data
df_omp = pd.DataFrame(matches, columns=["nodes", "nloops", "total", "per_loop", "numa"])

# Convert numeric columns to appropriate data types
df_omp = df_omp.astype({"nodes": int, "nloops": int, "total": float, "per_loop": float})
df_omp.drop(columns=["numa", "per_loop", "nloops"], inplace=True)
df_omp["speedup"] = df_omp["total"][0] / df_omp["total"]
df_omp.drop(columns=["total"], inplace=True)
# print(df_omp)

# merge the two dataframes
df_compare = df
df_compare.drop(index=df_compare.index[0], inplace=True)
df_compare["speedup"] = df_compare["total"][1] / df_compare["total"]
df_compare["nodes"] = df_compare["nodes"].astype(int)
df_compare = df_compare.merge(
    df_omp, left_on="nodes", right_on="nodes", suffixes=("_mpi", "_omp")
)

print(df_compare)

# Plot speedup in lineplot
ax = sns.lineplot(
    x="nodes", y="speedup_mpi", data=df_compare, markers=True, marker="o", label="MPI"
)
sns.lineplot(
    x="nodes", y="speedup_omp", data=df_compare, markers=True, marker="o", label="OMP"
)

ax.set_title("K-Means Speedup Comparison")
ax.set_xlabel("# Threads / Nodes")
ax.set_ylabel("Speedup")
plt.savefig("kmeans_speedup_comparison.svg")
plt.close()
