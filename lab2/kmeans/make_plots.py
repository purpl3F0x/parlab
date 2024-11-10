import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import os

# Dictionary to store data
data = []

# Create output directory
if not os.path.exists("./plots"):
    os.makedirs("./plots")

# Set the style of the plots
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12
bmh_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Regex to extract details from each line
pattern = re.compile(
    r"nthreads =\s+(?P<threads>(\d+)|sequential), nloops =\s+(?P<loops>\d+), total =\s+(?P<total_time>\d+\.\d+)s, per loop =\s+(?P<loop_time>\d+\.\d+)s.*"
)

# Read results
with open("./results/naive.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": match.group("threads"),
                    # "loops": int(match.group("loops")),
                    "total_time": float(match.group("total_time")),
                    "loop_time": float(match.group("loop_time")),
                }
            )

# Create a DataFrame from the data
df = pd.DataFrame(data)

data = []
with open("./results/naive_bind.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": match.group("threads"),
                    # "loops": int(match.group("loops")),
                    "bind_total_time": float(match.group("total_time")),
                    "bind_loop_time": float(match.group("loop_time")),
                }
            )

df_bind = pd.DataFrame(data)

# merge the two dataframes
df = df.merge(df_bind, on=["threads"])


# Calculate the speedup
df["speedup"] = df["loop_time"][0] / df["loop_time"]
df["bind_speedup"] = df["bind_loop_time"][0] / df["bind_loop_time"]

print(df)

# Plot the results and speedup
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
sns.barplot(data=df, x="threads", y="loop_time", color=next(bmh_colors), ax=axs[0])
axs[0].set_xlabel("Number of threads")
axs[0].set_ylabel("Time (s) - Per Loop")
axs[0].set_title("Naive K-means")

sns.lineplot(data=df, x="threads", y="speedup", ax=axs[1])
axs[1].set_xlabel("Number of threads")
axs[1].set_ylabel("Speedup")
axs[1].set_title("Naive K-means")

fig.suptitle("Naive K-Means", fontsize=24)
plt.tight_layout()
plt.savefig("./plots/naive.svg")
plt.close()

# Compare with and without binding

df_melted = df.melt(
    id_vars=["threads"],
    value_vars=["loop_time", "bind_loop_time"],
    var_name="Bind",
    value_name="time",
)
df_melted["Bind"] = df_melted["Bind"].replace(
    {"loop_time": "Without", "bind_loop_time": "With"}
)

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

sns.barplot(x="threads", y="time", hue="Bind", data=df_melted, ax=axs[0])
axs[0].set_xlabel("Number of threads")
axs[0].set_ylabel("Time (s) - Per Loop")
axs[0].set_title("Time")

df_melted = df.melt(
    id_vars=["threads"],
    value_vars=["speedup", "bind_speedup"],
    var_name="Bind",
    value_name="speedup_",
)
df_melted["Bind"] = df_melted["Bind"].replace(
    {"speedup": "Without", "bind_speedup": "With"}
)

sns.lineplot(x="threads", y="speedup_", hue="Bind", data=df_melted, ax=axs[1])
axs[1].set_xlabel("Number of threads")
axs[1].set_ylabel("Speedup")
axs[1].set_title("Speedup")

plt.suptitle("Comparison of With and Without Binding", fontsize=24)
plt.tight_layout()
plt.savefig("./plots/naive_bind.svg")
plt.close()

# plot the difference of bind and non-bind
df["diff"] = df["loop_time"] - df["bind_loop_time"]

sns.barplot(data=df, x="threads", y="diff", color=next(bmh_colors))
plt.xlabel("Number of threads")
plt.ylabel("Time (s) - Per Loop")
plt.title("Difference between With and Without Binding")
plt.savefig("./plots/naive_bind_diff.svg")
plt.close()

#######################################
########### Reduction Plots ###########
#######################################
# Read results
data = []
with open("./results/reduction.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": match.group("threads"),
                    # "loops": int(match.group("loops")),
                    "total_time": float(match.group("total_time")),
                    "loop_time": float(match.group("loop_time")),
                    "numa-aware": False,
                }
            )
df = pd.DataFrame(data)
df_reduce = df.copy()
df["speedup"] = df["loop_time"][0] / df["loop_time"]

print(df)

# plot the results
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

sns.barplot(data=df, x="threads", y="loop_time", ax=axs[0])
axs[0].set_xlabel("Number of threads")
axs[0].set_ylabel("Time (s) - Per Loop")
axs[0].set_title("Reduction K-means")

sns.lineplot(data=df, x="threads", y="speedup", ax=axs[1])
axs[1].set_xlabel("Number of threads")
axs[1].set_ylabel("Speedup")
axs[1].set_title("Reduction K-means")

fig.suptitle("Reduction K-Means", fontsize=24)
plt.tight_layout()
plt.savefig("./plots/reduction.svg")
plt.close()

data = []
with open("./results/reduction_small.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": match.group("threads"),
                    # "loops": int(match.group("loops")),
                    "total_time": float(match.group("total_time")),
                    "loop_time": float(match.group("loop_time")),
                    "numa-aware": False,
                }
            )
df_small = pd.DataFrame(data)

# Compare scaling of df and df_small
df["speedup"] = df["loop_time"][0] / df["loop_time"]
df_small["speedup"] = df_small["loop_time"][0] / df_small["loop_time"]

# Μerge the two dataframes
df = df.merge(df_small, on=["threads"], suffixes=("_big", "_small"))
print(df)

sns.lineplot(data=df, x="threads", y="speedup_big", label="{256, 16, 32, 10}")
sns.lineplot(data=df, x="threads", y="speedup_small", label="{256, 1, 4, 10}")

plt.xlabel("Number of threads")
plt.ylabel("Speedup")
plt.legend(title="(Size, Coords, Clusters, Loops)")
plt.title("Reduction K-means")
plt.savefig("./plots/reduction_speedup.svg")
plt.close()


data = []
with open("./results/reduction_ft.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": match.group("threads"),
                    "total_time": float(match.group("total_time")),
                    "loop_time": float(match.group("loop_time")),
                    "numa-aware": False,
                }
            )
df_ft = pd.DataFrame(data)

data = []
with open("./results/reduction_ft_small.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": match.group("threads"),
                    "total_time": float(match.group("total_time")),
                    "loop_time": float(match.group("loop_time")),
                    "numa-aware": False,
                }
            )
df_small_ft = pd.DataFrame(data)

# Merge all the dataframes
df = df.merge(df_ft, on=["threads"], suffixes=("_big", "_ft"))
df = df.merge(df_small_ft, on=["threads"], suffixes=("_ft", "_small_ft"))

# Calculate the speedup
df["speedup_ft"] = df["loop_time_ft"][0] / df["loop_time_ft"]
df["speedup_small_ft"] = df["loop_time_small_ft"][0] / df["loop_time_small_ft"]

# Plot the speedup
sns.lineplot(data=df, x="threads", y="speedup_big", label="Big")
sns.lineplot(data=df, x="threads", y="speedup_small", label="Small")
sns.lineplot(data=df, x="threads", y="speedup_ft", label="BIG with FT")
sns.lineplot(data=df, x="threads", y="speedup_small_ft", label="Small with FT")
plt.xlabel("Number of threads")
plt.ylabel("Speedup")
plt.legend(title="Size")
plt.title("Reduction K-means")
plt.savefig("./plots/reduction_speedup_ft.svg")
plt.close()

# Compare reduction with NUMA-aware io
data = []
with open("./results/reduction_numa_io.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": match.group("threads"),
                    "total_time": float(match.group("total_time")),
                    "loop_time": float(match.group("loop_time")),
                    "numa-aware": True,
                }
            )
df_numa = pd.DataFrame(data)
df_numa["speedup"] = df_numa["loop_time"][0] / df_numa["loop_time"]
df_reduce["speedup"] = df_reduce["loop_time"][0] / df_reduce["loop_time"]

# Merge the two dataframes
df = df_reduce.merge(df_numa, on=["threads"], suffixes=("_non_numa", "_numa"))
df1 = df.melt(
    id_vars=["threads"],
    value_vars=["loop_time_non_numa", "loop_time_numa"],
    var_name="NUMA",
    value_name="time",
)
df2 = df.melt(
    id_vars=["threads"],
    value_vars=["speedup_non_numa", "speedup_numa"],
    var_name="NUMA",
    value_name="speedup",
)

print(df1)

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
sns.barplot(x="threads", y="time", hue="NUMA", data=df1, ax=axs[0])
axs[0].set_xlabel("Number of threads")
axs[0].set_ylabel("Time (s) - Per Loop")
axs[0].set_title("Time")

sns.lineplot(x="threads", y="speedup", hue="NUMA", data=df2, ax=axs[1])
axs[1].set_xlabel("Number of threads")
axs[1].set_ylabel("Speedup")
axs[1].set_title("Speedup")

plt.suptitle("Reduction K-means using NUMA-Aware IO", fontsize=24)
plt.tight_layout()
plt.savefig("./plots/reduction_numa_io.svg")
