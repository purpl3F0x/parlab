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
# plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12
bmh_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# Regex to extract details from each line
pattern = re.compile(
    r"nthreads =\s+(?P<threads>\d+), nloops =\s+(?P<loops>\d+), total =\s+(?P<total_time>\d+\.\d+)s, per loop =\s+(?P<loop_time>\d+\.\d+)s.*"
)

# Read results
with open("./results/naive.out", "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            data.append(
                {
                    "threads": int(match.group("threads")),
                    "loops": int(match.group("loops")),
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
                    "threads": int(match.group("threads")),
                    "loops": int(match.group("loops")),
                    "bind_total_time": float(match.group("total_time")),
                    "bind_loop_time": float(match.group("loop_time")),
                }
            )

df_bind = pd.DataFrame(data)

# merge the two dataframes
df = df.merge(df_bind, on=["threads", "loops"])

df.sort_values(by=["threads"], inplace=True)

# Calculate the speedup
# df["speedup"] = df["loop_time"][0] / df["loop_time"]

print(df)

# Plot the results
sns.barplot(data=df, x="threads", y="loop_time", color=next(bmh_colors))
plt.xlabel("Number of threads")
plt.ylabel("Time (s)")
plt.title("Naive K-means")

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

sns.barplot(x="threads", y="time", hue="Bind", data=df_melted)
plt.xlabel("Number of threads")
plt.ylabel("Time (s)")
plt.title("Comparison of With and Without Binding")
plt.savefig("./plots/naive_bind.svg")
plt.close()

# plot the difference of bind and non-bind
df["diff"] = df["loop_time"] - df["bind_loop_time"]

sns.barplot(data=df, x="threads", y="diff", color=next(bmh_colors))
plt.xlabel("Number of threads")
plt.ylabel("Time (s)")
plt.title("Difference between With and Without Binding")
plt.savefig("./plots/naive_bind_diff.svg")
