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
df.sort_values(by=["threads"], inplace=True)
print(df)

sns.barplot(data=df, x="threads", y="loop_time", color=next(bmh_colors))

plt.savefig("./plots/naive.png")
