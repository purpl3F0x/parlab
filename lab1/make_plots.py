import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Dictionary to store data
data = []

# Regex to extract details from each line
pattern = re.compile(
    r"GameOfLife: Size (?P<size>\d+) Steps (?P<steps>\d+) Time (?P<time>\d+\.\d+) Threads (?P<threads>\d+)"
)

# Read results
with open("results.out") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            size = int(match.group("size"))
            steps = int(match.group("steps"))
            time = float(match.group("time"))
            threads = int(match.group("threads"))
            data += [
                {"size": size, "steps": steps, "time": time, "threads": threads},
            ]

# Create a DataFrame from the data
df = pd.DataFrame(data)
df.sort_values(by=["size", "steps", "threads"], inplace=True)
df = df.groupby(["size", "threads"]).mean().reset_index()
# Calculate speedup
df["speedup"] = df.groupby("size")["time"].transform(lambda x: x.min() / x)

print(df)


sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"

# Time plot
plt.title("Time vs Threads")
g = sns.FacetGrid(df, col="size", hue="size", sharey=False, height=5)
g.map(sns.lineplot, "threads", "time", marker="o", markersize=10)
g.add_legend()
g.set(xticks=df.threads.unique())
g.set_titles("Size = {col_name}")
g.set_xlabels("# Threads")
g.set_ylabels("Time (s)")
plt.savefig("time.svg")

# Speedup plot

g = sns.FacetGrid(df, col="size", hue="size", sharey=True, height=5)
g.map(sns.lineplot, "threads", "speedup", marker="o", markersize=10)
g.set_titles("Size = {col_name}")
g.add_legend()
g.set(xticks=df.threads.unique())
g.set_xlabels("# Threads")
g.set_ylabels("Speedup")
plt.savefig("speedup.svg")
