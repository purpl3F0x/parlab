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
df["speedup"] = df.groupby(["size", "steps"])[["time"]].transform(
    lambda x: x.iloc[0] / x
)

print(df)


sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12
bmh_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# # Time plot
# plt.title("Time vs Threads")
# g = sns.FacetGrid(df, col="size", hue="size", sharey=False, height=5)
# g.map(sns.lineplot, "threads", "time", marker="o", markersize=10)
# g.add_legend()
# g.set(xticks=df.threads.unique())
# g.set_titles("Size = {col_name}")
# g.set_xlabels("# Threads")
# g.set_ylabels("Time (s)")
# plt.savefig("time.svg")

# # Speedup plot

# g = sns.FacetGrid(df, col="size", hue="size", sharey=True, height=5)
# g.map(sns.lineplot, "threads", "speedup", marker="o", markersize=10)
# g.set_titles("Size = {col_name}")
# g.add_legend()
# g.set(xticks=df.threads.unique())
# g.set_xlabels("# Threads")
# g.set_ylabels("Speedup")
# plt.savefig("speedup.svg")

# Create three graphs for each size
for size in df["size"].unique():
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    current_color = next(bmh_colors)

    # Filter data for the current size
    size_df = df[df["size"] == size]

    # Plot Time vs Threads
    sns.lineplot(
        x="threads",
        y="time",
        data=size_df,
        marker="o",
        markersize=10,
        ax=axs[0],
        color=current_color,
    )
    axs[0].set_title(f"Time vs Threads for Size {size}")
    axs[0].set_xlabel("# Threads")
    axs[0].set_ylabel("Time (s)")
    axs[0].set_xticks(size_df.threads.unique())

    # Plot Speedup vs Threads
    sns.lineplot(
        x="threads",
        y="speedup",
        data=size_df,
        marker="o",
        markersize=10,
        ax=axs[1],
        color=current_color,
    )
    axs[1].set_title(f"Speedup vs Threads for Size {size}")
    axs[1].set_xlabel("# Threads")
    axs[1].set_ylabel("Speedup")
    axs[1].set_xticks(size_df.threads.unique())

    # Add a title to the figure
    fig.suptitle(f"Size = {size}", fontsize=24)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"size_{size}.svg")
    plt.show()
