import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# set current dir to the directory of the script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Setup seaborn and matplotlib
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (10, 6)
bmh_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# Read results
df = pd.read_csv("../Execution_logs/silver1-V100_Sz-1024_Coo-32_Cl-64.csv")

print(df.head)

for version in ["Naive", "Transpose", "Shared"]:
    ax = df.query(f'Implementation=="{version}" | Implementation=="Sequential"').plot(
        kind="bar",
        stacked=True,
        x="blockSize",
        y=["loop_total", "alloc_time", "gpu_alloc_time", "gpu_get_time"],
    )
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    ax.set_title(f"{version} vs Sequential")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Block Size")
    plt.xticks(rotation=45, ha="right")
    plt.show()
    plt.close()

    ax = df.query(f'Implementation=="{version}"').plot(
        kind="bar",
        stacked=True,
        x="blockSize",
        y=["avg_gpu_time", "avg_cpu_time", "transfers_time"],
    )
    ax.set_title(f"{version}: Per loop timings")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Block Size")
    plt.xticks(rotation=45, ha="right")
    plt.show()
    # plt.savefig(f"./{version}_per_loop.png")

first_3_df = df.query(
    'Implementation=="Naive" | Implementation=="Transpose" | Implementation=="Shared"'
)

g = sns.catplot(
    data=first_3_df,
    kind="bar",
    x="blockSize",
    y="avg_loop_t",
    hue="Implementation",
    height=6, aspect=8/6
)
g.despine(left=True)
g.set_axis_labels("Block Size", "Time (ms)")
g.fig.suptitle("Average loop time for each implementation", fontsize=16)
g.fig.subplots_adjust(top=0.92)
g.fig.subplots_adjust(bottom=0.11)
plt.xticks(rotation=45, ha="right")
plt.show()
# plt.savefig("./avg_loop_time.png")
