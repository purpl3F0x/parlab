import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Dictionary to store data
data = []

# Regex to extract details from each line
pattern = re.compile(
    r"GameOfLife: Size (?P<size>\d+) Steps (?P<steps>\d+) Time (?P<time>\d+\.\d+) Threads (?P<threads>\d)"
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
print(df)

sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"

# Create a plot for each size
sns.pointplot(data=df[df["size"] == 64], x="threads", y="time")
plt.title("Size 64")
plt.xlabel("Threads")
plt.ylabel("Time (s)")
plt.savefig("64.svg")
plt.clf()


sns.pointplot(data=df[df["size"] == 1024], x="threads", y="time")
plt.title("Size 1024")
plt.xlabel("Threads")
plt.ylabel("Time (s)")
plt.savefig("1024.svg")
plt.clf()


sns.pointplot(data=df[df["size"] == 4096], x="threads", y="time")
plt.title("Size 4096")
plt.xlabel("Threads")
plt.ylabel("Time (s)")
plt.savefig("4096.svg")
