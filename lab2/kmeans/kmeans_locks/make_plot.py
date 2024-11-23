# import os
# import re
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Directory containing the result files
# results_dir = "./results"
# output_dir = "./plots"
# os.makedirs(output_dir, exist_ok=True)

# # Set plot style
# sns.set_theme(style="whitegrid")
# plt.style.use("bmh")
# plt.rcParams["font.family"] = "Cambria"
# plt.rcParams["font.size"] = 12

# # Regex to extract data from the .out files
# pattern = re.compile(
#     r"OpenMP Kmeans - Lock \((?P<lock_type>[^\)]+)\)\s+\(number of threads: (?P<threads>\d+)\).+?"
#     r"nloops =\s+\d+\s+\(total =\s+[\d.]+s\)\s+\(per loop =\s+(?P<loop_time>[\d.]+)s\)",
#     re.DOTALL
# )

# # Extract data from all .out files
# data = []
# for file_name in os.listdir(results_dir):
#     if file_name.endswith(".out"):
#         lock_type = file_name.replace(".out", "").replace("kmeans_", "")
#         file_path = os.path.join(results_dir, file_name)
#         with open(file_path, "r") as f:
#             content = f.read()
#             matches = pattern.finditer(content)
#             for match in matches:
#                 data.append({
#                     "lock_type": lock_type,
#                     "threads": int(match.group("threads")),
#                     "loop_time": float(match.group("loop_time"))
#                 })

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Sort the data for proper plotting
# df.sort_values(by=["lock_type", "threads"], inplace=True)

# # Plot the data
# plt.figure(figsize=(12, 8))
# sns.lineplot(data=df, x="threads", y="loop_time", hue="lock_type", marker="o")

# # Customize plot
# plt.xlabel("Number of Threads")
# plt.ylabel("Loop Time (s)")
# plt.title("K-means Performance by Lock Type")
# plt.legend(title="Lock Type")

# unique_threads = sorted(df["threads"].unique())
# plt.xticks(unique_threads, labels=unique_threads)

# # Save and display the plot
# plt.savefig(os.path.join(output_dir, "kmeans_performance.svg"))
# plt.show()

import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the result files
results_dir = "./results"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# Set plot style
sns.set_theme(style="whitegrid")
plt.style.use("bmh")
plt.rcParams["font.family"] = "Cambria"
plt.rcParams["font.size"] = 12

# Regex to extract data from the .out files
pattern = re.compile(
    r"OpenMP Kmeans.*?\(number of threads: (?P<threads>\d+)\).+?"
    r"nloops =\s+\d+\s+\(total =\s+[\d.]+s\)\s+\(per loop =\s+(?P<loop_time>[\d.]+)s\)",
    re.DOTALL
)

# Extract data from all .out files
data = []
for file_name in os.listdir(results_dir):
    if file_name.endswith(".out"):
        lock_type = file_name.replace("kmeans_", "").replace(".out", "")
        file_path = os.path.join(results_dir, file_name)
        with open(file_path, "r") as f:
            content = f.read()
            matches = pattern.finditer(content)
            for match in matches:
                data.append({
                    "lock_type": lock_type,
                    "threads": int(match.group("threads")),
                    "loop_time": float(match.group("loop_time"))
                })

# Create a DataFrame
df = pd.DataFrame(data)

# Sort the data for proper plotting
df.sort_values(by=["lock_type", "threads"], inplace=True)

# Plot the data
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x="threads", y="loop_time", hue="lock_type", marker="o")

# Customize plot
plt.xlabel("Number of Threads")
plt.ylabel("Loop Time (s)")
plt.title("K-means Performance by Lock Type")
plt.legend(title="Lock Type")

# Force x-axis ticks to match unique thread values
unique_threads = sorted(df["threads"].unique())
plt.xticks(unique_threads, labels=unique_threads)

# Save and display the plot
plt.savefig(os.path.join(output_dir, "kmeans_performance.svg"))
plt.show()