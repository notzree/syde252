import matplotlib.pyplot as plt
import pandas as pd

# Load the metrics data from the CSV file
csv_file_path = "output_csv/implant_metrics.csv"
metrics_df = pd.read_csv(csv_file_path)
metrics_df = pd.read_csv(csv_file_path)

# Creating 3D plots for each metric
# fig = plt.figure(figsize=(12, 10))
# metrics_columns = ["snr", "spectral_distortion", "pesq", "mfcc_similarity", "freq_representation"]

# for i, column in enumerate(metrics_columns):
#     ax = fig.add_subplot(projection="3d")
#     ax.scatter(metrics_df["num_bands"], metrics_df["cutoff_freq"], zs metrics_df[column], label=column)
#     ax.set_title(f"{column} vs. Number of Bands and Cutoff Frequency")
#     ax.set_xlabel("Number of Bands")
#     ax.set_ylabel("Cutoff Frequency")
#     ax.set_zlabel
#     ax.grid(True)

# ax.view_init(elev=20.0, azim=-35, roll=0)
# plt.tight_layout()  # Adjust layout
# plt.savefig("output_csv/3d_metrics_plot.png")
# plt.show()

# Creating plots for each metric against number of bands and cutoff frequency
metrics_columns = ["snr", "spectral_distortion", "pesq", "mfcc_similarity", "freq_representation"]

plt.figure(figsize=(12, 10))
for column in metrics_columns:
    plt.subplot(3, 2, metrics_columns.index(column) + 1)
    plt.scatter(metrics_df["num_bands"], metrics_df[column], label=column)
    plt.title(f"{column} vs. Number of Bands")
    plt.xlabel("Number of Bands")
    plt.ylabel(column)
    plt.grid(True)
    plt.legend()

plt.tight_layout()  # Adjust layout
plt.savefig("output_csv/metrics_plot.png")
plt.show()
