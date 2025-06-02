import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the uploaded Excel file
file_path = "Raw_Data/HAR_mag_8x1 1.xlsx"
df = pd.read_excel(file_path)

plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='Diff', data=df, palette='Set2')

plt.title("Distribution of Diff (sma_gyro - sma_accel) by Activity Label")
plt.xlabel("Label (0-active, 1-fall, 2-idle, 3-jerk)")
plt.ylabel("Diff (sma_gyro - sma_accel)")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig("images/box_plot.png", dpi=300)
plt.show()
