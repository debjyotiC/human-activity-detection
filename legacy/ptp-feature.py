import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
file_path = "../Raw_Data/HAR_mag_8x1 1.xlsx"
df = pd.read_excel(file_path)

# 2. Filter for fall and jerk only
df_fj = df[df['label'].isin([1, 3])].copy()
label_map = {1: 'fall', 3: 'jerk'}
df_fj['label_named'] = df_fj['label'].map(label_map)

# 3. Extract feature matrix
X_fj = df_fj.drop(columns=['label', 'label_named']).values
labels_fj = df_fj['label_named'].values

# 4. Compute features
jerk_score = np.max(np.abs(np.diff(X_fj, axis=1)), axis=1)
ptp_score = np.ptp(X_fj, axis=1)
std_diff = np.std(np.diff(X_fj, axis=1), axis=1)

# 5. Boxplot: Max First-Difference
plt.figure()
data = [
    jerk_score[labels_fj == 'fall'],
    jerk_score[labels_fj == 'jerk']
]
plt.boxplot(data, labels=['fall', 'jerk'])
plt.ylabel('Max First-Difference')
plt.title('Max First-Difference by Label (fall vs. jerk)')
plt.show()

# 6. Boxplot: Std of First-Difference
plt.figure()
data_std = [
    std_diff[labels_fj == 'fall'],
    std_diff[labels_fj == 'jerk']
]
plt.boxplot(data_std, labels=['fall', 'jerk'])
plt.ylabel('Std of First-Difference')
plt.title('Std of First-Difference by Label (fall vs. jerk)')
plt.show()

# 7. Scatter: PTP vs. Max First-Difference
plt.figure()
plt.scatter(ptp_score[labels_fj == 'fall'], jerk_score[labels_fj == 'fall'], label='fall')
plt.scatter(ptp_score[labels_fj == 'jerk'], jerk_score[labels_fj == 'jerk'], label='jerk')
plt.xlabel('Peak-to-Peak Amplitude')
plt.ylabel('Max First-Difference')
plt.title('PTP vs. Max First-Difference (fall vs. jerk)')
plt.legend()
plt.show()
