import os
import numpy as np
import pandas as pd

# Path to dataset
data_set_path = "Raw_data"
targets = os.listdir(data_set_path)

def compute_selected_features(data_frame):
    # Extract accel and gyro arrays
    accel = data_frame[['accel-X', 'accel-Y', 'accel-Z']].values
    gyro = data_frame[['Gyro-X', 'Gyro-Y', 'Gyro-Z']].values

    # Compute magnitudes
    accel_magnitude = np.linalg.norm(accel, axis=1)
    gyro_magnitude = np.linalg.norm(gyro, axis=1)

    # Jerk = diff of accel or gyro
    jerk_accel = np.diff(accel, axis=0)
    jerk_gyro = np.diff(gyro, axis=0)

    # SMA = sum of abs values over all axes divided by window length
    sma_accel = np.sum(np.abs(accel)) / len(accel)
    sma_gyro = np.sum(np.abs(gyro)) / len(gyro)

    # Additional features
    jerk_score = np.max(np.abs(jerk_accel))
    ptp_score = np.ptp(accel)
    std_diff = np.std(jerk_accel)

    features = {
        'range_magnitude_accel': np.ptp(accel_magnitude),
        'range_magnitude_gyro': np.ptp(gyro_magnitude),
        'median_magnitude_accel': np.median(accel_magnitude),
        'median_magnitude_gyro': np.median(gyro_magnitude),
        'mean_abs_jerk_magnitude_accel': np.mean(np.linalg.norm(jerk_accel, axis=1)),
        'mean_abs_jerk_magnitude_gyro': np.mean(np.linalg.norm(jerk_gyro, axis=1)),
        'sma_accel': sma_accel,
        'sma_gyro': sma_gyro,
        'jerk_score': jerk_score,
        'ptp_score': ptp_score,
        'std_diff': std_diff,
    }

    return features

# Collect all features
all_features = []

for target in targets:
    folder_path = os.path.join(data_set_path, target)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_excel(file_path).dropna()
            features = compute_selected_features(df)
            features['Label'] = target
            all_features.append(features)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Convert features to DataFrame
features_df = pd.DataFrame(all_features)

# Save to CSV
output_path = "Features/features_selected.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
features_df.to_csv(output_path, index=False)
print(f"Selected features extracted and saved to '{output_path}'")
