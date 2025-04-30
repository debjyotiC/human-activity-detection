import os
import numpy as np
import pandas as pd

# Path to dataset
data_set_path = "Raw_data"
targets = os.listdir(data_set_path)

def compute_fall_features(data_frame):
    # Compute magnitudes
    accel_magnitude = np.sqrt(
        data_frame['accel-X']**2 + data_frame['accel-Y']**2 + data_frame['accel-Z']**2
    )
    gyro_magnitude = np.sqrt(
        data_frame['Gyro-X']**2 + data_frame['Gyro-Y']**2 + data_frame['Gyro-Z']**2
    )

    # Peak axis detection
    peak_axis = np.argmax([
        np.abs(data_frame['accel-X']).max(),
        np.abs(data_frame['accel-Y']).max(),
        np.abs(data_frame['accel-Z']).max()
    ])

    # Mean Crossing Rate for accel-Z
    mean_accel_Z = data_frame['accel-Z'].mean()
    shifted = data_frame['accel-Z'] - mean_accel_Z
    mcr_accel_Z = ((shifted[:-1] * shifted[1:]) < 0).sum()

    # Compute ext_features
    ext_features = {
        'max_accel_magnitude': accel_magnitude.max(),
        'min_accel_magnitude': accel_magnitude.min(),
        'range_accel_X': data_frame['accel-X'].max() - data_frame['accel-X'].min(),
        'mean_accel_Z': data_frame['accel-Z'].mean(),
        'std_gyro_X': data_frame['Gyro-X'].std(),
        'range_accel_Z': data_frame['accel-Z'].max() - data_frame['accel-Z'].min(),
        'mean_gyro_Y': data_frame['Gyro-Y'].mean(),
        'std_gyro_Z': data_frame['Gyro-Z'].std(),
        'std_accel_Z': data_frame['accel-Z'].std(),
        'range_accel_Y': data_frame['accel-Y'].max() - data_frame['accel-Y'].min(),
        'peak_axis': peak_axis,
        'mean_gyro_X': data_frame['Gyro-X'].mean(),
        'max_gyro_Z': data_frame['Gyro-Z'].max(),
        'std_gyro_Y': data_frame['Gyro-Y'].std(),
        'max_gyro_X': data_frame['Gyro-X'].max(),
    }

    return ext_features

# Collect all features
all_features = []

for target in targets:
    folder_path = os.path.join(data_set_path, target)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_excel(file_path).dropna()  # Read and clean the data
            features = compute_fall_features(df)
            features['Label'] = target  # Label = folder name
            all_features.append(features)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Convert features to DataFrame
features_df = pd.DataFrame(all_features)

# Save to CSV
output_path = "Features/features_all_classes.csv"
features_df.to_csv(output_path, index=False)
print(f"Features for all classes extracted and saved to '{output_path}'")
