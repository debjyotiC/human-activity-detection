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

    # Zero Crossing Rate for accel-Z
    zcr_accel_Z = ((data_frame['accel-Z'][:-1] * data_frame['accel-Z'][1:]) < 0).sum()

    # Compute ext_features
    ext_features = {
        # Accelerometer ext_features
        'max_accel_magnitude': accel_magnitude.max(),
        'min_accel_magnitude': accel_magnitude.min(),
        'mean_accel_Z': data_frame['accel-Z'].mean(),
        'std_accel_Z': data_frame['accel-Z'].std(),
        'range_accel_X': data_frame['accel-X'].max() - data_frame['accel-X'].min(),
        'range_accel_Y': data_frame['accel-Y'].max() - data_frame['accel-Y'].min(),
        'range_accel_Z': data_frame['accel-Z'].max() - data_frame['accel-Z'].min(),

        # Gyroscope ext_features
        'max_gyro_magnitude': gyro_magnitude.max(),
        'mean_gyro_X': data_frame['Gyro-X'].mean(),
        'mean_gyro_Y': data_frame['Gyro-Y'].mean(),
        'std_gyro_X': data_frame['Gyro-X'].std(),
        'std_gyro_Y': data_frame['Gyro-Y'].std(),
        'std_gyro_Z': data_frame['Gyro-Z'].std(),
        'max_gyro_X': data_frame['Gyro-X'].max(),
        'max_gyro_Y': data_frame['Gyro-Y'].max(),
        'max_gyro_Z': data_frame['Gyro-Z'].max(),

        # Peak axis and ZCR
        'peak_axis': peak_axis,
        'zcr_accel_Z': zcr_accel_Z,
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
