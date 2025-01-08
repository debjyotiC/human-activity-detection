import pandas as pd
import os
import sys
import numpy as np
import argparse
import shutil

# Step 1: Define the main folder containing the subfolders with Excel files
main_folder_path = "Dataset/Raw_data"  # Update with your main folder path
output_base_path = "Features"  # Update with your desired output base path

os.makedirs(output_base_path, exist_ok=True)


# Step 2: Function to calculate RMS
def calculate_rms(data):
    return np.sqrt((data ** 2).mean())


def median_absolute_deviation(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    return np.median(deviations)


# Step 3: Iterate through each subfolder in the main folder
for root, dirs, files in os.walk(main_folder_path):
    for filename in files:
        if filename.endswith(".xlsx"):  # Process only Excel files
            file_path = os.path.join(root, filename)

            # Load the Excel file into a DataFrame
            df = pd.read_excel(file_path).dropna()

            # Calculate energy (sum of squares divided by number of values) for accelerometer axes separately
            energy_accel_X = (df['accel-X'] ** 2).sum() / len(df)
            energy_accel_Y = (df['accel-Y'] ** 2).sum() / len(df)
            energy_accel_Z = (df['accel-Z'] ** 2).sum() / len(df)

            # Calculate RMS for accelerometer axes separately
            rms_accel_X = np.sqrt((df['accel-X'] ** 2).mean())
            rms_accel_Y = np.sqrt((df['accel-Y'] ** 2).mean())
            rms_accel_Z = np.sqrt((df['accel-Z'] ** 2).mean())

            # Calculate SMA (Signal Magnitude Area) for accelerometer data
            sma_accel = int(((np.abs(df['accel-X']) + np.abs(df['accel-Y']) + np.abs(df['accel-Z'])).mean()) / 64)

            # Calculate energy (sum of squares divided by number of values) for gyroscope axes separately
            energy_gyro_X = (df['Gyro-X'] ** 2).sum() / len(df)
            energy_gyro_Y = (df['Gyro-Y'] ** 2).sum() / len(df)
            energy_gyro_Z = (df['Gyro-Z'] ** 2).sum() / len(df)

            # Calculate RMS for gyroscope axes separately
            rms_gyro_X = np.sqrt((df['Gyro-X'] ** 2).mean())
            rms_gyro_Y = np.sqrt((df['Gyro-Y'] ** 2).mean())
            rms_gyro_Z = np.sqrt((df['Gyro-Z'] ** 2).mean())

            # Calculate Signal Magnitude Area (SMA) for gyroscope data
            sma_gyro = int(((np.abs(df['Gyro-X']) + np.abs(df['Gyro-Y']) + np.abs(df['Gyro-Z'])).mean()) / 64)

            # Calculate mean, min, max, std, energy, RMS, range, median, and median absolute deviation for magnitude_accel
            df['magnitude_accel'] = (
                        (np.sqrt(df['accel-X'] ** 2 + df['accel-Y'] ** 2 + df['accel-Z'] ** 2)) / 64).astype(int)
            mean_magnitude_accel = (df['magnitude_accel']).astype(int).mean()  # Divide by 32 and convert to int
            min_magnitude_accel = (df['magnitude_accel']).astype(int).min()
            max_magnitude_accel = (df['magnitude_accel']).astype(int).max()
            std_magnitude_accel = (df['magnitude_accel']).astype(int).std()
            energy_magnitude_accel = ((df['magnitude_accel']) ** 2).sum() / len(df)
            rms_magnitude_accel = np.sqrt(((df['magnitude_accel']) ** 2).mean()).astype(int)
            range_magnitude_accel = max_magnitude_accel - min_magnitude_accel
            median_magnitude_accel = (df['magnitude_accel']).astype(int).median()
            mad_magnitude_accel = median_absolute_deviation(df['magnitude_accel']).astype(int)

            # Calculate mean, min, max, std, energy, RMS, range, median, and median absolute deviation for magnitude_gyro
            df['magnitude_gyro'] = ((np.sqrt(df['Gyro-X'] ** 2 + df['Gyro-Y'] ** 2 + df['Gyro-Z'] ** 2)) / 64).astype(
                int)
            mean_magnitude_gyro = (df['magnitude_gyro']).astype(int).mean()  # Divide by 32 and convert to int
            min_magnitude_gyro = (df['magnitude_gyro']).astype(int).min()
            max_magnitude_gyro = (df['magnitude_gyro']).astype(int).max()
            std_magnitude_gyro = (df['magnitude_gyro']).astype(int).std()
            energy_magnitude_gyro = ((df['magnitude_gyro']) ** 2).sum() / len(df)
            rms_magnitude_gyro = np.sqrt(((df['magnitude_gyro']) ** 2).mean()).astype(int)
            range_magnitude_gyro = max_magnitude_gyro - min_magnitude_gyro
            median_magnitude_gyro = (df['magnitude_gyro']).astype(int).median()
            mad_magnitude_gyro = median_absolute_deviation(df['magnitude_gyro']).astype(int)

            # Calculate median for accelerometer axes
            median_accel_X = df['accel-X'].median()
            median_accel_Y = df['accel-Y'].median()
            median_accel_Z = df['accel-Z'].median()

            # Calculate median for gyroscope axes
            median_gyro_X = df['Gyro-X'].median()
            median_gyro_Y = df['Gyro-Y'].median()
            median_gyro_Z = df['Gyro-Z'].median()

            # Example usage:
            # Assuming df is your DataFrame containing accelerometer and gyroscope data

            # Calculate MAD for accelerometer axes
            mad_accel_X = median_absolute_deviation(df['accel-X'])
            mad_accel_Y = median_absolute_deviation(df['accel-Y'])
            mad_accel_Z = median_absolute_deviation(df['accel-Z'])

            # Calculate MAD for gyroscope axes
            mad_gyro_X = median_absolute_deviation(df['Gyro-X'])
            mad_gyro_Y = median_absolute_deviation(df['Gyro-Y'])
            mad_gyro_Z = median_absolute_deviation(df['Gyro-Z'])

            # Calculate peak acceleration (maximum of absolute values) for accelerometer and gyroscope data
            peak_accel_X = df['accel-X'].abs().max()
            peak_accel_Y = df['accel-Y'].abs().max()
            peak_accel_Z = df['accel-Z'].abs().max()
            peak_gyro_X = df['Gyro-X'].abs().max()
            peak_gyro_Y = df['Gyro-Y'].abs().max()
            peak_gyro_Z = df['Gyro-Z'].abs().max()
            peak_accel_magnitude = df['magnitude_accel'].abs().max()
            peak_gyro_magnitude = df['magnitude_gyro'].abs().max()

            # Apply np.diff to magnitude_gyro
            df['jerk_magnitude_accel'] = np.diff(df['magnitude_accel'], prepend=df['magnitude_accel'].iloc[0])
            rms_jerk_magnitude_accel = (np.sqrt((df['jerk_magnitude_accel'] ** 2).mean())).astype(int)
            # Apply np.diff to magnitude_gyro
            df['jerk_magnitude_gyro'] = np.diff(df['magnitude_gyro'], prepend=df['magnitude_gyro'].iloc[0])
            rms_jerk_magnitude_gyro = (np.sqrt((df['jerk_magnitude_gyro'] ** 2).mean())).astype(int)

            # Calculate range for accelerometer and gyroscope axes
            range_accel_X = df['accel-X'].max() - df['accel-X'].min()
            range_accel_Y = df['accel-Y'].max() - df['accel-Y'].min()
            range_accel_Z = df['accel-Z'].max() - df['accel-Z'].min()
            range_gyro_X = df['Gyro-X'].max() - df['Gyro-X'].min()
            range_gyro_Y = df['Gyro-Y'].max() - df['Gyro-Y'].min()
            range_gyro_Z = df['Gyro-Z'].max() - df['Gyro-Z'].min()

            # Combine all features into a single DataFrame with the specified sequence
            statistics_df = pd.DataFrame({
                'mean_magnitude_accel': [mean_magnitude_accel],
                'mean_magnitude_gyro': [mean_magnitude_gyro],
                'std_magnitude_accel': [std_magnitude_accel],
                'std_magnitude_gyro': [std_magnitude_gyro],
                'min_magnitude_accel': [min_magnitude_accel],
                'min_magnitude_gyro': [min_magnitude_gyro],
                'max_magnitude_accel': [max_magnitude_accel],
                'max_magnitude_gyro': [max_magnitude_gyro],
                'range_magnitude_accel': [range_magnitude_accel],
                'range_magnitude_gyro': [range_magnitude_gyro],
                'median_magnitude_accel': [median_magnitude_accel],
                'median_magnitude_gyro': [median_magnitude_gyro],
                'mad_magnitude_accel': [mad_magnitude_accel],
                'mad_magnitude_gyro': [mad_magnitude_gyro],
                'rms_magnitude_accel': [rms_magnitude_accel],
                'rms_magnitude_gyro': [rms_magnitude_gyro],
                'rms_jerk_magnitude_accel': [rms_jerk_magnitude_accel],
                'rms_jerk_magnitude_gyro': [rms_jerk_magnitude_gyro],
                'sma_accel': [sma_accel],
                'sma_gyro': [sma_gyro],

            })


            # Get the relative path of the current file with respect to the main folder
            relative_path = os.path.relpath(root, main_folder_path)

            # Create the corresponding subfolder structure in the output base path
            output_folder = os.path.join(output_base_path, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            # Save the DataFrame with statistics to a new Excel file in the output folder
            file_name, file_extension = os.path.splitext(filename)
            output_file_path = os.path.join(output_folder, f"{file_name}_features.xlsx")

            # Convert all columns to integer type
            statistics_df = statistics_df.astype(int)

            statistics_df.to_excel(output_file_path, index=False)

    print(f"Feature Extraction in progress")
print("Features Extraction completed")