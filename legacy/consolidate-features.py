import pandas as pd
import os

# Step 1: Define the main folder containing the subfolders with modified Excel files
main_folder_path = "../Features"  # Update with your main folder path
output_folder = "excel_files"  # Update with your desired output folder path
merged_file_path = "../"
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize a dictionary for specific label assignments
specific_label_assignments = {
    'backward_fall': 1,
    'climbing_up_stairs': 0,
    'front_fall': 1,
    'going_down_stairs': 0,
    'jogging': 0,
    'jumping': 0,
    'lateral': 1,
    'lying': 0,
    'misc': 0,
    'sitting': 0,
    'standing': 0,
    'walking': 0
}

# Step 2: Iterate through each subfolder in the main folder
for root, dirs, files in os.walk(main_folder_path):
    for subdir in dirs:
        subfolder_path = os.path.join(root, subdir)

        # Initialize an empty list to store DataFrames
        dfs = []

        # Iterate through each file in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".xlsx"):  # Process only Excel files
                file_path = os.path.join(subfolder_path, filename)

                # Load the Excel file into a DataFrame
                df = pd.read_excel(file_path)

                # Append the DataFrame to the list
                dfs.append(df)

        # Check if there are any DataFrames to concatenate
        if dfs:
            # Concatenate all DataFrames into a single DataFrame
            combined_df = pd.concat(dfs, ignore_index=True)

            # Add a 'label' column with the current label value
            label = specific_label_assignments.get(subdir, -1)  # Use -1 for any unmatched subdir
            combined_df['label'] = label

            # Define the output file path for the combined Excel file
            output_file_path = os.path.join(output_folder, f"{subdir}.xlsx")

            # Write the combined DataFrame to a new Excel file
            combined_df.to_excel(output_file_path, index=False)

        print("Label assignment in progress")

        # Print the label assignment
        # print(f"Label {label} assigned to {output_file_path}")

# Step 3: Initialize an empty list to store the final DataFrames
final_dfs = []

# Step 4: Iterate through each generated Excel file in the output folder
for filename in os.listdir(output_folder):
    if filename.endswith(".xlsx"):  # Process only Excel files
        file_path = os.path.join(output_folder, filename)

        # Load the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Append the DataFrame to the list
        final_dfs.append(df)

# Step 5: Concatenate all final DataFrames into a single DataFrame
final_combined_df = pd.concat(final_dfs, ignore_index=True)

# Step 6: Define the output file path for the final combined Excel file
final_output_file_path = os.path.join(merged_file_path, "fall_20_feature_train_data.xlsx")

# Step 7: Write the final combined DataFrame to a new Excel file
final_combined_df.to_excel(final_output_file_path, index=False)

print("Label assigment completed")

# print(f"All combined statistics saved to {final_output_file_path}")

# Optional: Save label assignments to a text file
label_assignments_file = os.path.join(output_folder, "label_assignments.txt")
with open(label_assignments_file, 'w') as f:
    for subdir, label in specific_label_assignments.items():
        f.write(f"{subdir}: Label {label}\n")

print(f"Label assignments saved to {label_assignments_file}")

# Remove the directory only if it already exists
# if os.path.exists(main_folder_path):
#     shutil.rmtree(main_folder_path)
#     print(f"Directory {main_folder_path} deleted successfully.")
# else:
#     print(f"Directory {main_folder_path} does not exist.")
#
# if os.path.exists(output_folder):
#     shutil.rmtree(output_folder)
#     print(f"Directory {output_folder} deleted successfully.")
# else:
#     print(f"Directory {output_folder} does not exist.")