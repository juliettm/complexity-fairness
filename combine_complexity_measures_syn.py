import os
import pandas as pd

cwd = os.getcwd()
print(cwd)
directory = cwd + '/results_measures_syn/'

# List to store dataframes
dfs = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        # Read the CSV file and append to the list
        df = pd.read_csv(file_path)
        dfs.append(df)

# Combine all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Specify the columns in the desired order
columns = ['F1v', 'N1', 'N2', 'N3', 'N4', 'LSC', 'T1', 'T2', 'T3', 'T4', 'C1', 'C2', 'L1', 'L2', 'L3', 'density', 'clustering_coefficient', 'hubs', 'att_value', 'att', 'dataset']


# Column renamed
combined_df = combined_df[columns]
combined_df.rename(columns={'clustering_coefficient': 'cls_coef'}, inplace=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('complexity_measures_syn.csv', index=False)

