import os
import pandas as pd


def process_datasets(base_path):
    # Iterate over all folders in the base_path directory
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"Processing dataset in folder: {folder_path}")

            # Load the datasets
            X_train = pd.read_csv(os.path.join(folder_path, 'X_train.csv'), index_col=0)
            X_test = pd.read_csv(os.path.join(folder_path, 'X_test.csv'), index_col=0)
            y_train = pd.read_csv(os.path.join(folder_path, 'y_train.csv'), index_col=0)
            y_test = pd.read_csv(os.path.join(folder_path, 'y_test.csv'), index_col=0)

            x_t = X_train.merge(y_train, how='inner', left_index=True, right_index=True)
            x_ts = X_test.merge(y_test, how='inner', left_index=True, right_index=True)
            combined_df = pd.concat([x_t, x_ts])

            print(f"Combined dataset shape: {combined_df.shape}")
            combined_df.to_csv(os.path.join("data/syn_data/", f"{folder}.csv"), index=False)

            print(f"Datasets saved: {folder}")


if __name__ == "__main__":
    base_path = "datasets/combinations/biased/"
    process_datasets(base_path)
