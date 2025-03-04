import os
import pandas as pd

# Function to apply LaTeX bold (\textbf) to values not within [-0.1, 0.1]
def apply_latex_bold(value):
    if float(value) < -0.1 or float(value) > 0.1:
        return f"\\textbf{{{value}}}"
    else:
        return f"{value}"

# Path to your directory
directory_path = 'results/results_fairness_syn'


for method in ['DT', 'KN', 'LR']:
    df_joined = pd.DataFrame()
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(f'_quartiles_{method}.csv'):
            # Perform your operations here
            print(f"Processing file: {filename}")
            df = pd.read_csv(os.path.join(directory_path, filename), index_col=0)

            columns = {'accuracies': 'Acc', 'statistical_parity_difference': 'SP', 'equal_opportunity_difference': 'EO',
                       'false_discovery_rate_difference': 'PP'}
            selected_columns_df = df[columns.keys()]
            selected_columns_df.rename(columns=columns, inplace=True)
            # Rows to drop
            rows_to_drop = ['count', 'min', 'max', '25%', '50%', '75%']
            # Dropping rows by index names
            selected_columns_df.drop(rows_to_drop, axis=0, inplace=True)
            selected_columns_df['Dataset'] = filename[:-len(f'_quartiles_{method}.csv')] #filename.split('_')[0]
            selected_columns_df['Att'] = 'Y'
            selected_columns_df.reset_index(inplace=True)
            final_df = selected_columns_df[['Dataset', 'Att', 'index', 'Acc', 'SP', 'EO', 'PP']]
            df_joined = pd.concat([df_joined, final_df], ignore_index=True)

    # all_df = pd.merge(df_joined, df_joined_pr, on=['Dataset', 'Att', 'index'], suffixes=('', '_pr'))
    all_df = df_joined.copy()
    all_df.sort_values(by=['Dataset', 'Att', 'index'], inplace=True)

    all_df.rename(columns={'index': 'Measure'}, inplace=True)

    # Apply the function element-wise
    latex_df = all_df[['SP', 'EO', 'PP']].applymap(apply_latex_bold)

    # Convert to LaTeX table
    # latex_table = latex_df.to_latex(escape=False)


    latex_code_classification = all_df.to_latex(index=False, float_format="%.2f", escape=False)

    print(latex_code_classification)

    all_df.to_csv(f'results/combined_quartiles_fairness_syn_combination_{method}.csv', index=False)
