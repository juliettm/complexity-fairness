import pandas as pd
import matplotlib.pyplot as plt

def extract_scenario_info(index):
    """Extract scenario type and parameter value from index"""
    base_scenario = index[:2]
    if len(index) > 2:
        variation = index[2]
    else:
        variation = 'A'

    if '_' in index:
        param_value = float(index.split('_')[-1])
    elif len(index) > 3 and index[3:].replace('.', '').isdigit():
        param_value = float(index[3:])
    else:
        param_value = 0

    return base_scenario, variation, param_value

def prepare_data(complexity_df):
    """Prepare and combine the datasets"""
    scenario_info = [extract_scenario_info(idx) for idx in complexity_df.index]
    complexity_df['base_scenario'] = [info[0] for info in scenario_info]
    complexity_df['variation'] = [info[1] for info in scenario_info]
    complexity_df['param_value'] = [info[2] for info in scenario_info]
    complexity_df['scenario_full'] = complexity_df['base_scenario'] + complexity_df['variation']
    return complexity_df


def plot_complexity_by_scenario(df, metrics, output_filename=None):
    """Create plots organized by scenario with all metrics and improved legend"""
    # Define scenarios and their styles

    scenario_styles = {
        'S1A': {'color': '#d62728', 'label': 'Scenario S1A: No Bias'},
        'S1B': {'color': '#bcbd22', 'label': 'Scenario S1B: Measurement Bias on R'},
        'S1C': {'color': '#2ca02c', 'label': 'Scenario S1C: R Omitted'},
        'S1D': {'color': '#1f77b4', 'label': 'Scenario S1D: Undersampling'},
        'S1E': {'color': '#9467bd', 'label': 'Scenario S1E: Measurement Bias on Y'},
        'S1F': {'color': '#8c564b', 'label': 'Scenario S1F: Conditional undersampling on R'},
        'S2A': {'color': '#e377c2', 'label': 'Scenario S2A: Historical Bias on R'},
        'S3A': {'color': '#ff7f0e', 'label': 'Scenario S3A: Historical Bias on Y'},
        'S4A': {'color': '#7f7f7f', 'label': 'Scenario S4A: Historical Bias on Q'}
    }

    # Define line styles for metrics with distinct patterns
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1))]
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '8', '*']

    # Create subplots for each scenario
    n_scenarios = len(scenario_styles)
    n_cols = 3
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    # Create figure
    fig = plt.figure(figsize=(20, 6 * n_rows))  # Increased height for legend space
    # plt.suptitle('Complexity Metrics by Scenario', fontsize=16, y=0.95)

    # Create subplot grid for the actual plots
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    gs.update(top=0.85)  # Adjust top margin to make room for legend

    # Create a plot for each scenario
    for idx, (scenario_full, style) in enumerate(scenario_styles.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Set y-axis limits from 0 to 1 for all plots
        ax.set_ylim(-0.1, 0.7)

        # Get data for this scenario
        scenario_data = df[df['scenario_full'] == scenario_full].sort_values('param_value')

        if not scenario_data.empty:
            # Plot each metric
            for metric_idx, metric in enumerate(metrics):
                line_style = line_styles[metric_idx % len(line_styles)]
                marker = markers[metric_idx % len(markers)]

                ax.plot(scenario_data['param_value'],
                        scenario_data[metric],
                        label=metric,
                        linestyle=line_style,
                        marker=marker,
                        markersize=6,
                        linewidth=2,
                        alpha=0.7)

        ax.set_xlabel('Parameter Value', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(f'{style["label"]}', fontsize=14)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=9)

        # Get legend handles and labels from the first plot
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Create legend with more columns and place it at the top
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.98),
               ncol=10,  # Increased number of columns
               fontsize=14,
               borderaxespad=0,
               framealpha=0.9,
               edgecolor='black',
               columnspacing=1.0,  # Increased space between columns
               handletextpad=0.5,  # Reduced space between handle and text
               bbox_transform=plt.gcf().transFigure)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjusted rect to account for legend

    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)

    plt.show()


def main():
    # Read the datasets
    complexity_df = pd.read_csv('results/df_complexity_measures_sync_diff.csv', index_col='complete_name') #complexity_df_complete_syn
    complexity_df.drop(columns=['base_experiment'], inplace=True)

    # Prepare data
    df = prepare_data(complexity_df)

    # Define metrics - you might want to select fewer metrics for better readability
    metrics = ['F1v', 'N1', 'N2', 'N3', 'N4', 'LSC', 'T1',
               'C1', 'C2', 'L1', 'L2', 'L3', 'density', 'cls_coef']

    # Create plots
    plot_complexity_by_scenario(df, metrics, 'plots/cm_lines_by_scenario_diff.pdf')


if __name__ == "__main__":
    main()