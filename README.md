

# Synthetic dataset generation

We used the framework [Bias on Demand](https://github.com/rcrupiISP/BiasOnDemand) to generate different datasets with various bias. Table 1 presents the labels of the datasets representing different bias scenarios (Label), their description (Explanation), the parameter (P) used to produce the desired bias, the values of the parameter used in each scenario (Values), and the number of variations of the datasets we generated according to the number of parameters used in each case (\#). The total number of generated datasets is 73, with S1A being the one without any associated bias. The rest of the datasets were generated with different values of a parameter and are labeled with the respective index of the array in column Values of Table 1 (Example: the combinations of S1F are S1F1, S1F2, S1F3, and S1F4 for the values 0.3, 0.5, 0.7, and 0.9, respectively).


### Table 1: Description of the synthetic datasets generated with different bias scenarios

| **Label** | **Explanation**                         | **P**        | **Values** | **#** |
|-----------|----------------------------------------|-------------|------------|----|
| S1A       | No bias present                       |             |            | 1  |
| S1B       | Measurement bias in $R$               | $l\_m$      | `[0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]` | 12 |
| S1C       | $R$ omitted from the dataset          | $l\_o$      | True       | 1  |
| S1D       | Undersampling of group $A = 1$        | $p\_u$      | `[0.003, 0.006, 0.008, 0.01, 0.1, 0.3, 0.5]` | 7  |
| S1E       | Measurement bias in label $Y$        | $l\_m\_y$   | `[0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]` | 12 |
| S1F       | Conditional undersampling on $R$     | $p\_u$      | `[0.3, 0.5, 0.7, 0.9]` | 4  |
|           |                                       | $l\_r$      | True       |    |
| S2A       | Historical bias on $R$               | $l\_h\_y$   | `[0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]` | 12 |
| S3A       | Historical bias on $Y$               | $l\_y$      | `[0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]` | 12 |
| S4A       | Historical bias on $Q$               | $l\_h\_q$   | `[0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]` | 12 |


Running the file `generate_ds_synthetic.py` the datasets will be generated in `datasets/combinations/biased`
The initial code generate a folder for each dataset divided in training and test subsets, we are combining them in a single dataset saved in `data/syn_data/` executing `combine_ds_synthetic.py`

# Dataset characterization

We will characterize the generated data using the parameters characterizing each bias scenario (Properties dataset), the absolute differences in complexity between the privileged and unprivileged groups (Complexity dataset) and the fairness metrics results for each problem (Fairness dataset). In the following datasets each row is a different problem corresponding to each of the synthetic dataset generated.

## Properties dataset

Running the file `generate_ds_propertied.py` the properties dataset will be generated in `datasets/analysis/XXX`


## Complexity dataset

The code for computing the complexity measures is in the file `complexity_measures.py`
Using `compute_measures_syn.py` will compute the complexity measures for privileged and unprivileged groups and the result will be saved in `results/results_measures_syn`  
Lastly, using `combine_complexity_measures_syn.py` we will obtain the dataset characterizing each dataset generated with the complexity measures for privileged and unprivileged.

### Complexity dataset analysis

We are using `analysis_complexity_diff_synthetic.ipynb` to analyze the complexity differences between privileged and unprivileged groups.
The dataset of the differences is saved in `results/df_complexity_measures_sync_diff.csv`
Here we are plotting:
- Figure 1: Distribution of absolute differences in various complexity metrics when comparing privileged and
unprivileged groups.
- Figure 2: Distribution of complexity metrics absolute differences by each bias scenario.
- Figure 4: Distribution of complexity metric values across bias scenarios. 
- Figure 5: Two-dimensional MDS visualization of synthetic datasets complexity metrics absolute differences
colored by bias scenario.

We used the `results/df_complexity_measures_sync_diff.csv` dataframe for the Figure 3: Evolution of complexity metrics difference by bias parameter (`plot_complexity_diff_evolution_scenario.py`) 

## Fairness dataset

We are using `compute_fairness_syn.py` to compute the fairness measures for all datasets: Statistical Parity (SP), Equal Opportunity (EO) and Predictive Parity (PP) are computed using Logistic regression (LR), Decision Trees (DT) and Nearest Neighbours (KN). The result are stored in the folder `results/results_fairness_syn`.
Then we combine all the result obtaining a file for each method using `combine_fairness_metrics_syn.py` we only keep the mean and standard deviation for each measure.

### Fairness dataset analysis

We are using `analysis_fairness_diff_synthetic.ipynb` to analyze the fairness measures we computed for each dataset.
We use the previous files to obtain the fairness measures combined in one table (file: `results/df_fairness_measures_sync.csv`) 

Here we are plotting:
- Figure 6: Distribution of fairness metrics result.
- Figure 7: Parallel coordinates visualization of fairness metrics across different bias scenarios.
- Figure 9: MDS visualization of synthetic and real-world datasets. 

We also load the complexity dataframe and merge with the fairness dataframe to make a visualization using MDS of the dataset colored by fairness results (Figure 7)
Lastly, we load the real-world data to make a combined visualization of the synthetic and rel-world datasets.

# Association rules

We are using `analysis_fairness&complexity_diff_synthetic.ipynb` to compute the association rules relating complexity differences and fairness metrics.

- Figure 8: Association rules. 

# Real-world problems application

The same procedure of the synthetic datasets was followed to characterize the real-world datasets:
- For the complexity measures results: The folder `results/result_measures` contain all the results of complexity metrics for privileged and unprivileged values. The file `results/complexity_measures.csv` contain the combination of the results for all datasets and `df_complexity_measures.csv` is the formatted dataset.
- For the fairness results: The folder `results/results_fairness` contain the results of the fairness metrics for each dataset. The files `results/combined_quartiles_fairness_DT.csv`, `results/combined_quartiles_fairness_KN.csv`and `results/combined_quartiles_fairness_LR.csv` contain the results combined by each technique used and 

## Analysis of real-world datasets

We are using `analysis_fairness&complexity_diff.ipynb` to analyze how the rules obtained using synthetic data apply to real-world data.




