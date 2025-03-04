import os
import sys
import warnings

import numpy as np
import pandas as pd
from utils.classification_syn import apply_c45_classifier

warnings.filterwarnings("ignore")

to_insert = os.getcwd()
# to import utils
sys.path.append(to_insert)



def run_experiment(dataset, mitigation, outcome, rand_state):
    if outcome == 'binary':
        return apply_c45_classifier(dataset, splits=10, mitigation=mitigation, rand_state=rand_state)
    else:
        raise AssertionError('not a valid outcome: ', outcome)



stats_results = True
outcome = 'binary'
mit = False


for file in os.listdir("data/syn_data/"): # all combinations: biased_complexity_combinations
    if file.endswith(".csv"):
        file_name = os.path.splitext(file)[0]
        data = pd.read_csv("data/syn_data/{}".format(file), sep=',', header=0)

        y = data['Y'].to_numpy()
        y = y.astype(int)
        x_ds = data.drop(columns=['Y'])
        X = data.drop(columns=['Y']).to_numpy()
        # Convert X to a DataFrame to easily handle column operations
        X_df = pd.DataFrame(X, columns=x_ds.columns)

        appended_results = []
        rand_state = 1
        results = run_experiment(data, mit, outcome, rand_state)
        results.replace([np.inf, -np.inf], np.nan, inplace=True)
        results['seed'] = rand_state
        appended_results.append(results)
        # Concatenate all results and print them
        appended_data = pd.concat(appended_results)
        if stats_results:
            # Para imprimir los resultados por quartiles
            for cls in appended_data['classifier'].unique():
                stats = appended_data[appended_data['classifier'] == cls].describe()
                stats.to_csv('results/results_fairness_syn/{name}_quartiles_{cls}.csv'.format(name=file_name, cls=cls))


        appended_data.to_csv('results/results_fairness_syn/{name}.csv'.format(name=file_name), index=False)


