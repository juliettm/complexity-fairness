import warnings
from sklearn.model_selection import KFold


import pandas as pd
from aif360.algorithms.inprocessing import GridSearchReduction
# fairness tools
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from fairlearn.reductions import DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# metrics
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_absolute_error
# training
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from datasets_processing.aif360datset import get_aif_dataset

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def convert_to_continuous(binary_output, prob_class_1, min_val, max_val, threshold):
    """Convert a binary output to a continuous output using probabilities and scale it to min_val-max_val."""
    # return min_val + (max_val - min_val) * (binary_output * prob_class_1 + (1 - binary_output) * (1 - prob_class_1))
    def scale_0_to_1(original_value):
        return (original_value - 0.5) / 0.5

    if binary_output == 0:
        min_val = min_val
        max_val = threshold - 1
        prob = scale_0_to_1(1 - prob_class_1)
        result = max_val - (max_val - min_val) * prob if prob_class_1 < 0.5 else threshold - 1
    else:
        min_val = threshold
        max_val = max_val
        prob = scale_0_to_1(prob_class_1)
        result = min_val + (max_val - min_val) * prob if prob_class_1 > 0.5 else threshold
    return result



def funct_average_predictive_value_difference(classified_metric):
    return 0.5 * (classified_metric.difference(classified_metric.positive_predictive_value)
                  + classified_metric.difference(classified_metric.false_omission_rate))


# compute fairness metrics for two classes
def fair_metrics(dataset, y_predicted, privileged_groups, unprivileged_groups):
    dataset_pred = dataset.copy()

    dataset_pred.labels = y_predicted

    classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    result = {'statistical_parity_difference': metric_pred.statistical_parity_difference(),
              'disparate_impact': metric_pred.disparate_impact(),
              'equal_opportunity_difference': classified_metric.equal_opportunity_difference(),
              'average_odds_difference': classified_metric.average_odds_difference(),
              'average_predictive_value_difference': funct_average_predictive_value_difference(classified_metric),
              'false_discovery_rate_difference': classified_metric.false_discovery_rate_difference()}

    return result, classified_metric, metric_pred

# optimización ...
# Data complexity... para cada subgrupo.. Author Ho. https://pypi.org/project/data-complexity/



def apply_c45_classifier(dataset, splits=10, mitigation=False, rand_state=1):


    lr_estimator = {'DT': DecisionTreeClassifier(random_state=42), #criterion='entropy',
                    'LR': LogisticRegression(random_state=42),
                    'KN': KNeighborsClassifier(n_neighbors=10)}

    # lr_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)  # With cost complexity pruning ccp_alpha=0.01,

    accuracies = []
    statistical_parity_difference = []
    disparate_impact = []
    equal_opportunity_difference = []
    average_odds_difference = []
    average_predictive_value_difference = []
    acc_m = []
    false_discovery_rate_difference = []


    df = dataset.copy()

    # Define the number of splits and the random state for reproducibility
    n_splits = 10
    random_state = 42

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # List to hold the train and test indices for each fold
    folds = []

    # Split the DataFrame
    for train_index, test_index in kf.split(df):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        folds.append((train_df, test_df))

    x_ds = dataset.drop(columns=['Y'])
    explanatory_variables = x_ds.columns
    outcome_label = 'Y'
    protected_att_name = 'A'

    dict_metrics = []
    for algorithm in lr_estimator.keys():
        print("Algorithm: ", algorithm)
        accuracies = []
        statistical_parity_difference = []
        disparate_impact = []
        equal_opportunity_difference = []
        average_odds_difference = []
        average_predictive_value_difference = []
        acc_m = []
        false_discovery_rate_difference = []

        # Example: Print the shape of train and test data for each fold
        for num, (train_df, test_df) in enumerate(folds):
            print("Fold: ", num + 1)
            seed = 100 + num

            df_tra = train_df
            df_tst_b = test_df

            X_train = df_tra[explanatory_variables]
            X_test = df_tst_b[explanatory_variables]
            y_train = df_tra[outcome_label]
            y_test_binary = df_tst_b[outcome_label]

            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_test_binary.reset_index(drop=True, inplace=True)

            column_predicted = outcome_label + '_predicted'
            target_variable = outcome_label  # if binary else target_variable_ordinal

            y_test = y_test_binary
            favorable_label = [1] #dataset.favorable_label_binary

            clf = lr_estimator[algorithm].fit(X_train, y_train)
            results_ = clf.predict(X_test)
            results = pd.DataFrame(results_, columns=[column_predicted])
            pred_prob = clf.predict_proba(X_test)
            results_cont = pd.DataFrame(results_, columns=[column_predicted])


            results_cm = cm(y_test, results)
            # print(results_cm)

            # Extract True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN)
            TP = results_cm[1, 1]
            TN = results_cm[0, 0]
            FP = results_cm[0, 1]
            FN = results_cm[1, 0]

            # Compute conditions
            is_FN_greater_than_TN = FN > TN
            is_FP_greater_than_TP = FP > TP

            acc = accuracy_score(y_test, results)
            # print(acc)
            accuracies.append(acc)

            ds_tra = get_aif_dataset(X_test, y_test, label=target_variable,
                                     protected_attribute_names=protected_att_name,
                                     privileged_classes=[[1]], #dataset.privileged_classes,
                                     favorable_classes=favorable_label)

            # self._privileged_groups = [{self._att: 1}]
            # self._unprivileged_groups = [{self._att: 0}]

            res, classm, predm = fair_metrics(ds_tra, results[column_predicted], [{'A': 0}] ,# dataset.privileged_groups,
                                              [{'A': 1}] )# dataset.unprivileged_groups)
            statistical_parity_difference.append(predm.statistical_parity_difference())
            disparate_impact.append(predm.disparate_impact())
            equal_opportunity_difference.append(classm.equal_opportunity_difference())
            average_odds_difference.append(classm.average_odds_difference())
            average_predictive_value_difference.append(funct_average_predictive_value_difference(classm))
            false_discovery_rate_difference.append(classm.false_discovery_rate_difference())
            acc_m.append(classm.accuracy())


        dict_metrics.append( {algorithm:  {'accuracies': accuracies,
                        'acc_m': acc_m,
                        'statistical_parity_difference': statistical_parity_difference,
                        'disparate_impact': disparate_impact,
                        'equal_opportunity_difference': equal_opportunity_difference,
                        'average_odds_difference': average_odds_difference,
                        'average_predictive_value_difference': average_predictive_value_difference,
                        'false_discovery_rate_difference': false_discovery_rate_difference
                        }
                              })

    # Create the DataFrame from the data
    df_list = []

    for classifier_data in dict_metrics:
        for classifier, metrics in classifier_data.items():
            temp_df = pd.DataFrame(metrics)
            temp_df['classifier'] = classifier  # Add classifier column
            df_list.append(temp_df)

    # Concatenate all into one DataFrame
    df_metrics = pd.concat(df_list, ignore_index=True)
    return df_metrics
