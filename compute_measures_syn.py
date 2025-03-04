from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from complexity_measures import precompute_fx, ft_F1, ft_F2, ft_F3, ft_F4, network_measures, ft_F1v, ft_N1, ft_N2, ft_N3, LSC, ft_T2T3T4, C12, L123, T1, ft_F2, ft_F4, ft_F3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import gc
import os

# measures = ['att_value','F1v', 'N1', 'N2', 'N3', 'N4', 'LSC', 'T1', 'T2', 'T3', 'T4', 'C1', 'C2', 'L1', 'L2', 'L3', 'density', 'clustering_coefficient', 'hubs', 'complete_name']


def split_data_by_sensitive_attribute(X, y, sensitive_attribute_name):
    sensitive_attribute_index = sensitive_attribute_name
    sensitive_values = X[:, sensitive_attribute_index]
    unique_sensitive_values = np.unique(sensitive_values)

    subsets = {}
    for value in unique_sensitive_values:
        mask = sensitive_values == value
        X_subset = np.delete(X[mask], sensitive_attribute_index, axis=1)
        y_subset = y[mask]
        subsets[value] = (X_subset, y_subset)

    return subsets



measures_names = ['F1v', 'N1', 'N2', 'N3', 'LSC', 'T1', 'T2', 'T3', 'T4', 'C1', 'C2', 'L1', 'L2', 'L3', 'N4', 'density', 'clustering_coefficient', 'hubs', 'att_value', 'att', 'dataset'] #,
# measures_names = ['F1', 'F1v', 'F2', 'F3', 'F4', 'att_value', 'att', 'dataset']

for file in os.listdir("data/biased_complexity_combinations/"):
    file_name = os.path.splitext(file)[0]
    print('Computing complexity measures for {}'.format(file))
    data = pd.read_csv("data/biased_complexity_combinations/{}".format(file), sep=',', header=0)
    y = data['Y'].to_numpy()
    y = y.astype(int)
    x_ds = data.drop(columns=['Y'])
    X = data.drop(columns=['Y']).to_numpy()

    # Convert X to a DataFrame to easily handle column operations
    X_df = pd.DataFrame(X, columns=x_ds.columns)


    # Compute the index of the sensitive attribute
    sensitive_attribute_index = X_df.columns.get_loc('A')

    # Convert X_df back to a numpy array if needed
    X = X_df.to_numpy()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_scaled = scaler.transform(X)
    std_scaler = StandardScaler().fit(X)
    x_sensitive = X[:, sensitive_attribute_index].copy()
    X_standard_scaled = std_scaler.transform(X)
    X_standard_scaled[:, sensitive_attribute_index] = x_sensitive


    # to compute the subsets
    subsets = split_data_by_sensitive_attribute(X, y, sensitive_attribute_index)
    subsets_scaled = split_data_by_sensitive_attribute(X_scaled, y, sensitive_attribute_index)
    subsets_standard_scaled = split_data_by_sensitive_attribute(X_standard_scaled, y, sensitive_attribute_index)

    # subsets = {0: (X, y)}
    # subsets_scaled = {0: (X_scaled, y)}
    # subsets_standard_scaled = {0: (X_standard_scaled, y)}

    subsets_scaled_matrix = {}
    # Distance matrix for N1 y T1 with scaled data
    for key, (X_subset, y_subset) in subsets_scaled.items():
        distance_matrix = distance.cdist(X_subset, X_subset, 'euclidean')
        subsets_scaled_matrix[key] = (X_subset, y_subset, distance_matrix)

    # print('distance matrix done')

    measures = {}
    try:

        F1v = ft_F1v(subsets)
        for key in F1v.keys():
            measures[key].append(F1v[key])
        print('F1v done')

        '''
        F1 = ft_F1(subsets)
        for key in F1.keys():
            measures[key] = [F1[key]]
        print('F1 done')
    
        F2 = ft_F2(subsets)
        for key in F2.keys():
            measures[key].append(F2[key])
        print('F2 done')
    
        F3 = ft_F3(subsets)
        for key in F3.keys():
            measures[key].append(F3[key])
        print('F3 done')
    
        F4 = ft_F4(subsets)
        for key in F4.keys():
            measures[key].append(F4[key])
        print('F4 done')
        '''
        #print("F1v", ft_F1v(X, y, sensitive_attribute_index))
        if file_name not in ['S1C', 'S2C', 'S3C', 'S4C', 'S5C']:
            F1 = ft_F1v(subsets)
            for key in F1.keys():
                measures[key] = [F1[key]]
        else:
            for key in subsets.keys():
                measures[key] = [1]
        # print("N1", ft_N1(X, y, sensitive_attribute_index))
        print('F1v done')
        N1 = ft_N1(subsets_scaled_matrix)
        for key in N1.keys():
            measures[key].append(N1[key])
        # print("N2", ft_N2(X, y, sensitive_attribute_index))
        print('N1 done')
        N2 = ft_N2(subsets_scaled)
        for key in N2.keys():
            measures[key].append(N2[key])
        # print("N3", ft_N3(X, y, sensitive_attribute_index))
        print('N2 done')
        N3 = ft_N3(subsets_scaled)
        for key in N3.keys():
            measures[key].append(N3[key])
        #print("LSC", LSC(X, y, sensitive_attribute_index))
        print('N3 done')
        LSC_ = LSC(subsets_scaled)
        for key in LSC_.keys():
            measures[key].append(LSC_[key])
        #print("T1", T1(X, y, sensitive_attribute_index))
        print('LSC done')
        T1_ = T1(subsets_scaled_matrix)
        for key in T1_.keys():
            measures[key].append(T1_[key])
        del subsets_scaled_matrix
        gc.collect()
        print('T1 done')
        #print("T2, T3, T4", ft_T2T3T4(X, y, sensitive_attribute_index))
        T2T3T4 = ft_T2T3T4(subsets_standard_scaled)
        for m in ["T2", "T3", "T4"]:
            for key in T2T3T4[m].keys():
                measures[key].append(T2T3T4[m][key])
        print('T2T3T4 done')
        #print("C1, C2", C12(X, y, sensitive_attribute_index))
        C12_ = C12(subsets)
        for m in ["C1", "C2"]:
            for key in C12_[m].keys():
                measures[key].append(C12_[m][key])
        #print("L123", L123(X, y, sensitive_attribute_index))
        print('C12 done')
        L123_ = L123(subsets_scaled)
        for m in ["L1", "L2", "L3", "N4"]:
            for key in L123_[m].keys():
                measures[key].append(L123_[m][key])
        #print("density, CC, hubs", network_measures(X, y, sensitive_attribute_index))
        print('L123 done')
        NM = network_measures(subsets)
        for m in ['density', 'clustering_coefficient', 'hubs']:
            #print(m)
            for key in NM[m].keys():
                #print(key)
                measures[key].append(NM[m][key])
                
        

        print('{} Done'.format(file))


    except Exception as e:
        print('Error computing complexity measures for {}'.format(file_name))
        print(e)
        measures = {}

    print('{} Done'.format(file))
    list_measures = []
    if len(measures.keys()) > 0:
        for key in measures.keys():
            #print(measures[key])
            list_att = measures[key]
            list_att.append(key)
            list_att.append('A')
            list_att.append(file_name)
            list_measures.append(list_att)
            measures_df = pd.DataFrame(list_measures, columns=measures_names)
            measures_df.to_csv(

                "results_measures_syn/results_{}.csv".format(file_name), index=False) #results_measures_syn_combination













