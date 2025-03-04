#!/usr/bin/env python
import typing as t
import numpy as np
import itertools

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import accuracy_score
import collections
import math
from sklearn.svm import SVC
from sklearn.utils import resample

from scipy.spatial import distance
import numpy as np
import networkx as nx
import gower
# from scipy.spatial import distance_matrix



def precompute_fx(X: np.ndarray, y: np.ndarray) -> t.Dict[str, t.Any]:
    """Precompute some useful things to support complexity measures.
    Parameters
    ----------
    X : :obj:`np.ndarray`, optional
            Attributes from fitted data.
    y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.
    Returns
    -------
    :obj:`dict`
        With following precomputed items:
        - ``ovo_comb`` (:obj:`list`): List of all class OVO combination, 
            i.e., [(0,1), (0,2) ...].
        - ``cls_index`` (:obj:`list`):  The list of boolean vectors
            indicating the example of each class. 
        - ``cls_n_ex`` (:obj:`np.ndarray`): The number of examples in
            each class. The array indexes represent the classes.
    """

    prepcomp_vals = {}
    
    classes, class_freqs = np.unique(y, return_counts=True)
    cls_index = [np.equal(y, i) for i in range(classes.shape[0])]

    #cls_n_ex = np.array([np.sum(aux) for aux in cls_index])
    cls_n_ex = list(class_freqs)
    ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))
    prepcomp_vals["ovo_comb"] = ovo_comb
    prepcomp_vals["cls_index"] = cls_index
    prepcomp_vals["cls_n_ex"] = cls_n_ex
    return prepcomp_vals



def numerator(X: np.ndarray, cls_index, cls_n_ex, i) -> float:
    return np.sum([cls_n_ex[j]*np.power((np.mean(X[cls_index[j], i])-
                                         np.mean(X[:, i], axis=0)),2) for j in range (len(cls_index))])


def denominator(X: np.ndarray, cls_index, cls_n_ex, i) -> float:
    return np.sum([np.sum(np.power(X[cls_index[j], i]-np.mean(X[cls_index[j], i], axis=0), 2)) 
                   for j in range(0, len(cls_n_ex))]) 


def compute_rfi (X: np.ndarray, cls_index, cls_n_ex) -> float:
    return [numerator(X, cls_index, cls_n_ex, i)/denominator(X, cls_index, cls_n_ex, i)
            for i in range(np.shape(X)[1])]


def ft_F1_old(X: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> float:
    return 1/(1 + np.max(compute_rfi (X, cls_index, cls_n_ex)))


def ft_F1(subsets):
    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        # logger = getLogger()
        maxr = -math.inf
        d_ratios = {}
        d_values = {}
        X_classes = {}
        averages_classes = {}
        averages = np.average(X_subset, axis=0)
        counter = collections.Counter(y_subset)
        for c in counter.keys():
            X_classes[c] = X_subset[y_subset == c]
            averages_classes[c] = np.average(X_classes[c], axis=0)
        for i in range(X_subset.shape[-1]):
            numerator = 0.0
            denominator = 0.0
            for c in counter.keys():
                X_c = X_classes[c]
                averages_c = averages_classes[c]
                m = (averages_c[i] - averages[i]) ** 2
                numerator += counter[c] * m
                x = X_c[:, i]
                x = x - averages_c[i]
                x = x ** 2
                denominator += np.sum(x)
            r = (numerator / denominator) if denominator > 0 else 0
            d_ratios[i] = r
            d_values[i] = 1 / (1 + r)
            maxr = r if r > maxr else maxr
            # logger.debug("{} => numerator = {}, denominator = {}, r = {}".format(i, numerator, denominator, r))
            # np.mean(d_ratios, 1 / (1 + maxr))
        # results[key] = 1 / (1 + maxr) # original
        results[key] = d_ratios, d_values, 1 / (1 + maxr)
        # TODO this says nothing about overlap
        # print(d_ratios)

    return results


def ft_F1v(subsets) -> float:

    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_fx = precompute_fx(X_subset, y_subset)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']

        df_list = []

        for idx1, idx2 in ovo_comb:
            y_class1 = cls_index[idx1]
            y_class2 = cls_index[idx2]
            print(X_subset.shape)
            dF = dVector(X_subset, y_class1, y_class2)
            df_list.append(1/(1+dF))

        results[key] = np.mean(df_list)
        
    return results


def dVector(X: np.ndarray, y_class1: np.ndarray, y_class2: np.ndarray) -> float:
    X_class1 = X[y_class1]; u_class1 = np.mean(X_class1, axis= 0)
    X_class2 = X[y_class2]; u_class2 = np.mean(X_class2, axis= 0)


    W = ((np.shape(X_class1)[0]/ (np.shape(X_class1)[0] + np.shape(X_class2)[0]))* np.cov(X_class1.T)) \
     + (np.shape(X_class2)[0]/(np.shape(X_class1)[0] + (np.shape(X_class2)[0])) * np.cov(X_class2.T))
    
    d = np.dot(np.linalg.pinv(W), (u_class1 - u_class2)) #inv changed by pinv because of singular matrices
    
    B = np.dot((u_class1 - u_class2),((u_class1 - u_class2).T))
    
    return np.dot(np.dot(d.T, B), d)/ np.dot(np.dot(d.T, W), d)


# #### Volume of Overlapping Region (F2)

def _minmax(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the minimum of the maximum values per class
    for all features.
    """
    max_cls = np.zeros((2, X.shape[1]))
    max_cls[0, :] = np.max(X[class1], axis=0)
    max_cls[1, :] = np.max(X[class2], axis=0)
    aux = np.min(max_cls, axis=0)
    return aux


# In[15]:


def _minmin(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the minimum of the minimum values per class
    for all features.
    """
    min_cls = np.zeros((2, X.shape[1]))
    min_cls[0, :] = np.min(X[class1], axis=0)
    min_cls[1, :] = np.min(X[class2], axis=0)
    aux = np.min(min_cls, axis=0)
    return aux


# In[16]:
def _maxmin(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the maximum of the minimum values per class
    for all features.
    """
    min_cls = np.zeros((2, X.shape[1]))
    min_cls[0, :] = np.min(X[class1], axis=0)
    min_cls[1, :] = np.min(X[class2], axis=0)
    aux = np.max(min_cls, axis=0)
    return aux


# In[17]:


def _maxmax(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the maximum of the maximum values per class
    for all features. 
    """
    max_cls = np.zeros((2, X.shape[1]))
    max_cls[0, :] = np.max(X[class1], axis=0)
    max_cls[1, :] = np.max(X[class2], axis=0)
    aux = np.max(max_cls, axis=0)
    return aux


# In[18]:


def ft_F2(subsets) -> float:

    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_fx = precompute_fx(X_subset, y_subset)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']

        f2_list = []
        for idx1, idx2 in ovo_comb:
            y_class1 = cls_index[idx1]
            y_class2 = cls_index[idx2]
            zero_ = np.zeros(np.shape(X_subset)[1])
            overlap_ = np.maximum(zero_, _minmax(X_subset, y_class1, y_class2)-_maxmin(X_subset, y_class1, y_class2))
            range_ = _maxmax(X_subset, y_class1, y_class2)-_minmin(X_subset, y_class1, y_class2)
            ratio = overlap_/range_
            f2_list.append(np.prod(ratio))

        results[key] = np.mean(f2_list)
        # np.mean(f2_list)
        
    return results



def _compute_f3(X_: np.ndarray, minmax_: np.ndarray, maxmin_: np.ndarray) -> np.ndarray:
    """ This function computes the F3 complexity measure given minmax and maxmin."""

    overlapped_region_by_feature = np.logical_and(X_ >= maxmin_, X_ <= minmax_)

    n_fi = np.sum(overlapped_region_by_feature, axis=0)
    idx_min = np.argmin(n_fi)

    return idx_min, n_fi, overlapped_region_by_feature


def ft_F3(subsets) -> np.ndarray:
    """TODO
    """
    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_fx = precompute_fx(X_subset, y_subset)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']

        f3 = []
        for idx1, idx2 in ovo_comb:
            idx_min, n_fi, _ = _compute_f3(X_subset, _minmax(X_subset, cls_index[idx1], cls_index[idx2]),
            _maxmin(X_subset, cls_index[idx1], cls_index[idx2]))
            # print(idx_min, n_fi, cls_n_ex[idx1] + cls_n_ex[idx2])

            f3.append(n_fi[idx_min] / (cls_n_ex[idx1] + cls_n_ex[idx2]))
            # f3.append(np.mean(n_fi) / (cls_n_ex[idx1] + cls_n_ex[idx2]))

        results[key] = np.min(f3)

    return results



# #### Collective Feature Efficiency (F4)


def ft_F4(subsets) -> np.ndarray:
    """TODO - not should be taken into account, it works by eliminating features according to its relevance
    """
    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_fx = precompute_fx(X_subset, y_subset)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']

        f4 = []
        for idx1, idx2 in ovo_comb:
            aux = 0

            y_class1 = cls_index[idx1]
            y_class2 = cls_index[idx2]
            sub_set = np.logical_or(y_class1, y_class2)
            y_class1 = y_class1[sub_set]
            y_class2 = y_class2[sub_set]
            X_ = X_subset[sub_set, :]
            # X_ = X[np.logical_or(y_class1, y_class2),:]

            while X_.shape[1] > 0 and X_.shape[0] > 0:
                # True if the example is in the overlapping region
                idx_min, _, overlapped_region_by_feature = _compute_f3(X_,_minmax(X_, y_class1, y_class2),_maxmin(X_, y_class1, y_class2))

                # boolean that if True, this example is in the overlapping region
                overlapped_region = overlapped_region_by_feature[:, idx_min]

                # removing the non overlapped features
                X_ = X_[overlapped_region, :]
                y_class1 = y_class1[overlapped_region]
                y_class2 = y_class2[overlapped_region]

                if X_.shape[0] > 0:
                    aux = X_.shape[0]
                else:
                    aux = 0

                # removing the most efficient feature
                X_ = np.delete(X_, idx_min, axis=1)

            f4.append(aux/(cls_n_ex[idx1] + cls_n_ex[idx2]))

        results[key] = np.mean(f4)# f4
        #np.mean(f4)

    return results




# ### Neighborhood Measures

# Fraction of Borderline Points (N1)
def ft_N1(subsets) -> np.ndarray:

    results = {}

    for key, (X_subset, y_subset, distance_matrix) in subsets.items():
        # compute the distance matrix and the minimum spanning tree.
        dist_m = np.triu(distance_matrix, k=1) #distance.cdist(X_subset, X_subset, metric)
        mst = minimum_spanning_tree(dist_m)
        node_i, node_j = np.where(mst.toarray() > 0)

        # which edges have nodes with different class
        which_have_diff_cls = y_subset[node_i] != y_subset[node_j]

        # number of different vertices connected
        aux = np.unique(np.concatenate([node_i[which_have_diff_cls],node_j[which_have_diff_cls]])).shape[0]
        results[key] = aux/X_subset.shape[0]

    return results

# #### Ratio of Intra/Extra Class Nearest Neighbor Distance (N2)


def nearest_enemy(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray,
                   i: int, metric: str = "euclidean", n_neighbors=1) :
    "This function computes the distance from a point x_i to their nearest enemy"
    
    X_ = X[np.logical_not(cls_index[y[i]])]
    y_ = y[np.logical_not(cls_index[y[i]])]
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    neigh.fit(X_, y_) 
    dist_enemy, pos_enemy = neigh.kneighbors([X[i, :]])
    dist_enemy = np.reshape(dist_enemy, (n_neighbors,))
    pos_enemy_ = np.reshape(pos_enemy, (n_neighbors,))
    query = X_[pos_enemy_, :]
    pos_enemy = np.where(np.all(X == query, axis=1))
    pos_enemy = np.reshape(pos_enemy[0][0], (n_neighbors,)) # before pos_enemy

    return dist_enemy, pos_enemy


def nearest_neighboor_same_class (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray,
                                  i: int, metric: str = "euclidean", n_neighbors=1) :
    " This function computes the distance from a point x_i to their nearest neighboor from its own class"
    query = X[i, :]
    label_query = y[i]
    X_ = X[cls_index[label_query]]
    y_ = y[cls_index[label_query]]
    
    pos_query = np.where(np.all(X_==query,axis=1))
    X_ = np.delete(X_, pos_query, axis = 0)
    y_ = np.delete(y_, pos_query, axis = 0) 
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    neigh.fit(X_, y_)
    dist_neigh, pos_neigh = neigh.kneighbors([X[i, :]])
    dist_neigh = np.reshape(dist_neigh, (n_neighbors,))
    pos_neigh = np.reshape(pos_neigh, (n_neighbors,))
    return dist_neigh, pos_neigh


def intra_extra(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray):
    intra = np.sum([nearest_neighboor_same_class(X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    extra = np.sum([nearest_enemy(X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    return intra/extra


def ft_N2 (subsets):

    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_fx = precompute_fx(X_subset, y_subset)
        cls_index = precomp_fx['cls_index']
        intra_extra_ = intra_extra(X_subset, y_subset, cls_index)
        results[key] = intra_extra_/(1+intra_extra_)

    return results



# #### Error Rate of the Nearest Neighbor Classifier (N3)

def ft_N3 (subsets, metric: str = "euclidean") -> float:
    results = {}
    for key, (X_subset, y_subset) in subsets.items():
    
        loo = LeaveOneOut()
        loo.get_n_splits(X_subset, y_subset)

        y_test_ = []
        pred_y_ = []
        for train_index, test_index in loo.split(X_subset):
            X_train, X_test = X_subset[train_index], X_subset[test_index]
            y_train, y_test = y_subset[train_index], y_subset[test_index]
            model = KNeighborsClassifier(n_neighbors=1, metric=metric)
            model.fit(X_train, y_train)
            pred_y = model.predict(X_test)
            y_test_.append(y_test)
            pred_y_.append(pred_y)

        results[key] = 1 - accuracy_score(y_test_, pred_y_)
    return results


# #### Fraction of Hyperspheres Covering Data (T1)



def radios (D: np.ndarray, y: np.ndarray, X: np.ndarray, 
                        cls_index:np.ndarray, i: int) -> float:
    d_i, x_j = nearest_enemy(X, y, cls_index, i)
    d_j, x_k = nearest_enemy(X, y, cls_index, x_j[0])
    if (i == x_k[0]):
        return d_i/2
    else:
        d_t = radios (D, y, X, cls_index, x_j[0]) 
        var = d_i - d_t
        return d_i - d_t


def hyperspher (D: np.ndarray, y: np.ndarray, X: np.ndarray, cls_index:np.ndarray) :

  aux = [radios(D, y, X, cls_index, i) for i in range(X.shape[0])]
  return aux


# #### Local Set Average Cardinality (LSC)
def LS_i (X: np.ndarray, y: np.ndarray, i: int, cls_index, metric: str = "euclidean"):
    dist_enemy, pos_enemy = nearest_enemy(X, y, cls_index, i)
    dist_ = distance.cdist(X, [X[i, :]], metric=metric)
    X_j = dist_[np.logical_and(dist_ < dist_enemy, dist_ != 0)]
    return X_j


def LSC (subsets):
    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_fx = precompute_fx(X_subset, y_subset)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']

        n = np.shape(X_subset)[0]
        x = [np.shape(LS_i(X_subset, y_subset, i, cls_index)) for i in range(n)]
        results[key] = 1 - np.sum(x)/n**2

    return results





# ### Dimensionality Measures
# - Average number of features per dimension (T2)
# - Average number of PCA dimensions per points (T3)
# - Ratio of the PCA Dimension to the Original Dimension (T4)


def precompute_pca_tx(X: np.ndarray) -> t.Dict[str, t.Any]:
    """Precompute PCA to Tx complexity measures.
    Parameters
    ----------
    X : :obj:`np.ndarray`, optional
            Attributes from fitted data.
    Returns
    -------
    :obj:`dict`
        With following precomputed items:
        - ``m`` (:obj:`int`): Number of features.
        - ``m_`` (:obj:`int`):  Number of features after PCA with 0.95.
        - ``n`` (:obj:`int`): Number of examples.
    """
    prepcomp_vals = {}
    pca = PCA(n_components=0.95)
    pca.fit(X)

    m_ = pca.explained_variance_ratio_.shape[0]
    m = X.shape[1]
    n = X.shape[0]

    prepcomp_vals["m_"] = m_
    prepcomp_vals["m"] = m
    prepcomp_vals["n"] = n

    return prepcomp_vals


# #### Average number of features per dimension (T2)


def ft_T2T3T4(subsets):

    results = {'T2': {}, 'T3': {}, 'T4': {}}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_pca = precompute_pca_tx(X_subset)
        m = precomp_pca['m']
        n = precomp_pca['n']
        m_ = precomp_pca['m_']
        results['T2'][key] = m / n
        results['T3'][key] = m_ / n
        results['T4'][key] = m_ / m

    return results


def C12(subsets):
    """
    Calculate Entropy of Class Proportions (C1) and Imbalance Ratio (C2)
      - X: ndarray features
      - y: ndarray target
      - sensitive_attribute_index: index of the sensitive attribute in X
    """
    results = {'C1': {}, 'C2': {}}

    for key, (X_subset, y_subset) in subsets.items():
        n = len(y_subset)
        counter = collections.Counter(y_subset)
        entroy = 0.0
        tmp_sum = 0.0
        for c in counter.keys():
            pc = counter[c] / n
            entroy += pc * math.log(pc)
            tmp_sum += counter[c] / (n - counter[c])
        C1 = 1 - (-entroy / math.log(len(counter)))
        IR = ((len(counter) - 1) / len(counter)) * tmp_sum
        C2 = 1 - (1 / IR)
        results['C1'][key] = C1
        results['C2'][key] = C2
    return results


def interpolate_data(X, y, num_samples):
    X_interpolated = []
    y_interpolated = []

    for _ in range(num_samples):
        # Seleccionar dos ejemplos al azar de la misma clase
        class_label = np.random.choice(np.unique(y))
        X_class = X[y == class_label]
        if len(X_class) < 2:
            continue

        x1, x2 = resample(X_class, n_samples=2, replace=False)
        # Interpolar con coeficientes aleatorios
        alpha = np.random.rand()
        x_new = alpha * x1 + (1 - alpha) * x2
        X_interpolated.append(x_new)
        y_interpolated.append(class_label)

    return np.array(X_interpolated), np.array(y_interpolated)


def L123(subsets, metric: str = "euclidean", p=2, n_neighbors=1):
    results = {'L1': {}, 'L2': {}, 'L3': {}, 'N4': {}}

    for key, (X_subset, y_subset) in subsets.items():

        # Ajustar un modelo SVM lineal
        svm = SVC(kernel='linear', C=1)
        svm.fit(X_subset, y_subset)

        # Predecir y calcular errores
        predictions = svm.predict(X_subset)
        errors = predictions != y_subset

        # Calcular los márgenes (distancias al hiperplano)
        distances = svm.decision_function(X_subset)
        epsilon = 1 - y_subset * distances  # εi

        # Medida L1
        sum_error_dist = np.sum(epsilon[errors]) / len(X_subset)
        L1 = 1 / (1 + sum_error_dist)


        # Medida L2
        L2 = np.mean(errors)

        # Medida L3
        num_samples = len(X_subset)
        X_interpolated, y_interpolated = interpolate_data(X_subset, y_subset, num_samples)
        interpolated_predictions = svm.predict(X_interpolated)
        L3 = accuracy_score(y_interpolated, interpolated_predictions)

        # Medida N4
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric=metric).fit(X_subset, y_subset)
        y_pred = knn.predict(X_interpolated)
        N4 = accuracy_score(y_interpolated, y_pred)

        results['L1'][key] = L1
        results['L2'][key] = L2
        results['L3'][key] = L3
        results['N4'][key] = N4

    return results


# #### Non-Linearity of the Nearest Neighbor Classifier (N4)
def ft_N4(subsets,
          metric: str = "euclidean", p=2, n_neighbors=1) -> np.ndarray:

    results = {}

    for key, (X_subset, y_subset) in subsets.items():
        precomp_fx = precompute_fx(X_subset, y_subset)
        cls_index = precomp_fx['cls_index']

        interp_X = []
        interp_y = []


        for idx in cls_index:
            #creates a new dataset by interpolating pairs of training examples of the same class.
            X_ = X_subset[idx]

            #two examples from the same class are chosen randomly and
            #they are linearly interpolated (with random coefficients), producing a new example.
            A = np.random.choice(X_.shape[0], X_.shape[0])
            A = X_[A]
            B = np.random.choice(X_.shape[0], X_.shape[0])
            B = X_[B]
            delta = np.random.ranf(X_.shape)

            interp_X_ = A + ((B - A) * delta)
            interp_y_ = y_subset[idx]

            interp_X.append(interp_X_)
            interp_y.append(interp_y_)

        # join the datasets
        X_test = np.concatenate(interp_X)
        y_test = np.concatenate(interp_y)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric=metric).fit(X_subset, y_subset)
        y_pred = knn.predict(X_test)
        results[key] = 1 - accuracy_score(y_test, y_pred)

    return results




def compute_distance_matrix(X):
    dist = distance.cdist(X, X, 'euclidean')
    return dist



def nearest_enemy_(dist_matrix, y, i):
    same_class = np.where(y == y[i])[0]
    other_class = np.where(y != y[i])[0]

    # Distancias a puntos de la clase opuesta
    dists_to_other_class = dist_matrix[i, other_class]

    # Índice del enemigo más cercano
    nearest_enemy_index = other_class[np.argmin(dists_to_other_class)]
    return nearest_enemy_index, np.min(dists_to_other_class)


def radius_(dist_matrix, y, i, computed_radii):
    if i in computed_radii:
        return computed_radii[i]

    j, di = nearest_enemy_(dist_matrix, y, i)
    k, _ = nearest_enemy_(dist_matrix, y, j)

    if i == k:
        computed_radii[i] = di / 2
    else:
        dt = radius_(dist_matrix, y, j, computed_radii)
        computed_radii[i] = di - dt

    return computed_radii[i]


def compute_hyperspheres(X, y, distance_matrix):
    computed_radii = {}

    for i in range(len(X)):
        radius_(distance_matrix, y, i, computed_radii)

    hyperspheres = [(i, computed_radii[i]) for i in range(len(X))]

    # Eliminar hiperesferas contenidas en otras más grandes
    hyperspheres = sorted(hyperspheres, key=lambda x: x[1], reverse=True)
    final_hyperspheres = []

    for i, (index, radius) in enumerate(hyperspheres):
        contained = False
        for j in range(i):
            if distance_matrix[index, hyperspheres[j][0]] + radius <= hyperspheres[j][1]:
                contained = True
                break
        if not contained:
            final_hyperspheres.append((index, radius))

    return final_hyperspheres


def T1(subsets):
    results = {}

    for key, (X_subset, y_subset, distance_matrix) in subsets.items():

        hyperspheres = compute_hyperspheres(X_subset, y_subset, distance_matrix)
        results[key] = len(hyperspheres) / len(X_subset)

    return results


def gower_distance(X):
    """Calcula la distancia de Gower para un conjunto de datos"""
    num_rows, num_cols = X.shape
    ranges = np.ptp(X, axis=0)
    ranges[ranges == 0] = 1
    norm_X = X / ranges
    dist_matrix = compute_distance_matrix(norm_X, norm_X)
    return dist_matrix


def build_graph(X, y, epsilon=0.15):
    X = X.astype(float)
    dist_matrix = gower.gower_matrix(X) #gower_distance(X)
    G = nx.Graph()

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if dist_matrix[i, j] < epsilon:
                G.add_edge(i, j, weight=dist_matrix[i, j])

    # Podar el grafo desconectando ejemplos de diferentes clases
    for edge in list(G.edges()):
        if y[edge[0]] != y[edge[1]]:
            G.remove_edge(edge[0], edge[1])

    return G


def compute_density(G):
    n = len(G.nodes())
    if n == 0:
        print("Graph with zero nodes")
    m = len(G.edges())
    density = 2 * m / (n * (n - 1)) if n > 0 else -1
    return 1 - density


def compute_clustering_coefficient(G):
    n = len(G.nodes())
    clustering_coeffs = nx.clustering(G)
    avg_clustering_coeff = sum(clustering_coeffs.values()) / n if n > 0 else -1
    return 1 - avg_clustering_coeff


def compute_hub_score(G):
    n = len(G.nodes())
    hubs, _ = nx.hits(G)
    avg_hub_score = sum(hubs.values()) / n if n > 0 else -1
    return 1 - avg_hub_score


def network_measures(subsets, epsilon=0.15):
    results = {'density': {}, 'clustering_coefficient': {}, 'hubs': {}}

    for key, (X_subset, y_subset) in subsets.items():

        G = build_graph(X_subset, y_subset, epsilon)
        density = compute_density(G)
        clustering_coefficient = compute_clustering_coefficient(G)
        hub_score = compute_hub_score(G)

        results['density'][key] = density
        results['clustering_coefficient'][key] = clustering_coefficient
        results['hubs'][key] = hub_score
        print(hub_score)

    return results
