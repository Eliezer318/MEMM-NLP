from scipy.sparse import csr_matrix
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from data import FeatureClass

losses = []


def train(dataset='train1', max_train=1_000, threshold=2, reg=0.5):
    path = f'/datashare/hw1/{dataset}.wtag'
    features = FeatureClass(file_path=path, threshold=threshold)
    return calc_till_convergence(features, max_train, dataset, reg)


def calc_objective(v: np.ndarray, tot_matrix: csr_matrix, true_indices, empirical_counts, num_of_labels, reg=0.5):
    # reg = (0.5 * np.random.randn()) ** 2
    # count log-linear objective function
    dot = tot_matrix @ v
    linear_term = dot[true_indices].sum()  # [N, D] x [D, 1] = [N, 1] them sum all to scalar
    tmp = np.exp(dot)  # [|S|*|N|, D] x [D, 1] = [|S|*N, 1] => [N, |S|]
    tmp_summed = tmp.reshape(-1, num_of_labels).sum(axis=1)
    norm_term = np.log(tmp_summed).sum()  # scalar
    reg_term = 0.5 * reg * (v ** 2).sum()  # scalar
    log_linear = linear_term - norm_term - reg_term

    # count gradient of log-linear objective function

    # divider = np.repeat(tmp.sum(axis=1), num_of_labels)  # Somehow line 52 create whats needed faster.
    divider = np.outer(tmp_summed, np.ones(num_of_labels)).reshape(-1)
    # pointwise multiplication, probability with the corresponding vector.
    expected_counts = tot_matrix.T.multiply((tmp / divider)).sum(axis=1).reshape(-1)
    reg_grad = reg * v
    gradient = empirical_counts - expected_counts - reg_grad
    losses.append(-log_linear)
    return (-1) * log_linear, (-1) * gradient.reshape(-1)


def calc_till_convergence(features: FeatureClass, max_iter=1000, dataset='train1', reg=0.5):
    """
    :param features: include features, train matrix and label set
    :param max_iter: how many iteration will be in worst case
    :param dataset: which dataset to train on
    :param reg: regularization term
    :return: optimal vector
    """
    emprical_count = (features.tot_matrix[features.true_indices]).sum(axis=0)
    tot_matrix = features.tot_matrix
    true_indices = features.true_indices
    labels_set = features.labels
    args = [tot_matrix, true_indices, emprical_count, len(labels_set), reg]
    w_0 = np.random.randn(tot_matrix.shape[1])
    optimal_params = fmin_l_bfgs_b(func=calc_objective, x0=w_0, args=args, maxiter=max_iter, iprint=50)
    weights = optimal_params[0]
    # pickle.dump((weights, optimal_params, losses), open(f'cache/weights_{dataset}.pkl', 'wb'))
    return weights, features, losses
