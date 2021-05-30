import numpy as np
import re
from scipy.sparse import csr_matrix
import pandas as pd
from collections import defaultdict

from data import FeatureClass


def print_confusion_matrix_top_10(features: FeatureClass, confusion_matrix: np.ndarray):
    n = len(features.labels)
    confusion_matrix_cpy = confusion_matrix.copy()
    confusion_matrix_cpy[np.arange(n), np.arange(n)] = 0
    indices_wrong = confusion_matrix_cpy.sum(axis=1).argsort()[-10:]
    wrong_columns = [features.labels[idx_wrong] for idx_wrong in indices_wrong]
    print(pd.DataFrame(confusion_matrix[indices_wrong, :][:, indices_wrong].astype(int), wrong_columns, wrong_columns))


def find_acc(test_path: str, weight_v: np.ndarray, features: FeatureClass) -> float:
    n = len(features.labels)
    label2id = {features.labels[i]: i for i in range(n)}
    confusion_matrix = np.zeros((n, n))
    dic_counts_errors = defaultdict(int)
    with open(test_path) as f:
        for line in f:
            splited_words = re.split(' |\n', line)[:-1]
            words_1_n, tags_gt = [], []
            for word_idx in range(len(splited_words)):
                cur_word, cur_tag = splited_words[word_idx].split('_')
                tags_gt.append(cur_tag)
                words_1_n.append(cur_word)
            tags_inferd = memm_viterbi(weight_v, features, words_1_n)  # TODO insert true parameters
            for idx, (tag_gt, tag_inferd, word) in enumerate(zip(tags_gt, tags_inferd, words_1_n)):
                if tag_gt != tag_inferd:
                    dic_counts_errors[word] += 1
                confusion_matrix[label2id[tag_gt], label2id[tag_inferd]] += 1

    print_confusion_matrix_top_10(features, confusion_matrix)
    return confusion_matrix[np.arange(n), np.arange(n)].sum()/confusion_matrix.sum()


def q_func(weight_vector: np.ndarray, features: FeatureClass, w_1_n, idx, p_tag_list, pp_tag_list) -> np.ndarray:
    # create matrix of combinations of pp_tag list, p_tag list and current tag list and return probabilities matrix
    rows, cols = [], []
    counter = 0
    ppword = '*' if idx - 2 < 0 else w_1_n[idx - 2]
    pword = '*' if idx - 1 < 0 else w_1_n[idx - 1]
    nword = '*' if idx + 1 >= len(w_1_n) else w_1_n[idx + 1]
    nnword = '*' if idx + 2 >= len(w_1_n) else w_1_n[idx + 2]
    word = w_1_n[idx]
    for pp in pp_tag_list:
        for p in p_tag_list:
            for ctag in features.labels:
                feat = features.represent_input_with_features((ppword, pword, word, nword, nnword, pp, p, ctag))
                cols.extend(feat)
                rows.extend((counter + np.zeros_like(feat)).tolist())
                counter += 1

    mat = csr_matrix((np.ones_like(cols), (rows, cols)), shape=(counter, features.n_total_features), dtype=bool)
    scores = np.exp(mat @ weight_vector).reshape(len(pp_tag_list), len(p_tag_list), len(features.labels))
    return scores/(scores.sum(axis=2).reshape(len(pp_tag_list), len(p_tag_list), -1))


def memm_viterbi(weight_v: np.ndarray, features: FeatureClass, w_1_n, B=2) -> np.ndarray:
    # B - Beam Search parameter
    labels = list(features.labels)
    n = len(w_1_n)
    pi_matrix = {(0, '*', '*'): 1}
    bp_matrix = {}
    S = {}
    S[-1] = ['*']
    S[0] = ['*']
    for k in range(1, n + 1):
        probs = q_func(weight_v, features, w_1_n, k - 1, S[k - 1], S[k - 2])
        dic_pbs = defaultdict(int)
        for iu, u in enumerate(S[k - 1]):
            for iv, v in enumerate(labels):
                temp_vec = np.array([pi_matrix[(k - 1, t, u)] * probs[it, iu, iv] for it, t in enumerate(S[k - 2])])
                pi_matrix[(k, u, v)] = temp_vec.max()
                bp_matrix[(k, u, v)] = S[k - 2][temp_vec.argmax()]

        for p, v in sorted([(pi_matrix[(k, u, v)], v) for v in labels for u in S[k - 1]], reverse=True)[:B * 2]:
            dic_pbs[v] += p
        S[k] = sorted(dic_pbs, key=lambda x: dic_pbs[x], reverse=True)[:B]

    prediction = np.empty(n, dtype=np.chararray)
    idx, u, v = max({key: pi_matrix[key] for key in pi_matrix if key[0] == n}, key=lambda key: pi_matrix[key])
    prediction[n - 1] = v
    if (n - 2) >= 0:
        prediction[n - 2] = u
    for k in range(n - 3, -1, -1):
        prediction[k] = bp_matrix[(k + 3, prediction[k + 1], prediction[k + 2])]
    return prediction


def create_comp_file(comp_path: str, comp_words: str, weight_v: np.ndarray, features: FeatureClass):
    with open(comp_path, 'w') as new_f:
        with open(comp_words) as f:
            for line in f:
                splited_words = re.split(' |\n', line)[:-1]
                words_1_n, tags_gt = [], []
                for word_idx in range(len(splited_words)):
                    cur_word = splited_words[word_idx]
                    words_1_n.append(cur_word)
                tags_inferd = memm_viterbi(weight_v, features, words_1_n)
                new_f.write(f"{' '.join([f'{word}_{tag}' for tag, word in zip(tags_inferd, words_1_n)])}\n")
