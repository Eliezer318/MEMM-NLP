from collections import defaultdict
import re
from scipy.sparse import csr_matrix
import numpy as np


def create_mandatory_features(pptag, ptag, cword, ctag, dic_features):
    # feature words with capital letter
    if bool(re.search(r'[A-Z]', cword)):
        dic_features['feature_words_with_capital_letters'][(cword, ctag)] += 1

    # feature words with digits playing
    if bool(re.search(r'[0-9]', cword)):
        dic_features['feature_words_with_numbers'][(cword, ctag)] += 1
        dic_features['digit_tag_exist'][ctag] += 1

    # feature 100
    dic_features['feature_100'][(cword, ctag)] += 1

    # feature 101 + 102
    for i in range(1, min(5, len(cword) + 1)):
        dic_features['feature_101'][(cword[-i:], ctag)] += 1  # suffix
        dic_features['feature_102'][(cword[:i], ctag)] += 1  # prefix

    # feature 103
    dic_features['feature_103'][(pptag, ptag, ctag)] += 1

    # feature 104
    dic_features['feature_104'][(ptag, ctag)] += 1

    # feature 105
    dic_features['feature_105'][ctag] += 1


def features_lower_cases(cword, ctag, pword, nword, dic_features):
    cword = cword.lower()
    pword = pword.lower()
    nword = nword.lower()
    # feature 100
    dic_features['feature_100_lower'][(cword, ctag)] += 1

    # feature 101 + 102
    for i in range(1, min(5, len(cword) + 1)):
        dic_features['feature_101_lower'][(cword[-i:], ctag)] += 1  # suffix
        dic_features['feature_101_lower_complete'][(cword[:-i], ctag)] += 1  # suffix
        dic_features['feature_102_lower'][(cword[:i], ctag)] += 1  # prefix
        dic_features['feature_102_lower_complete'][(cword[i:], ctag)] += 1  # prefix

    # feature 106
    dic_features['feature_106'][(pword, ctag)] += 1

    # feature 107
    dic_features['feature_107'][(nword, ctag)] += 1


def change_word(word: str):
    def trans_letter(c: str):
        if c.isupper():
            return 'X'
        if c.islower():
            return 'x'
        if c.isdigit():
            return 'd'
        return c
    return ''.join([trans_letter(ch) for ch in word])


class FeatureClass:

    def __init__(self, file_path, threshold=4):
        self.history_list = []
        self.n_total_features = 0  # Total number of features accumulated
        self.threshold = threshold
        self.dic_features, self.labels = self.create_features(file_path)
        self.tot_matrix = self.create_total_matrix()

    def create_features(self, file_path):
        dic_features = {
            # word/tag features for all word/tag pairs
            'feature_100': defaultdict(int),

            # spelling features for prefixes/suffixes of length <= 4
            'feature_101': defaultdict(int),
            'feature_102': defaultdict(int),

            # Contextual Features
            'feature_103': defaultdict(int),
            'feature_104': defaultdict(int),
            'feature_105': defaultdict(int),
            'feature_106': defaultdict(int),
            'feature_107': defaultdict(int),

            'feature_words_with_capital_letters': defaultdict(int),
            'feature_words_with_numbers': defaultdict(int),

            'feature_words_with_hyphens': defaultdict(int),
            'feature_words_with_capitals_only': defaultdict(int),

            'feature_nnword_ctag': defaultdict(int),
            'feature_ppword_ctag': defaultdict(int),
            'feature_pword_ptag_ctag': defaultdict(int),
            'feature_ppword_pword_ctag': defaultdict(int),
            'feature_nnword_nword_ctag': defaultdict(int),

            'feature_100_lower': defaultdict(int),
            'feature_101_lower': defaultdict(int),
            'feature_101_lower_complete': defaultdict(int),
            'feature_102_lower': defaultdict(int),
            'feature_102_lower_complete': defaultdict(int),

            'upper_tag_exist': defaultdict(int),
            'hyphen_tag_exist': defaultdict(int),
            'digit_tag_exist': defaultdict(int),

            'cXXxx_dd': defaultdict(int),
            'nXXxx_dd': defaultdict(int),
            'pXXxx_dd': defaultdict(int),

            # 'stemmed_word_tag': defaultdict(int),

        }
        labels_set = set()
        with open(file_path) as f:
            for line in f:
                splited_words = re.split(' |\n', line)[:-1]
                pword, ppword, pptag, ptag = '*', '*', '*', '*'
                for word_idx in range(len(splited_words)):
                    cword, ctag = splited_words[word_idx].split('_')

                    create_mandatory_features(pptag, ptag, cword, ctag, dic_features)

                    if '-' in cword:
                        dic_features['feature_words_with_hyphens'][(cword, ctag)] += 1

                    if bool(re.search(r'[A-Z]', cword)) and not bool(re.search(r'[a-z]', cword)):
                        dic_features['feature_words_with_capitals_only'][ctag] += 1

                    if bool(re.search(r'[A-Z]', cword)):
                        dic_features['upper_tag_exist'][ctag] += 1

                    if '-' in cword:
                        dic_features['hyphen_tag_exist'][ctag] += 1

                    # feature 106
                    dic_features['feature_106'][(pword, ctag)] += 1

                    # feature 107
                    nword = splited_words[word_idx + 1].split('_')[0] if word_idx + 1 < len(splited_words) else '*'
                    dic_features['feature_107'][(nword, ctag)] += 1

                    # feature next next wort with cur tag
                    nnword = splited_words[word_idx + 2].split('_')[0] if word_idx + 2 < len(splited_words) else '*'

                    # feature 107 in tags
                    dic_features['feature_ppword_ctag'][(ppword, ctag)] += 1
                    dic_features['feature_nnword_ctag'][(nnword, ctag)] += 1

                    dic_features['feature_pword_ptag_ctag'][(pword, ptag, ctag)] += 1

                    features_lower_cases(cword, ctag, pword, nword, dic_features)

                    dic_features['cXXxx_dd'][(change_word(cword), ctag)] += 1
                    dic_features['nXXxx_dd'][(change_word(nword), ctag)] += 1
                    dic_features['pXXxx_dd'][(change_word(cword), ctag)] += 1
                    dic_features['feature_ppword_pword_ctag'][(ppword, pword, ctag)] += 1
                    dic_features['feature_nnword_nword_ctag'][(ppword, pword, ctag)] += 1
                    self.history_list.append((ppword, pword, cword, nword, nnword, pptag, ptag, ctag))
                    labels_set.add(ctag)

                    ppword = pword
                    pword = cword
                    pptag = ptag
                    ptag = ctag

        dic_features_id = {}
        for feature in dic_features:
            dic_features_id[feature] = {}
            for key in dic_features[feature]:
                if dic_features[feature][key] >= self.threshold:
                    dic_features_id[feature][key] = self.n_total_features
                    self.n_total_features += 1

        return dic_features_id, sorted(labels_set)

    def append_mandatory_features(self, history, features):
        """
        get history and feature list and append all the features that we were asked to implement and exist in the
        given history.
        """
        ppword, pword, cword, nword, nnword, pptag, ptag, ctag = history
        self.add_if_exist((cword, ctag), 'feature_100', features)  # feature 100
        self.add_if_exist((cword, ctag), 'feature_100_lower', features)  # feature 100
        # feature 101
        for i in range(1, min(6, len(cword) + 1)):
            self.add_if_exist((cword[-i:], ctag), 'feature_101', features)  # feature 101
            self.add_if_exist((cword[:i], ctag), 'feature_102', features)  # feature 102

        self.add_if_exist((pptag, ptag, ctag), 'feature_103', features)  # feature 103
        self.add_if_exist((ptag, ctag), 'feature_104', features)  # feature 104
        self.add_if_exist(ctag, 'feature_105', features)  # feature 105
        self.add_if_exist((pword, ctag), 'feature_106', features)  # feature 106
        self.add_if_exist((nword, ctag), 'feature_107', features)  # feature 107
        # feature words with capital letters
        self.add_if_exist((cword, ctag), 'feature_words_with_capital_letters', features)
        # feature words with digits
        self.add_if_exist((cword, ctag), 'feature_words_with_numbers', features)

    def append_lower_cases_features(self, history, features):
        """
        get history and feature list and append all the features that we were asked to implement and exist in the
        given history.
        """
        ppword, pword, cword, nword, nnword, pptag, ptag, ctag = history
        cword = cword.lower()
        pword = pword.lower()
        nword = nword.lower()
        self.add_if_exist((cword, ctag), 'feature_100_lower', features)  # feature 100
        # feature 101
        for i in range(1, min(5, len(cword) + 1)):
            self.add_if_exist((cword[-i:], ctag), 'feature_101_lower', features)  # feature 101
            self.add_if_exist((cword[:-i], ctag), 'feature_101_lower_complete', features)  # feature 101
            self.add_if_exist((cword[:i], ctag), 'feature_102_lower', features)  # feature 102
            self.add_if_exist((cword[i:], ctag), 'feature_102_lower_complete', features)  # feature 102

        self.add_if_exist((pword, ctag), 'feature_106', features)  # feature 106
        self.add_if_exist((nword, ctag), 'feature_107', features)  # feature 107

    def add_if_exist(self, hist: tuple, feature_name: str, features: list):
        if hist in self.dic_features[feature_name]:
            features.append(self.dic_features[feature_name][hist])

    def represent_input_with_features(self, history) -> list:
        """
                Extract feature vector in per a given history
                :param history: touple{word, pptag, ptag, ctag}
                :param word_tags_dict: word\tag dict
                    Return a list with all features that are relevant to the given history
            """
        ppword, pword, cword, nword, nnword, pptag, ptag, ctag = history

        features = []
        self.append_mandatory_features(history, features)
        self.append_lower_cases_features(history, features)
        self.add_if_exist((ppword, ctag), 'feature_ppword_ctag', features)
        self.add_if_exist((nnword, ctag), 'feature_nnword_ctag', features)

        self.add_if_exist((pword, ptag, ctag), 'feature_pword_ptag_ctag', features)

        self.add_if_exist((cword, ctag), 'feature_words_with_hyphens', features)
        if bool(re.search(r'[A-Z]', cword)) and not bool(re.search(r'[a-z]', cword)):
            self.add_if_exist(ctag, 'feature_words_with_capitals_only', features)

        if bool(re.search(r'[A-Z]', cword)):
            self.add_if_exist(ctag, 'upper_tag_exist', features)
        if '-' in cword:
            self.add_if_exist(ctag, 'hyphen_tag_exist', features)

        if bool(re.search(r'[A-Z]', cword)):
            self.add_if_exist(ctag, 'digit_tag_exist', features)

        self.add_if_exist((change_word(cword), ctag), 'cXXxx_dd', features)
        self.add_if_exist((change_word(nword), ctag), 'nXXxx_dd', features)
        self.add_if_exist((change_word(pword), ctag), 'pXXxx_dd', features)

        self.add_if_exist((ppword, pword, ctag), 'feature_ppword_pword_ctag', features)
        self.add_if_exist((nnword, nword, ctag), 'feature_nnword_nword_ctag', features)

        return features

    def create_total_matrix(self) -> csr_matrix:
        true_indices, rows, cols = [], [], []
        counter = 0
        repeats = []
        for history in self.history_list:
            for i, tmp_tag in enumerate(self.labels):
                if self.labels[i] == history[-1]:
                    true_indices.append(counter)
                feat = self.represent_input_with_features(history[:-1] + tuple([tmp_tag]))
                cols.extend(feat)
                repeats.append(len(feat))
                # rows.extend((counter + np.zeros_like(feat)).tolist())
                counter += 1
        self.true_indices = true_indices
        rows = np.repeat(np.arange(counter), repeats)
        return csr_matrix((np.ones_like(cols), (rows, cols)), shape=(counter, self.n_total_features), dtype=bool)
