from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance
from sklearn. preprocessing import normalize
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import pandas as pd
from itertools import combinations


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

import numpy as np

EPS = 1e-20
EPS_LOG = 1e-3
THRESHOLD = 0.2

class FeatureGenerationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, thr=THRESHOLD, features_mask=None, important_features=None, stand_gen=True, corr_gen=True):
        self.thr = thr
        self._features_mask = features_mask
        self._important_features = important_features
        self._stand_gen = stand_gen
        self._corr_gen = corr_gen
        self.desc_dict = {}

    #Construction of a matrix of correlation coefficients of features
    def _correlation_create(self, X):
        return np.corrcoef(X.T)

    def fit(self, X, y=None):
        if self._features_mask is None:
            self._features_mask = np.arange(X.shape[1])
        else:
            self._features_mask = np.array(self._features_mask)

        if self._important_features is None:
            self._important_features = self._features_mask
        else:
            self._important_features = np.array(self._important_features)

        if self._important_features.size < 2:
            self._corr_gen = False

        if self._corr_gen:
            self._corr_mat = self._correlation_create(X[:, self._important_features])

        return self

    #Generating features using standard functions
    def _standard_generation(self, X):
        features_mask = self._features_mask
        important_features = self._important_features

        if features_mask.size:
            #exponent
            self.desc_dict['exp'] = [features_mask, np.arange(X.shape[1], X.shape[1] + features_mask.shape[0])]
            X = np.concatenate([X, np.exp(np.clip(X[:,features_mask], -750, 700))], axis=1)

            #x^2
            self.desc_dict['^2'] = [features_mask, np.arange(X.shape[1], X.shape[1] + features_mask.shape[0])]
            X = np.concatenate([X, np.power(X[:,features_mask], 2)], axis=1)

            #x^3
            self.desc_dict['^3'] = [features_mask, np.arange(X.shape[1], X.shape[1] + features_mask.shape[0])]
            X = np.concatenate([X, np.power(X[:,features_mask], 3)], axis=1)


        if important_features.size:
            #logarithm
            self.desc_dict['log'] = [important_features, np.arange(X.shape[1], X.shape[1] + important_features.shape[0])]
            X = np.concatenate([X, np.where(X[:,important_features] <= -1,
                                                    np.log(EPS_LOG),
                                                    np.log(X[:,important_features] + 1,
                                                          where=X[:,important_features] > -1))], axis=1)

            #x^0.5
            self.desc_dict['^0.5'] = [important_features, np.arange(X.shape[1], X.shape[1] + important_features.shape[0])]
            X = np.concatenate([X, np.where(X[:,important_features] < 0,
                                                    -np.power(-X[:,important_features], 0.5,
                                                              where=X[:,important_features] < 0),
                                                    np.power(X[:,important_features], 0.5,
                                                              where=X[:,important_features] >= 0))], axis=1)

        return X

    #Generating features from two that have a correlation coefficient less than the threshold
    def _correlation_generation(self, X):
        important_features = self._important_features
        X_features = X[:, important_features]

        pairs_indxs_mat = np.array(list(combinations(range(self._corr_mat.shape[0]), 2)))

        #x1 * x2
        self.desc_dict['*'] = [important_features[pairs_indxs_mat], np.arange(X.shape[1], X.shape[1] + pairs_indxs_mat.shape[0])]
        X = np.concatenate([X, np.prod(X_features.T[pairs_indxs_mat], axis=1).T], axis=1)

        #x1 / x2, x2 / x1
        self.desc_dict['/'] = [np.hstack([important_features[pairs_indxs_mat], important_features[pairs_indxs_mat][:,::-1]]).reshape(-1, 2),
                               np.arange(X.shape[1], X.shape[1] + 2 * pairs_indxs_mat.shape[0])]
        X = np.concatenate([X, (X_features.T[pairs_indxs_mat] / (X_features.T[pairs_indxs_mat][:,::-1] + EPS)).reshape(pairs_indxs_mat.shape[0] * 2,
                                                                                                                                           X_features.shape[0]).T], axis=1)

        pairs_indxs_mat = np.array([[[i, j] for j in range(self._corr_mat.shape[1])] for i in range(self._corr_mat.shape[0])])
        pairs_indxs_mat = pairs_indxs_mat[abs(self._corr_mat) <= self.thr]
        pairs_indxs_mat = pairs_indxs_mat[pairs_indxs_mat[:, 0] > pairs_indxs_mat[:, 1]]

        if pairs_indxs_mat.size:
            #x1 + x2
            self.desc_dict['+'] = [important_features[pairs_indxs_mat], np.arange(X.shape[1], X.shape[1] + pairs_indxs_mat.shape[0])]
            X = np.concatenate([X, np.sum(X_features.T[pairs_indxs_mat], axis=1).T], axis=1)

            #x1 - x2, x2 - x1
            self.desc_dict['-'] = [np.hstack([important_features[pairs_indxs_mat], important_features[pairs_indxs_mat][:,::-1]]).reshape(-1, 2),
                                   np.arange(X.shape[1], X.shape[1] + 2 * pairs_indxs_mat.shape[0])]
            X = np.concatenate([X, (X_features.T[pairs_indxs_mat] - X_features.T[pairs_indxs_mat][:,::-1]).reshape(pairs_indxs_mat.shape[0] * 2,
                                                                                                                                      X_features.shape[0]).T], axis=1)

        return X

    def transform(self, X):
        if self._stand_gen:
            X = self._standard_generation(X)
        if self._corr_gen:
            X = self._correlation_generation(X)

        return X


if __name__ == '__main__':
    X, y = make_classification(
    n_samples=100000, n_features=100, n_informative=80, n_redundant=2,
    random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()

    result = scaler.fit_transform(X_train)
    feat_gen = FeatureGenerationTransformer(thr=0.01,  important_features=[1, 2, 3, 4, 5, 6, 7, 8, 9, 92])
    start_time = time.time()
    result_2 = feat_gen.fit_transform(result)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Время генерации: {elapsed_time:.5f}')
