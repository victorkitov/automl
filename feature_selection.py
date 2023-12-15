class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, typ='skb', k=50, task_type='class', n_estimators=500):
        if task_type == 'class':
            if typ == 'skb':
                self.predictor = SelectKBest(
                    score_func=mutual_info_classif, k=k)
            elif typ == 'rtree':
                self.predictor = SelectFromModel(ensemble.RandomForestClassifier(
                    n_estimators=n_estimators), max_features=k)
            elif typ == 'chi2':
                self.predictor = CHI2Selection(k=k)
            elif typ == 'lsvm':
                self.predictor = SelectFromModel(svm.LinearSVC(
                    C=0.1, penalty="l1", dual=False), max_features=k)

    def fit(self, X, y=None):
        return self.predictor.fit(X, y)

    def transform(self, X):
        return self.predictor.transform(X)

    def get_support(self, indices):
        return self.predictor.get_support(indices=indices)
