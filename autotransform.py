class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'work_time' in kwargs.keys():
            self.work_time = kwargs['work_time']
        else:
            self.work_time = 1
        if 'val_size' in kwargs.keys():
            self.val_size = kwargs['val_size']
        else:
            self.val_size = 0.2
        if 'random_state' in kwargs.keys():
            self.random_state = kwargs['random_state']
        else:
            self.random_state = 42

    def fit(self, X, y=None):
        obj_count, f_dim = X.shape[0], X.shape[1]

        if 'encoder' in self.kwargs.keys():
            self.encoder = CategoricalEncoder(**self.kwargs['encoder'])
        else:
            self.encoder = CategoricalEncoder()
        # self.encoder = preprocessing.StandardScaler() # TODO add normalization after generation
        if 'generator' in self.kwargs.keys():
            self.generator = FeatureGenerationTransformer(
                **self.kwargs['generator'])
        else:
            self.generator = FeatureGenerationTransformer()
        if 'selector' in self.kwargs.keys():
            self.selector = FeatureSelectionTransformer(
                k=f_dim, **self.kwargs['selector'])
        else:
            self.selector = FeatureSelectionTransformer(k=f_dim)
        self.encoder.fit(X, y)
        X_trrrrrrr, X_val, y_trrrrrrr, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state)
        # Cross validation (2 blocks)
        self.need_to_generate = 1.15*1e-7 * obj_count * f_dim**2 <= self.work_time
        if self.need_to_generate:
            X_enc = self.encoder.transform(X_val)
            self.generator.fit(X_enc, y_val)
            X_tte = self.generator.transform(X_enc)
            self.selector.fit((X_tte), y_val)
        return X_trrrrrrr, y_trrrrrrr

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X = self.encoder.transform(X)
        if self.need_to_generate:
            X = self.generator.transform(X)
            X = self.selector.transform(X)
        return X
