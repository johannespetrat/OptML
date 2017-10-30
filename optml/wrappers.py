from sklearn.base import BaseEstimator

class StatsmodelsWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, X):
        params = self.get_params()
        params.update({'endog': X})
        print(params)
        self.model = self.model.__class__(**params)
        self.model.fit()

    def get_params(self, deep=False):
        params = {}
        try:
            params['order'] = self.model.order
        except AttributeError:
            pass
        try:
            params['seasonal_order'] = self.model.seasonal_order
        except AttributeError:
            pass
        try:
            params['order'] = self.model.order
        except AttributeError:
            pass
        return params

    def build_new_model(self, new_params):
        self.model = self.model.__class__(**params)
        

class KerasWrapper(BaseEstimator):
    def __init__(self, config):
        self.model = Model.from_config(config)

    def fit(self, X, y):
        self.model.fit(X,y)

    def get_params(self):
        self.model.get_config()
