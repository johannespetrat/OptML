import abc
from sklearn.base import BaseEstimator
class Model(BaseEstimator):
    def __init__(self):
        raise NotImplementedError("You need to implement the initialisation function for this model!")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("You need to implement the 'get_params' function for this model!")

class KerasModel(Model):
    def __init__(self):
        raise NotImplementedError("You need to implement the initialisation function for this model! " + 
                                  "It should at least specify 'batch_size' and the number of epochs.")

    def fit(self, X, y, verbose=0):
        return self.model.fit(X,y, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)

    def predict(self, X):
        return self.model.predict(X)


class ModelConverter(object):
    def __init__(self, model):
        self.model = model

    def statsmodels_get_params(self):
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

    def statsmodels_fit_safe(self, X_train):
        """
        overwrites the fit function
        """                
        hyperparams = self.model.get_params()
        hyperparams.update({'endog':X_train})
        from nose.tools import set_trace; set_trace()
        self.model = self.model.__class__(**hyperparams)
        try:            
            fitted_model = self.model.__class__.fit(self.model)
        except ValueError:
            hyperparams.update({"enforce_invertibility":False, "enforce_stationarity":False})
            new_model = self.model.__class__(**hyperparams)
            fitted_model = self.model.__class__.fit(new_model)
            from nose.tools import set_trace; set_trace()
        return fitted_model

    def convert_statsmodels(self):
        self.model.get_params = self.statsmodels_get_params
        self.model.fit = self.statsmodels_fit_safe
        return self.model

    def convert(self):
        """
        create scikit-learn interface for all models
        """
        model_module = self.model.__module__.split('.')[0]
        if model_module == 'statsmodels':
            return self.convert_statsmodels()
        else:
            return self.model