import abc
from sklearn.base import BaseEstimator

class Model(BaseEstimator):
    def __init__(self):
        raise NotImplementedError("You need to implement the initialisation function for this model!")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("You need to implement the 'get_params' function for this model!")

class KerasModel(Model):
    __model_module__ = 'keras'
    def __init__(self):
        raise NotImplementedError("You need to implement the initialisation function for this model! " + 
                                  "It should at least specify 'batch_size' and the number of epochs.")

    def fit(self, X, y, verbose=0):
        return self.model.fit(X,y, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)

    def predict(self, X):
        return self.model.predict(X)
