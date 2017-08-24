import numpy as np
import abc

class Optimizer(object):

    def __init__(self, model, hyperparams, eval_func):        
        self.model = model
        self.hyperparam_history = []
        self.eval_func = eval_func

    @abc.abstractmethod
    def get_next_hyperparameters(self):
        raise NotImplementedError("This class needs a get_next_hyperparameters(...) function")

    @abc.abstractmethod
    def fit(self, X, y, params):
        raise NotImplementedError("This class needs a self.fit(X, y, params) function")

    def build_new_model(self, new_hyperparams):
        hyperparams = self.model.get_params()
        hyperparams.update(new_hyperparams)
        new_model = self.model.__class__(**hyperparams)
        return new_model

    def get_best_params_and_model(self):
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])
        best_params = self.hyperparam_history[best_params_idx][1]        
        best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model

class MissingValueException(Exception):
    pass

class Parameter(object):
    def __init__(self, name, param_type, lower, upper, possible_values=None, distribution=None):
        # continuous, categorical, binary, integer
        param_type = param_type.lower()
        if not param_type in ['categorical', 'continuous', 'integer', 'boolean']:
            raise ValueError("param_type needs to be 'categorical','continuous','integer' or 'boolean'")
        if (param_type == 'categorical') and (possible_values is None):
            raise MissingValueException("Need to provide possible values for categorical parameters.")
        self.possible_values = possible_values
        self.param_type = param_type.lower()
        self.lower = lower
        self.upper = upper
        self.name = name
        if distribution is not None:
            self.distribution = distribution 