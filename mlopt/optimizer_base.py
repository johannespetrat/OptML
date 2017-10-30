import numpy as np
import abc

class Optimizer(object):

    def __init__(self, model, hyperparams, eval_func):
        """
        Keyword arguments:
            model - a model as specified in the readme
            hyperparams - a list of Parameter instances
            eval_func - scoring function to be minimized. Takes input (y_true, y_predicted) where 
                        y_true and y_predicted are numpy arrays
        """
        self.model = model
        self.hyperparam_history = []
        self.hyperparams = hyperparams
        self.eval_func = eval_func
        self.param_dict = {p.name:p for p in hyperparams}

    @abc.abstractmethod
    def get_next_hyperparameters(self):
        raise NotImplementedError("This class needs a get_next_hyperparameters(...) function")

    @abc.abstractmethod
    def fit(self, X, y, params):
        raise NotImplementedError("This class needs a self.fit(X, y, params) function")

    def getParamType(self, parameter_name):
        return self.param_dict[parameter_name].param_type

    def build_new_model(self, new_hyperparams):
        """
        builds a model given a dictionary of parameters. Note that if parameters are not specified
        the model assumes that previous values remain unchanged.
        Keyword arguments:
            new_hyperparams - a dictionary with parameters and their updated values
        """
        hyperparams = self.model.get_params()
        hyperparams.update(new_hyperparams)
        new_model = self.model.__class__(**hyperparams)
        return new_model

    def get_best_params_and_model(self):
        """
        Returns the best parameters and model after optimization.
        Keyword arguments:
            None
        """
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])
        best_params = self.hyperparam_history[best_params_idx][1]        
        best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model

class MissingValueException(Exception):
    pass

class Parameter(object):
    def __init__(self, name, param_type, lower=None, upper=None, possible_values=None, distribution=None):
        """
        Keywords:
            name - String specifying the name of this parameter
            param_type - 'categorical', 'continuous', 'integer', 'boolean', 'int_array' or 'continuous_array'
            lower - lower bound of parameter (only applicable to numerical types)
            upper - upper bound of parameter (only applicable to numerical types)
            possible_values - list of possible values a parameter can take (only applicable to categorical type)
            distribution - specifies a distribution to sample from (only applicable to continuous types); not actually implemented yet
        """
        param_type = param_type.lower()
        if not param_type in ['categorical', 'continuous', 'integer', 'boolean', 'int_array', 'continuous_array']:
            raise ValueError("param_type needs to be 'categorical','continuous','integer', 'int_array', 'continuous_array' or 'boolean'")
        if (param_type == 'categorical') and (possible_values is None):
            raise MissingValueException("Need to provide possible values for categorical parameters.")
        self.possible_values = possible_values
        self.param_type = param_type.lower()
        self.lower = lower
        self.upper = upper
        self.name = name
        if distribution is not None:
            self.distribution = distribution 
        if param_type.lower() in ['int_array', 'continuous_array']:
            if len(lower)!=len(upper):
                raise ValueError("'lower' and 'upper' must be of the same length.")
            self.size = len(lower)

    def random_sample(self):
        """
        returns a uniformly random sample of the parameter
        Keywords:
            None
        """
        if self.param_type == 'integer':
            return np.random.choice(np.arange(self.lower, self.upper+1, 1))
        elif self.param_type == 'categorical':
            return np.random.choice(self.possible_values)
        elif self.param_type == 'continuous':
            return np.random.uniform(self.lower, self.upper)
        elif self.param_type == 'boolean':
            return np.random.choice([True, False])
        elif self.param_type == 'continuous_array':
            return [np.random.uniform(self.lower[i],self.upper[i])[0] for i in range(len(self.lower))]
        elif self.param_type == 'int_array':
            return [np.random.choice(np.arange(self.lower[i],self.upper[i]),1)[0] for i in range(len(self.lower))]

