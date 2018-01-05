import numpy as np
import abc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class Optimizer(object):
    """ The base class used for all optimizers
    Args:
        model: a model (currently supports scikit-learn, xgboost, or a class 
               derived from optml.models.Model)
        hyperparams: a list of Parameter instances
        eval_func: loss function to be minimized. Takes input (y_true, y_predicted) where 
                    y_true and y_predicted are numpy arrays

    Attributes:
        model: a model (currently supports scikit-learn, xgboost, or a class 
               derived from optml.models.Model)
        hyperparam_history: a list of dictionaries with parameters and scores
        hyperparams: the list of parameters that the model is optimized over
        eval_func: loss function to be minimized
        model_module: can be 'sklearn', 'pipeline', 'xgboost', 'keras' or user-defined model
        param_dict: dictionary where key=parameter name and value is the Parameter instance
    """

    def __init__(self, model, hyperparams, eval_func):
        self.model = model
        self.hyperparam_history = []
        self.hyperparams = hyperparams
        self.eval_func = eval_func
        self.model_module = self.infer_model_type(model)
        self.param_dict = {p.name:p for p in hyperparams}
        
    def infer_model_type(self, model):
        """ Identifies the type of model
        Args:
            model: a model (currently supports scikit-learn, xgboost, or a class 
               derived from optml.models.Model)

        Returns:
            string: the type of the model; can be 'sklearn', 'pipeline', 'xgboost', 'keras' 
                    or a user-defined model
        
        """
        if 'xgboost' in model.__module__.lower():
            return 'xgboost'
        elif 'pipeline' in model.__module__.lower():
            return 'pipeline'
        elif 'sklearn' in model.__module__.lower():
            return 'sklearn'
        elif (hasattr(model, '__model_module__')) and ('keras' in model.__model_module__.lower()):
            return 'keras'
        else:
            raise NotImplementedError("{} not implemented for module '{}'".format(
                    str(type(self))[:-2].split('.')[-1], model.__module__))

    @abc.abstractmethod
    def get_next_hyperparameters(self):
        """ Returns a hyperparameter based on previous iterations
        """
        raise NotImplementedError("This class needs a get_next_hyperparameters(...) function")

    @abc.abstractmethod
        """ Run the optimizer for some data
        Args:
            X: numpy array with features
            y: numpy array with labels
            params: dictionary with additional parameters used for fitting the model 
        """
    def fit(self, X, y, params):
        raise NotImplementedError("This class needs a self.fit(X, y, params) function")

    def build_new_model(self, new_hyperparams):
        """ Creates a new instance of the model with given parameters
        Args:
            new_hyperparams: dictionary of parameter name and value
            
        Returns:
            a model (currently supports scikit-learn, xgboost, or a class 
               derived from optml.models.Model)
        """
        if self.model_module == 'pipeline':
                new_model = self.model.set_params(**new_hyperparams)
        elif (self.model_module == 'sklearn') or (self.model_module == 'xgboost'):
            new_model = self.model.__class__(**new_hyperparams)
        elif self.model_module == 'statsmodels':
            raise NotImplementedError("Not yet implemented for 'statsmodels'")
            #new_model = self.model.__class__(**new_hyperparams)
            #new_model = ModelConverter(new_model).convert()
        elif self.model_module == 'keras':
            new_model = self.model.__class__(**new_hyperparams)
        else:
            raise NotImplementedError("{} not implemented for module '{}'".format(
                    str(type(self))[:-2].split('.')[-1], self.model_module))
        return new_model

    def get_best_params_and_model(self):
        """ Get the best parameters and model after running the optimizer
        Args:
            None

        Returns:
            a dictionary with the optimal value for each parameter
            a model (currently supports scikit-learn, xgboost, or a class 
               derived from optml.models.Model)
        """
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])
        best_params = self.hyperparam_history[best_params_idx][1]
        if isinstance(self.model, Pipeline):
            all_params = self.model.get_params()
            all_params.update(best_params)
            best_model = self.model.set_params(**all_params)
        else:
            best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model

class MissingValueException(Exception):
    pass

class Parameter(object):
    def __init__(self, name, param_type, lower=None, upper=None, possible_values=None, distribution=None):
        """
        Args:
            name: string specifying the name of this parameter
            param_type: 'categorical', 'continuous', 'integer', 'boolean', 'int_array' or 'continuous_array'
            lower: lower bound of parameter (only applicable to numerical types)
            upper: upper bound of parameter (only applicable to numerical types)
            possible_values: list of possible values a parameter can take (only applicable to categorical type)
            distribution: not actually implemented yet; specifies a distribution to sample from 
                          (only applicable to continuous types)
        """
        param_type = param_type.lower()
        if not param_type in ['categorical', 'continuous', 'integer', 'boolean', 'int_array', 'continuous_array']:
            raise ValueError("param_type needs to be 'categorical','continuous','integer', 'int_array', 'continuous_array' or 'boolean'")
        if (param_type == 'categorical') and (possible_values is None):
            raise MissingValueException("Need to provide possible values for categorical parameters.")
        self.possible_values = possible_values
        self.param_type = param_type.lower()
        if (param_type in ['continuous', 'integer', 'int_array', 'continuous_array']) and (
            (lower is None) or (upper is None)):
            raise MissingValueException("Need to provide 'lower' and 'upper' for parameters of type.".format(
                param_type))
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
        """ Returns a uniformly random sample of the parameter
        Args:
            None

        Returns:
            a value of the parameter (integer for type 'integer' etc.)
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
            return [np.random.uniform(self.lower[i],self.upper[i]) for i in range(len(self.lower))]
        elif self.param_type == 'int_array':
            return [np.random.choice(np.arange(self.lower[i],self.upper[i]),1)[0] for i in range(len(self.lower))]

