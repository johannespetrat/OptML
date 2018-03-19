import numpy as np
import abc
import copy
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from types import ModuleType

class Optimizer(object):

    def __init__(self, model, hyperparams, eval_func, start_vals=None):
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
        self.model_module = self.infer_model_type(model)
        self.param_dict = {p.name:p for p in hyperparams}
        self.start_vals = start_vals
        
    def infer_model_type(self, model):
        if ('train' in dir(model)) and (isinstance(model, ModuleType)) and (
                model.__repr__().startswith("<module 'xgboost' from")):
            return 'xgboost'
        elif 'xgboost' in model.__module__.lower():
            return 'xgboost_sklearn'
        elif 'pipeline' in model.__module__.lower():
            return 'pipeline'
        elif 'sklearn' in model.__module__.lower():
            return 'sklearn'
        elif (hasattr(model, '__model_module__')) and ('keras' in model.__model_module__.lower()):
            return 'keras'
        else:
            raise NotImplementedError("{} not implemented for module '{}'".format(
                    str(type(self))[:-2].split('.')[-1], model.__module__))

    def get_kfold_split(self, n_folds, X):
        """
        Splits X into n_folds folds

        Args:
            n_folds: integer specifying number of folds
            X: data to be split

        Returns:
            a generator with tuples of form (train_idxs, test_idxs)
        """
        kf = KFold(n_splits=n_folds)
        return kf.split(X)


    @abc.abstractmethod
    def get_next_hyperparameters(self):
        raise NotImplementedError("This class needs a get_next_hyperparameters(...) function")

    def _fit_and_score_model(self, params, X_train, y_train, X_test, y_test, n_folds):
        """
        For a given set of parameters fit the model on (X_train, y_train) and score it
        on (X_test, y_test). 

        Args:
            params: parameters for the model
            X_train: a numpy array with training data. each row corresponds to a data point
            y_train: a numpy array containing the target variable for the training data
            X_test: a numpy array with validation data. each row corresponds to a data point
            y_test: a numpy array containing the target variable for the validation data
            n_folds: the number of folds for K-fold cross-validation
        Returns:
            the value of self.eval_func evaluated on (X_test, y_test)
        """
        if n_folds is not None:
            splits = self.get_kfold_split(n_folds, X_train)
            scores = []
            for train_idxs, test_idxs in splits:
                if self.model_module == 'xgboost':
                    dtrain = self.convert_to_xgboost_dataset(X_train[train_idxs], y_train[train_idxs])
                    dtest = self.convert_to_xgboost_dataset(X_train[test_idxs], y_train[test_idxs])
                    fitted_model = self.model.train(params, dtrain, evals=[(dtest, 'test')],
                        verbose_eval=False)
                    y_pred = fitted_model.predict(dtest)
                else:
                    new_model = self.build_new_model(params)
                    new_model.fit(X_train[train_idxs], y_train[train_idxs])
                    y_pred = new_model.predict(X_train[test_idxs])
                scores.append(self.eval_func(y_train[test_idxs], y_pred))
                score = np.mean(scores)
        else:
            if self.model_module == 'xgboost':
                dtrain = self.convert_to_xgboost_dataset(X_train, y_train)
                dtest = self.convert_to_xgboost_dataset(X_test, y_test)
                fitted_model = self.model.train(params, dtrain, evals=[(dtest, 'test')],
                        verbose_eval=False)
                y_pred = fitted_model.predict(dtest)
            else:
                new_model = self.build_new_model(params)
                new_model.fit(X_train, y_train)
                y_pred = new_model.predict(X_test)
            score = self.eval_func(y_test, y_pred)
        return score

    def check_parameters(self, params):
        """
        Checks if all parameters are within their bounds and have the correct type.

        Args:
            params: a dictionary with parameter values

        Returns:
            None
        """
        for param in self.hyperparams:
            if param.name in params.keys():
                val = params[param.name]
                if isinstance(val, str):
                    assert(param.param_type=='categorical')
                    assert(val in param.possible_values)
                elif isinstance(val, bool):
                    assert(param.param_type=='boolean')
                    assert((val==True) or (val==False))
                elif isinstance(val, float):
                    assert(param.param_type=='continuous')
                    assert(val>=param.lower)
                    assert(val<=param.upper)
                else:
                    assert((param.param_type=='continuous') or (param.param_type=='integer'))
                    assert(val>=param.lower)
                    assert(val<=param.upper)
                    assert(round(val,0)==val)

    @abc.abstractmethod
    def fit(self, X, y, params):
        raise NotImplementedError("This class needs a self.fit(X, y, params) function")

    def build_new_model(self, new_hyperparams):
        if self.model_module == 'pipeline':
                new_model = self.model.set_params(**new_hyperparams)
        elif (self.model_module == 'sklearn') or (self.model_module == 'xgboost_sklearn'):
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


    def get_random_values_arr(self):
        """
        Generates a numpy array with randomly sampled values for 
        each hyperparameter. Note that the order is the same as in 
        self.hyperparams.

        Args:
            None

        Returns:
            a numpy array with potentially mixed types
        """
        start_vals = []
        for hp in self.hyperparams:
            start_vals.append(hp.random_sample())
        return np.array([start_vals], dtype=object)


    def get_random_values_dict(self):
        """
        Generates a dictionary with randomly sampled values for 
        each hyperparameter.

        Args:
            None

        Returns:
            a dictionary with parameter names as keys
        """
        start_vals = {hp.name:hp.random_sample() for hp in self.hyperparams}
        return start_vals


    def get_default_values_dict(self):
        """
        Returns a dictionary with the provided default values for 
        each hyperparameter.

        Args:
            None

        Returns:
            a dictionary with parameter names as keys
        """
        if self.start_vals is not None:
            start_vals = copy.deepcopy(self.start_vals)
            for hp in self.hyperparams:
                if hp.name not in start_vals.keys():
                    start_vals[hp.name] = hp.random_sample()
            return start_vals
        else:
            raise Exception("You need to provide 'start_vals' when initializing the optimizer!")


    def get_default_values_arr(self):
        """
        Generates a numpy array with default values for 
        each hyperparameter. Note that the order is the same as in 
        self.hyperparams. If a parameter has no default value provided then
        it will be sampled randomly.

        Args:
            None

        Returns:
            a numpy array with potentially mixed types
        """
        if self.start_vals is not None:
            start_vals = []
            for hp in self.hyperparams:
                if hp.name in self.start_vals.keys():
                    start_vals.append(self.start_vals[hp.name])
                else:
                    start_vals.append(hp.random_sample())
            return np.array([start_vals], dtype=object)
        else:
            raise Exception("You need to provide 'start_vals' when initializing the optimizer!")


    def get_best_params_and_model(self):
        """
        Returns the best parameters and model after optimization.
        Keyword arguments:
            None
        """
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])
        best_params = self.hyperparam_history[best_params_idx][1]
        if isinstance(self.model, Pipeline):
            all_params = self.model.get_params()
            all_params.update(best_params)
            best_model = self.model.set_params(**all_params)
        elif self.model_module=='xgboost':
            return best_params, self.model
        else:
            best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model

    def convert_to_xgboost_dataset(self, data, label):
        return xgb.DMatrix(data, label)

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
        """
        returns a uniformly random sample of the parameter
        Keywords:
            None
        """
        if self.param_type == 'integer':
            return np.random.choice(np.arange(self.lower, self.upper+1, 1))
        elif self.param_type == 'categorical':
            return str(np.random.choice(self.possible_values))
        elif self.param_type == 'continuous':
            return np.random.uniform(self.lower, self.upper)
        elif self.param_type == 'boolean':
            return np.random.choice([True, False])
        elif self.param_type == 'continuous_array':
            return [np.random.uniform(self.lower[i],self.upper[i]) for i in range(len(self.lower))]
        elif self.param_type == 'int_array':
            return [np.random.choice(np.arange(self.lower[i],self.upper[i]),1)[0] for i in range(len(self.lower))]

