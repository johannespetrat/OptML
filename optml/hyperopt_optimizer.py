import numpy as np
import warnings

import sklearn.gaussian_process as gp
from scipy.optimize import minimize
from scipy.stats import norm
from hyperopt import hp, fmin, tpe, Trials

from optml.optimizer_base import Optimizer, MissingValueException


class HyperoptOptimizer(Optimizer):
    """    
    """
    def __init__(self, model, hyperparams, eval_func):
        super(HyperoptOptimizer, self).__init__(model, hyperparams, eval_func)
        self.eval_func = eval_func
        self.bounds_arr = np.array([[param.lower, param.upper] for param in self.hyperparams])
        self.param_space = self.list_to_param_space(hyperparams)

    def list_to_param_space(self, params):
        param_space = {}
        for param in params:
            if param.param_type == 'integer':
                param_space[param.name] = param.lower + hp.randint(param.name, param.upper-param.lower)
            elif param.param_type == 'categorical':
                param_space[param.name] = hp.choice(param.name, param.possible_values)
            elif param.param_type == 'boolean':
                param_space[param.name] = hp.choice(param.name, [True, False])
            elif param.param_type == 'continuous':
                param_space[param.name] = hp.uniform(param.name, param.lower, param.upper)
            else:
                raise ValueError("HyperOpt only takes parameters of type 'integer', 'categorical', 'boolean' and 'continuous'")
        return param_space


    def fit(self, X_train, y_train, X_test=None, y_test=None, n_iters=10, start_vals=None):
        """
        """
        if (X_test is None) and (y_test is None):
            X_test = X_train
            y_test = y_train
        elif (X_test is None) or (y_test is None):
            raise MissingValueException("Need to provide 'X_test' and 'y_test'")

        def objective(params):
            model_params = self.model.get_params()
            model_params.update(params)
            self.model = self.build_new_model(model_params)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_true = y_test
            score = -self.eval_func(y_true, y_pred)
            return score
        
        self.trials = Trials()
        best_params = fmin(objective,
                    self.param_space,
                    algo=tpe.suggest,
                    max_evals=n_iters,
                    trials=self.trials)

        self.hyperparam_history = []
        for i, loss in enumerate(self.trials.losses()):
            param_vals = {k:v[i] for k,v in self.trials.vals.items()}
            self.hyperparam_history.append((-loss, param_vals))

        model_params = self.model.get_params()
        model_params.update(best_params)
        best_model = self.build_new_model(model_params)
        return best_params, best_model
