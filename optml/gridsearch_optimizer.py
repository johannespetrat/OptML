import numpy as np
import warnings
import itertools
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
from optml.optimizer_base import Optimizer, MissingValueException

def objective(model, model_module, eval_func, X_train, y_train, X_test, y_test, params):
    model_params = model.get_params()
    model_params.update(params)
    if model_module == 'pipeline':
            new_model = model.set_params(**model_params)
    elif (model_module == 'sklearn') or (model_module == 'xgboost'):
        new_model = model.__class__(**model_params)
    elif model_module == 'statsmodels':
        raise NotImplementedError("Not yet implemented for 'statsmodels'")
    elif model_module == 'keras':
        new_model = model.__class__(**model_params)

    new_model.fit(X_train, y_train)
    y_pred = new_model.predict(X_test)
    y_true = y_test
    score = eval_func(y_true, y_pred)
    return (score, params)

class GridSearchOptimizer(Optimizer):
    """
    """
    def __init__(self, model, hyperparams, eval_func, grid_sizes, n_jobs=1):
        super(GridSearchOptimizer, self).__init__(model, hyperparams, eval_func)
        self.eval_func = eval_func
        self.bounds_arr = np.array([[hp.lower, hp.upper] for hp in self.hyperparams])
        self.n_jobs = n_jobs
        self.grid = np.array(self.build_grid(grid_sizes))
        #self.split_grid_across_jobs(grid)

    def build_grid(self, grid_sizes):
        grid_dict = {}
        for param_name, param in self.param_dict.items():
            if param.param_type == 'continuous':
                grid_dict[param_name] = np.linspace(param.lower, param.upper, grid_sizes[param_name])
            elif param.param_type == 'integer':
                step_size = int(round((param.upper - param.lower)/float(grid_sizes[param_name])))
                grid_dict[param_name] = np.concatenate([np.arange(param.lower, param.upper, step_size), [param.upper]])
            elif param.param_type == 'categorical':
                grid_dict[param_name] = param.possible_values
            elif param.param_type == 'boolean':
                grid_dict[param_name] = [True, False]
        # now build the grid as a list with all possible combinations i.e. the cartesian product
        grid = []
        for params in list(itertools.product(*[[(k,v) for v in vals] for k, vals in grid_dict.items()])):
            grid.append(dict(params))
        return grid

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """
        """
        if (X_test is None) and (y_test is None):
            X_test = X_train
            y_test = y_train
        elif (X_test is None) or (y_test is None):
            raise MissingValueException("Need to provide 'X_test' and 'y_test'")

        fun = partial(objective, deepcopy(self.model), 
                                 deepcopy(self.model_module), 
                                 deepcopy(self.eval_func),
                                 X_train, y_train, X_test, y_test)
        pool = Pool(self.n_jobs)
        scores = pool.map(fun, deepcopy(self.grid))
        self.hyperparam_history = scores

        best_params, best_model = self.get_best_params_and_model()
        return best_params, best_model
