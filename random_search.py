import numpy as np
from optimizer_base import Optimizer, MissingValueException

class RandomSearchOptimizer(Optimizer):
    def __init__(self, model, eval_func, hyperparams, grid_size, categorical_possible_values=None):        
        self.model = model
        self.eval_func = eval_func
        self.hyperparam_history = []
        self.hyperparams = hyperparams
        self.categorical_possible_values = categorical_possible_values
        self.hyperparams_grid = self.build_param_grid(grid_size)     

    def build_param_grid(self, grid_size):
        grid = {}
        for h in self.hyperparams:
            if h.param_type == 'integer':
                stepsize = int(round((h.upper - h.lower)/float(grid_size)))
                if stepsize == 0:
                    stepsize = 1
                param_range = np.arange(h.lower, h.upper, stepsize)                
            elif h.param_type == 'continuous':                
                param_range = np.linspace(h.lower, h.upper, grid_size)
            elif h.param_type == 'categorical': 
                try:
                    param_range = self.categorical_possible_values[h.name]               
                except KeyError:
                    raise MissingValueException("Need to provide possible values for the parameter '{}'".format(h.name))
            elif h.param_type == 'boolean':
                param_range = [True, False]
            grid[h.name] = param_range
        return grid

    def get_next_hyperparameters(self):
        new_hyperparams = {}
        for key, val_range in self.hyperparams_grid.items():
            new_hyperparams[key] = np.random.choice(val_range)
        return new_hyperparams

    def fit(self, X_train, y_train, X_test, y_test, n_iters):
        # get the hyperparameters of the base model
        hyperparams = self.model.get_params()
        # and update them with the new hyperparameters
        for i in range(n_iters):
            new_hyperparams = self.get_next_hyperparameters()
            hyperparams.update(new_hyperparams)
            new_model = self.model.__class__(**hyperparams)

            new_model.fit(X_train,y_train)
            score = self.eval_func(y_test, new_model.predict(X_test))
            self.hyperparam_history.append((score, new_hyperparams))
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])

        best_params = self.hyperparam_history[best_params_idx][1]        
        best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model


class AntColonyOptimizer(RandomSearchOptimizer):
    pass
