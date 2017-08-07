import numpy as np
from optimizer_base import Optimizer

class RandomSearchOptimizer(Optimizer):
    def __init__(self, model, eval_func, hyperparams, hyperparams_grid):        
        self.model = model
        self.eval_func = eval_func
        self.hyperparam_history = []
        self.hyperparams = hyperparams
        self.hyperparams_grid = hyperparams_grid

    def get_next_hyperparameters(self):
        new_hyperparams = {}
        for key, val_range in self.hyperparams_grid.items():
            new_hyperparams[key] = np.random.choice(val_range)
        return new_hyperparams

    def fit(self, X, y, n_iters):
        # get the hyperparameters of the base model
        hyperparams = self.model.get_params()
        # and update them with the new hyperparameters
        for i in range(n_iters):
            new_hyperparams = self.get_next_hyperparameters()
            hyperparams.update(new_hyperparams)
            new_model = self.model.__class__(**hyperparams)

            new_model.fit(X,y)
            score = self.eval_func(y, new_model.predict(X))
            self.hyperparam_history.append((score, new_hyperparams))
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])

        best_params = self.hyperparam_history[best_params_idx][1]        
        best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model


class AntColonyOptimizer(RandomSearchOptimizer):
    pass
