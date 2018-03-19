import numpy as np
from optml.optimizer_base import Optimizer, MissingValueException
from .models import Model

class RandomSearchOptimizer(Optimizer):
    def __init__(self, model, hyperparams, eval_func, start_vals=None):
        super(RandomSearchOptimizer, self).__init__(model, hyperparams, eval_func, start_vals)

    def get_next_hyperparameters(self):
        new_hyperparams = {}
        for hp in self.hyperparams:            
            new_hyperparams[hp.name] = hp.random_sample()                    
        return new_hyperparams

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_iters=10, n_folds=None):
        # get the hyperparameters of the base model
        if (X_test is None) and (y_test is None):
            X_test = X_train
            y_test = y_train
        elif (X_test is None) or (y_test is None):
            raise MissingValueException("Need to provide 'X_test' and 'y_test'")
        elif (X_test is not None) and (y_test is not None) and (n_folds is not None):
            raise Exception("Provide either 'X_test' and 'y_test' or 'n_folds'")

        #hyperparams = self.model.get_params()
        # and update them with the new hyperparameters
        for i in range(n_iters):
            new_hyperparams = self.get_next_hyperparameters()
            #hyperparams.update(new_hyperparams)
            score = self._fit_and_score_model(new_hyperparams, X_train, y_train, X_test, y_test, 
                                              n_folds)
            self.hyperparam_history.append((score, new_hyperparams))
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])
        best_params, best_model = self.get_best_params_and_model()
        return best_params, best_model


class AntColonyOptimizer(RandomSearchOptimizer):
    pass
