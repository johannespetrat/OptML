import numpy as np
from optml.optimizer_base import Optimizer, MissingValueException
from models import Model
#from model_converter import ModelConverter


class RandomSearchOptimizer(Optimizer):
    def __init__(self, model, hyperparams, eval_func):
        super(RandomSearchOptimizer, self).__init__(model, hyperparams, eval_func)

    def get_next_hyperparameters(self):
        new_hyperparams = {}
        for hp in self.hyperparams:            
            new_hyperparams[hp.name] = hp.random_sample()                    
        return new_hyperparams

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_iters=10):
        # get the hyperparameters of the base model
        if (X_test is None) and (y_test is None):
            X_test = X_train
            y_test = y_train
        elif (X_test is None) or (y_test is None):
            raise MissingValueException("Need to provide 'X_test' and 'y_test'")

        hyperparams = self.model.get_params()
        # and update them with the new hyperparameters
        for i in range(n_iters):
            new_hyperparams = self.get_next_hyperparameters()
            hyperparams.update(new_hyperparams)
            if self.model_module == 'statsmodels':                
                hyperparams.update({'endog': X_train})

            new_model = self.build_new_model(hyperparams)
            fitted_model = new_model.fit(X_train, y_train)                        
            score = self.eval_func(y_test, fitted_model.predict(X_train))
            self.hyperparam_history.append((score, new_hyperparams))
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])

        #best_params = self.hyperparam_history[best_params_idx][1]        
        #best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        best_params, best_model = self.get_best_params_and_model()
        return best_params, best_model


class AntColonyOptimizer(RandomSearchOptimizer):
    pass
