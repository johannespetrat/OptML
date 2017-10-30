import numpy as np
from mlopt.optimizer_base import Optimizer, MissingValueException
from models import Model
#from model_converter import ModelConverter


class RandomSearchOptimizer(Optimizer):
    def __init__(self, model, eval_func, hyperparams, grid_size):
        #self.model = ModelConverter(model).convert()
        super(RandomSearchOptimizer, self).__init__(model, hyperparams, eval_func)
        self.hyperparams_grid = self.build_param_grid(grid_size) 
        self.model_module = model.__module__.split('.')[0]

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
                    param_range = np.choice(h.possible_values)
                except KeyError:
                    raise MissingValueException("Need to provide possible values for the parameter '{}'".format(h.name))
            elif h.param_type == 'boolean':
                param_range = [True, False]
            elif h.param_type == 'continuous_array':
                param_range = np.array([np.linspace(h.lower[idx], h.upper[idx], grid_size) for idx in range(len(h.lower))])
            elif h.param_type == 'int_array':
                param_range = np.array([np.arange(h.lower[idx], h.upper[idx], 1) for idx in range(len(h.lower))])
            else:
                raise ValueError("param_type needs to be 'categorical','continuous','integer', 'int_array', 'continuous_array' or 'boolean'")
            grid[h.name] = param_range
        return grid

    def get_next_hyperparameters(self):
        new_hyperparams = {}
        #for key, val_range in self.hyperparams_grid.items():
        for hp in self.hyperparams:            
            new_hyperparams[hp.name] = hp.random_sample()                    
        return new_hyperparams

    def build_new_model(self, new_hyperparams):
        if self.model_module == 'sklearn':
            new_model = self.model.__class__(**new_hyperparams)
        elif self.model_module == 'statsmodels':
            new_model = self.model.__class__(**new_hyperparams)
            #new_model = ModelConverter(new_model).convert()
        elif isinstance(self.model, Model):
            new_model = self.model.__class__(**new_hyperparams)
        else:
            raise NotImplementedError("RandomSearchOptimizer not implemented for module '{}'".format(
                    self.model_module))
        return new_model

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
            if self.model_module == 'sklearn': 
                new_model = self.build_new_model(hyperparams)
                new_model.fit(X_train, y_train)
                score = self.eval_func(y_test, new_model.predict(X_test))
            elif self.model_module == 'statsmodels':                
                hyperparams.update({'endog': X_train})
                new_model = self.build_new_model(hyperparams)
                fitted_model = new_model.fit(X_train)                        
                score = self.eval_func(y_test, fitted_model.predict())
            elif isinstance(self.model, Model):
                new_model = self.build_new_model(hyperparams)
                new_model.fit(X_train, y_train)
                score = self.eval_func(y_test, new_model.predict(X_test))
            else:
                raise NotImplementedError("RandomSearchOptimizer not implemented for module '{}'".format(
                    self.model_module))
            self.hyperparam_history.append((score, new_hyperparams))
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])

        best_params = self.hyperparam_history[best_params_idx][1]        
        best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model


class AntColonyOptimizer(RandomSearchOptimizer):
    pass
