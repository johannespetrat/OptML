import numpy as np
import warnings

import sklearn.gaussian_process as gp
from scipy.optimize import minimize
from scipy.stats import norm

from optml.optimizer_base import Optimizer, MissingValueException
from optml.bayesian_optimizer.kernels import HammingKernel, WeightedHammingKernel
from sklearn.gaussian_process.kernels import Matern

from optml.bayesian_optimizer.gp_categorical import GaussianProcessRegressorWithCategorical

class BayesianOptimizer(Optimizer):
    """ Bayesian Optimizer
    Implemented as described in the paper 'Practical Bayesian Optimization of Machine 
    Learning Algorithms' (https://arxiv.org/abs/1206.2944)

    For categorical parameters the optimizer used a WeightedHammingKernel as described in
    'Sequential Model-Based Optimization for General Algorithm Configuration' by 
    Frank Hutter, Holger H. Hoos, Kevin Leyton-Brown doi:10.1007/978-3-642-25566-3_40

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
        kernel
        n_restart_optimizer
        eval_func
        bounds_arr
        success
        method
    """
    def __init__(self, model, hyperparams, eval_func, method='expected_improvement',
                 n_restarts_optimizer=10, exploration_control=0.01):
        super(BayesianOptimizer, self).__init__(model, hyperparams, eval_func)
        self.get_type_of_optimization()
        self.kernel = self.choose_kernel()
        self.n_restarts_optimizer = n_restarts_optimizer
        self.eval_func = eval_func
        self.set_hyperparam_bounds()
        self.success = None
        self.method = method
        if method == 'generalized_expected_improvement':
            self.exploration_control = exploration_control

    def choose_kernel(self):
        n_categorical = np.sum([hp.param_type=='categorical' for hp in self.hyperparams])
        if self.optimization_type == 'categorical':
            kernel = HammingKernel()
        elif self.optimization_type == 'mixed':
            param_types = ['categorical' if hp.param_type=='categorical' else 'numeric' for hp in self.hyperparams]
            kernel = WeightedHammingKernel()
        else:
            kernel = Matern()
        return kernel

    def get_type_of_optimization(self):
        n_categorical = np.sum([hp.param_type=='categorical' for hp in self.hyperparams])
        if n_categorical == len(self.hyperparams):
            self.optimization_type = 'categorical'
        elif n_categorical>=0:
            self.optimization_type = 'mixed'
        else:
            self.optimization_type = 'numerical'

    def set_hyperparam_bounds(self):
        self.bounds_arr = np.array([[hp.lower, hp.upper] for hp in self.hyperparams if hp.param_type!='categorical'])
        #self.bounds_arr = self.add_bounds_for_categorical(bounds_arr)

    def add_bounds_for_categorical(self, bounds_arr):
        for param in self.hyperparams:
            if param.param_type == 'categorical':
                lower = np.zeros(len(param.possible_values))
                upper = np.ones(len(param.possible_values))
                bounds_arr = np.concatenate([bounds_arr, np.vstack([lower, upper]).T])
        return bounds_arr

    def upper_confidence_bound(self, optimizer, x):
        mu,std = optimizer.predict(np.atleast_2d(x), return_std=True)
        return -1 * (mu+1.96*std)[0]

    def expected_improvement(self, optimizer, x):
        mu, std = optimizer.predict(np.atleast_2d(x), return_std=True)
        current_best = max([score for score, params in self.hyperparam_history])
        gamma = (mu[0] - current_best)/std[0]
        exp_improv = std[0] * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return -1 * exp_improv

    def generalized_expected_improvement(self, optimizer, x, xi=0.01):
        """
        Args:
            xi: controls the trade-off between exploration and exploitation
        as in https://arxiv.org/pdf/1012.2599 page 14
        """
        mu,std = optimizer.predict(np.atleast_2d(x), return_std=True)
        if std == 0:
            return 0
        else:
            current_best = max([score for score, params in self.hyperparam_history])
            gamma = (mu[0] - current_best - xi)/std[0]
            exp_improv = (mu[0] - current_best - xi) * norm.cdf(gamma) + std[0] * norm.pdf(gamma)
            return -1 * exp_improv

    def probability_of_improvement(self, optimizer, x):
        mu,std = optimizer.predict(np.atleast_2d(x), return_std=True)
        current_best = max([score for score, params in self.hyperparam_history])
        gamma = (mu[0] - current_best)/std[0]
        return -1 * norm.cdf(gamma)

    def get_start_values_arr(self):
        start_vals = []
        for hp in self.hyperparams:
            start_vals.append(hp.random_sample())
        return np.array([start_vals], dtype=object)

    def get_start_values_dict(self):
        start_vals = {hp.name:hp.random_sample() for hp in self.hyperparams}
        return start_vals

    def optimize_continuous_problem(self, optimizer, start_vals):
        if self.method == 'expected_improvement':
            minimized = minimize(lambda x: self.expected_improvement(optimizer, x), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')
        elif self.method == 'upper_confidence_bound':
            minimized = minimize(lambda x: self.upper_confidence_bound(optimizer, x), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')
        elif self.method == 'probability_of_improvement':
            minimized = minimize(lambda x: self.probability_of_improvement(optimizer, x), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')
        elif self.method == 'generalized_expected_improvement':
            minimized = minimize(lambda x: self.generalized_expected_improvement(optimizer, x, self.exploration_control), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')
        return minimized

    def optimize_categorical_problem(self, optimizer, start_vals):
        xs = np.array([list(params.values()) for score, params in self.hyperparam_history])

    def optimize_mixed_problem(self, optimizer, start_vals):
        import pdb; pdb.set_trace()

    def get_next_hyperparameters(self, optimizer):
        best_params = {}
        for i in range(self.n_restarts_optimizer):
            start_vals = self.get_start_values_arr()
            if self.optimization_type == 'numerical':
                minimized = self.optimize_continuous_problem(optimizer, start_vals)
            elif self.optimization_type == 'categorical':
                minimized = self.optimize_categorical_problem(optimizer, start_vals)
            else:
                minimized = self.optimize_mixed_problem(optimizer, start_vals)
            

            self.success = minimized['success']
            if minimized['success']:
                new_params = {}
                for hp,v in zip(self.hyperparams, minimized['x']):
                    if hp.param_type == 'integer':
                        new_params[hp.name] = int(round(v))
                    elif hp.param_type == 'categorical':
                        new_params[hp.name] = str(v)
                    else:
                        new_params[hp.name] = v
                return new_params                
        else:
            self.success = False
            #assert False, 'optimizer did not converge!'
            warnings.warn('optimizer did not converge! Continuing with randomly sampled data...')
            self.non_convergence_count += 1
            return {hp.name:v for hp,v in zip(self.hyperparams, start_vals)}

    def _random_sample(self):
        sampled_params = {}
        for hp in self.hyperparams:
            sampled_params[hp.name] = hp.random_sample()
        return sampled_params

    def _get_ordered_param_dict(self, params):
        return [params[hp.name] for hp in self.hyperparams]

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_iters=10, start_vals=None):
        """
        """
        if (X_test is None) and (y_test is None):
            X_test = X_train
            y_test = y_train
        elif (X_test is None) or (y_test is None):
            raise MissingValueException("Need to provide 'X_test' and 'y_test'")

        self.non_convergence_count = 0
        optimizer = GaussianProcessRegressorWithCategorical(kernel=self.kernel,
                                                alpha=1e-4,
                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                normalize_y=True)
        for i in range(n_iters):            
            if i>0:           
                xs = [self._get_ordered_param_dict(params) for score, params in self.hyperparam_history]
                xs = np.array(xs, dtype=object)
                ys = np.array([score for score, params in self.hyperparam_history])
                optimizer.fit(xs,ys)
                new_hyperparams = self.get_next_hyperparameters(optimizer)
            else:
                new_hyperparams = self._random_sample()

            new_model = self.build_new_model(new_hyperparams)
            new_model.fit(X_train, y_train)
            score = self.eval_func(y_test, new_model.predict(X_test))
            self.hyperparam_history.append((score, new_hyperparams))
        
        best_params, best_model = self.get_best_params_and_model()
        return best_params, best_model
