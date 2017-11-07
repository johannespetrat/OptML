import numpy as np
import warnings

import sklearn.gaussian_process as gp
from scipy.optimize import minimize
from scipy.stats import norm

from optml.optimizer_base import Optimizer, MissingValueException


class BayesianOptimizer(Optimizer):
    """
    Bayesian Optimizer as described here: https://arxiv.org/pdf/1206.2944.pdf
    """
    def __init__(self, model, hyperparams, eval_func, method='expected_improvement',
                 kernel=gp.kernels.Matern(), n_restarts_optimizer=10):
        super(BayesianOptimizer, self).__init__(model, hyperparams, eval_func)
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.eval_func = eval_func
        self.bounds_arr = np.array([[hp.lower, hp.upper] for hp in self.hyperparams])
        self.success = None
        self.method = method

    def upper_confidence_bound(self, optimiser, x):
        mu,std = optimiser.predict([x], return_std=True)
        return -1 * (mu+1.96*std)[0]

    def expected_improvement(self, optimiser, x):
        mu,std = optimiser.predict([x], return_std=True)
        current_best = max([score for score, params in self.hyperparam_history])
        gamma = (current_best - mu[0])/std[0]
        exp_improv = std[0] * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return -1 * exp_improv

    def probability_of_improvement(self, optimiser, x):
        mu,std = optimiser.predict([x], return_std=True)
        current_best = max([score for score, params in self.hyperparam_history])
        gamma = (current_best - mu[0])/std[0]
        return -1 * norm.cdf(gamma)

    def get_next_hyperparameters(self, optimiser):
        best_params = {}
        for i in range(self.n_restarts_optimizer):
            start_vals = np.random.uniform(np.array(self.bounds_arr)[:,0], np.array(self.bounds_arr)[:,1])
            #minimized = minimize(lambda x: -1 * optimiser.predict(x), start_vals, bounds=, method='L-BFGS-B')
            if self.method == 'expected_improvement':
                minimized = minimize(lambda x: self.expected_improvement(optimiser, x), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')
            elif self.method == 'upper_confidence_bound':
                minimized = minimize(lambda x: self.upper_confidence_bound(optimiser, x), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')
            elif self.method == 'probability_of_improvement':
                minimized = minimize(lambda x: self.probability_of_improvement(optimiser, x), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')

            self.success = minimized['success']
            if minimized['success']:
                new_params = {}
                for hp,v in zip(self.hyperparams, minimized['x']):
                    if hp.param_type == 'integer':
                        new_params[hp.name] = int(round(v))
                    else:
                        new_params[hp.name] = v
                return new_params                
        else:
            self.success = False
            #assert False, 'Optimiser did not converge!'
            warnings.warn('Optimiser did not converge! Continuing with randomly sampled data...')
            self.non_convergence_count += 1
            return {hp.name:v for hp,v in zip(self.hyperparams, start_vals)}

    def _random_sample(self):
        sampled_params = {}
        for hp,v in zip(self.hyperparams, np.random.uniform(self.bounds_arr[:,0],self.bounds_arr[:,1])):
            if hp.param_type == 'integer':
                sampled_params[hp.name] = int(round(v))
            else:
                sampled_params[hp.name] = v
        return sampled_params

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_iters=10, start_vals=None):
        """
        """
        if (X_test is None) and (y_test is None):
            X_test = X_train
            y_test = y_train
        elif (X_test is None) or (y_test is None):
            raise MissingValueException("Need to provide 'X_test' and 'y_test'")

        self.non_convergence_count = 0
        optimiser = gp.GaussianProcessRegressor(kernel=self.kernel,
                                                alpha=1e-4,
                                                n_restarts_optimizer=self.n_restarts_optimizer,
                                                normalize_y=True)
        for i in range(n_iters):            
            if i>0:                
                xs = np.array([params.values() for score, params in self.hyperparam_history])
                ys = np.array([score for score, params in self.hyperparam_history])
                optimiser.fit(xs,ys) 
                new_hyperparams = self.get_next_hyperparameters(optimiser)
            else:
                new_hyperparams = self._random_sample()

            new_model = self.build_new_model(new_hyperparams)

            new_model.fit(X_train, y_train)
            score = self.eval_func(y_test, new_model.predict(X_test))
            self.hyperparam_history.append((score, new_hyperparams))
        
        best_params, best_model = self.get_best_params_and_model()
        return best_params, best_model
