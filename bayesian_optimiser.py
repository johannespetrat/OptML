# todo list
# 1. algorithms:
# add genetic algorithms
# add grid search
# add meta heuristics (ant colony etc)
# 2. functionality
# add early stopping; no improvement after x iterations
# parallelization??
# add optional cross validation 
# automatic detection if Keras, Scikit-learn, XGBoost or statsmodel

import abc
import numpy as np
import matplotlib.pyplot as plt
import warnings

import sklearn.gaussian_process as gp
from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.base import BaseEstimator
from keras.layers import Dense, Activation

from random_search import *

class Optimizer(object):

    def __init__(self, model, hyperparams, eval_func):        
        self.model = model
        self.hyperparam_history = []
        self.eval_func = eval_func

    @abc.abstractmethod
    def get_next_hyperparameters(self):
        raise NotImplementedError("This class needs a get_next_hyperparameters(...) function")

    @abc.abstractmethod
    def fit(self, X, y, params):
        raise NotImplementedError("This class needs a self.fit(X, y, params) function")

    def build_new_model(self, new_hyperparams):
        hyperparams = self.model.get_params()
        hyperparams.update(new_hyperparams)
        new_model = self.model.__class__(**hyperparams)
        return new_model

    def get_best_params_and_model(self):
        best_params_idx = np.argmax([score for score, params in self.hyperparam_history])
        best_params = self.hyperparam_history[best_params_idx][1]        
        best_model = self.model.__class__(**dict(self.model.get_params(), **best_params))
        return best_params, best_model


class BayesianOptimizer(Optimizer):
    """
    different objective functions taken from here https://arxiv.org/pdf/1206.2944.pdf
    """
    def __init__(self, model, hyperparams, kernel, n_restarts_optimizer, eval_func, bounds):
        self.model = model
        self.hyperparams = hyperparams
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.hyperparam_history = []
        self.eval_func = eval_func
        self.bounds_arr = np.array([bounds[hp]for hp in self.hyperparams])

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
        for i in range(n_restarts_optimizer):
            start_vals = np.random.uniform(np.array(self.bounds_arr)[:,0], np.array(self.bounds_arr)[:,1])
            #minimized = minimize(lambda x: -1 * optimiser.predict(x), start_vals, bounds=, method='L-BFGS-B')            
            minimized = minimize(lambda x: self.upper_confidence_bound(optimiser, x), start_vals, bounds=self.bounds_arr, method='L-BFGS-B')
            #from nose.tools import set_trace; set_trace()
            if minimized['success']:                
                return {k:v for k,v in zip(self.hyperparams, minimized['x'])}
        else:
            #assert False, 'Optimiser did not converge!'
            warnings.warn('Optimiser did not converge! Continuing with randomly sampled data...')
            self.non_convergence_count += 1
            return {k:v for k,v in zip(self.hyperparams, start_vals)}

    def fit(self, X, y, n_iters, start_vals):
        """
        
        Arguments:
        ----------
            n_iters: int
                Number of iterations

        """
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
                new_hyperparams = {k:v for k,v in zip(self.hyperparams, 
                                                      np.random.uniform(self.bounds_arr[:,0],self.bounds_arr[:,1]))}

            new_model = self.build_new_model(new_hyperparams)

            new_model.fit(X, y)
            score = self.eval_func(y, new_model.predict(X))
            self.hyperparam_history.append((score, new_hyperparams))
        
        best_params, best_model = self.get_best_params_and_model()
        return best_params, best_model


class GridSearchOptimizer(Optimizer):
    pass


class GeneticOptimizer(Optimizer):
    pass


def plot_surface(optimiser, bounds):
    grid = np.array(np.meshgrid(np.linspace(bounds[0][0],bounds[0][1], 100),
                                np.linspace(bounds[1][0],bounds[1][1],100)))
    grid = np.swapaxes(grid,0,2)
    orig_shape = grid.shape
    ys = optimiser.predict(grid.reshape(-1,2))
    ys = ys.reshape(orig_shape[:2])
    plt.imshow(ys)
    plt.show()


class NNModel(BaseEstimator):
    def __init__(self, input_dim, hidden_dim, train_epochs=100, batch_size=32): 
        self.epochs = train_epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nn_model = Sequential()
        self.nn_model.add(Dense(units=int(hidden_dim), input_dim=input_dim))
        self.nn_model.add(Activation('relu'))
        self.nn_model.add(Dense(units=1))
        self.nn_model.add(Activation('sigmoid'))
        self.nn_model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    def fit(self, X, y, verbose=0):
        return self.nn_model.fit(X,y, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)

    def predict(self, X):
        return self.nn_model.predict(X)

    def get_params(self, deep=False):
        return {'batch_size': self.batch_size, 
                'hidden_dim': self.hidden_dim, 
                'input_dim': self.input_dim, 
                'train_epochs': self.epochs}

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from keras.models import Sequential


    data, target = make_classification(n_samples=2000,
                                       n_features=45,
                                       n_informative=15,
                                       n_redundant=5,
                                       class_sep=0.8)

    rf = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_split=4)
    svm = SVC(C=1, kernel='rbf', degree=3)
    nn_model = NNModel(input_dim=data.shape[1], hidden_dim=10, train_epochs=100, batch_size=32)

    rf_hyperparams_grid = {'min_samples_split':[2,3,4,5,6], 'min_weight_fraction_leaf':[0,0.1,0.2,0.3,0.4,0.5]}
    svm_hyperparams_grid = {'C':[0.1,0.2,0.4,0.8,1,2,3,5], 'degree':[1,2,3,4,5]}
    nn_hyperparams_grid = {'hidden_dim':[10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]}

    def clf_score(y_true,y_pred):
        return np.sum(y_true==y_pred)/float(len(y_true))

    rand_search = RandomSearchOptimizer(nn_model,
                                        eval_func=clf_score,
                                        hyperparams=nn_hyperparams_grid.keys(), 
                                        hyperparams_grid=nn_hyperparams_grid)
    rand_best_params, rand_best_model = rand_search.fit(data, target, 100)

    rand_best_model.fit(data, target)
    print(clf_score(target, rand_best_model.predict(data)))

    kernel = gp.kernels.Matern()
    n_restarts_optimizer = 10
    rf_bounds = {'min_samples_split':[2,6],'min_weight_fraction_leaf':[0,0.5]}
    svm_bounds = {'C':[0.1,5],'degree':[1,5]}
    nn_bounds = {'hidden_dim':[2,200]}

    bayesOpt = BayesianOptimizer(nn_model, ['hidden_dim'], kernel, n_restarts_optimizer, clf_score, nn_bounds)
    bayes_best_params, bayes_best_model = bayesOpt.fit(data, target, 10, [10])

    bayes_best_model.fit(data, target)
    print(clf_score(target, bayes_best_model.predict(data)))


    plt.plot([v[0] for v in bayesOpt.hyperparam_history])
    plt.plot([v[0] for v in rand_search.hyperparam_history])
    plt.show()
