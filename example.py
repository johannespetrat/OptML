import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation

from random_search import RandomSearchOptimizer
from bayesian_optimizer import BayesianOptimizer
from genetic_optimizer import GeneticOptimizer


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

class Parameter(object):
    def __init__(self, name, param_type, lower, upper, distribution=None):
        # continuous, categorical, binary, integer
        param_type = param_type.lower()
        if not param_type in ['categorical', 'continuous', 'integer', 'boolean']:
            raise ValueError("param_type needs to be 'categorical','continuous','integer' or 'boolean'")
        self.param_type = param_type.lower()
        self.lower = lower
        self.upper = upper
        self.name = name
        if distribution is not None:
            self.distribution = distribution     


if __name__ == "__main__":
    data, target = make_classification(n_samples=2000,
                                       n_features=45,
                                       n_informative=15,
                                       n_redundant=5,
                                       class_sep=0.5)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

    rf = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_split=4)
    svm = SVC(C=1, kernel='rbf', degree=3)
    nn_model = NNModel(input_dim=data.shape[1], hidden_dim=10, train_epochs=100, batch_size=32)

    rf_params = [Parameter(name='min_samples_split', param_type='integer', lower=2, upper=6),
                 Parameter(name='min_weight_fraction_leaf', param_type='continuous', lower=0, upper=0.5)]
    svm_params = [Parameter(name='C', param_type='continuous', lower=0.1, upper=5),
                  Parameter(name='degree', param_type='integer', lower=1, upper=5)]
    nn_params = [Parameter(name='hidden_dim', param_type='integer', lower=10, upper=200)]


    rf_hyperparams_grid = {'min_samples_split':[2,3,4,5,6], 'min_weight_fraction_leaf':[0,0.1,0.2,0.3,0.4,0.5]}
    svm_hyperparams_grid = {'C':[0.1,0.2,0.4,0.8,1,2,3,5], 'degree':[1,2,3,4,5]}
    nn_hyperparams_grid = {'hidden_dim':[10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]}

    def clf_score(y_true,y_pred):
        return np.sum(y_true==y_pred)/float(len(y_true))

    rand_search = RandomSearchOptimizer(model=svm,
                                        eval_func=clf_score,
                                        hyperparams=svm_params,
                                        grid_size=10)

    kernel = gp.kernels.Matern()
    n_restarts_optimizer = 10
    rf_bounds = {'min_samples_split':[2,6],'min_weight_fraction_leaf':[0,0.5]}
    svm_bounds = {'C':[0.1,5],'degree':[1,5]}
    nn_bounds = {'hidden_dim':[2,200]}

    bayesOpt = BayesianOptimizer(model=svm, 
                                 hyperparams=svm_params, 
                                 kernel=kernel, 
                                 n_restarts_optimizer=n_restarts_optimizer, 
                                 eval_func=clf_score)
    bayes_best_params, bayes_best_model = bayesOpt.fit(X_train, y_train, X_test, y_test, 10)


    n_init_samples = 4    
    #mutation_noise = {'hidden_dim': 5}
    mutation_noise1 = {'C': 0.4, 'degree': 0.4}
    mutation_noise2 = {'C': 2, 'degree': 2}
    geneticOpt1 = GeneticOptimizer(svm, ['C','degree'], clf_score, svm_bounds, n_init_samples, 
                                  'RouletteWheel', mutation_noise1)
    geneticOpt2 = GeneticOptimizer(svm, ['C','degree'], clf_score, svm_bounds, n_init_samples, 
                                  'RouletteWheel', mutation_noise2)

    rand_best_params, rand_best_model = rand_search.fit(X_train, y_train, X_test, y_test, 50)
    bayes_best_params, bayes_best_model = bayesOpt.fit(X_train, y_train, X_test, y_test, 10)
    genetic_best_params1, genetic_best_model1 = geneticOpt1.fit(X_train, y_train, X_test, y_test, 50)
    genetic_best_params2, genetic_best_model2 = geneticOpt2.fit(X_train, y_train, X_test, y_test, 50)

    rand_best_model.fit(data, target)
    print("Random Search: {}".format(clf_score(y_test, rand_best_model.predict(X_test))))

    bayes_best_model.fit(data, target)
    print("Bayesian Optimisation: {}".format(clf_score(y_test, bayes_best_model.predict(X_test))))

    genetic_best_model1.fit(data, target)
    genetic_best_model2.fit(data, target)
    print("Genetic Algo 1: {}".format(clf_score(y_test, genetic_best_model1.predict(X_test))))
    print("Genetic Algo 2: {}".format(clf_score(y_test, genetic_best_model2.predict(X_test))))

    plt.plot([v[0] for v in bayesOpt.hyperparam_history], label='BayesOpt')
    plt.plot([v[0] for v in rand_search.hyperparam_history], label='Random Search')
    plt.plot([v[0] for v in geneticOpt1.hyperparam_history], label='Genetic Algo 1')
    plt.plot([v[0] for v in geneticOpt2.hyperparam_history], label='Genetic Algo 2')
    plt.legend()
    plt.show()
