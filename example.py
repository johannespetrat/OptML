import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

if __name__ == "__main__":
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

    n_init_samples = 4    
    mutation_noise = {'hidden_dim': 5}
    geneticOpt = GeneticOptimizer(nn_model, ['hidden_dim'], clf_score, nn_bounds, n_init_samples, 
                                  'RouletteWheel', mutation_noise)
    genetic_best_params, genetic_best_model = geneticOpt.fit(data, target, 50)

    bayes_best_model.fit(data, target)
    print(clf_score(target, bayes_best_model.predict(data)))


    plt.plot([v[0] for v in bayesOpt.hyperparam_history])
    plt.plot([v[0] for v in rand_search.hyperparam_history])
    plt.show()
