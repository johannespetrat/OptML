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

from hyperpy.random_search import RandomSearchOptimizer
from hyperpy.bayesian_optimizer import BayesianOptimizer
from hyperpy.genetic_optimizer import GeneticOptimizer
from hyperpy.optimizer_base import Parameter
from hyperpy.models import KerasModel

class NNModel(KerasModel):
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

    def clf_score(y_true,y_pred):
        return np.sum(y_true==y_pred)/float(len(y_true))

    rand_search = RandomSearchOptimizer(model=svm,
                                        eval_func=clf_score,
                                        hyperparams=svm_params,
                                        grid_size=10)

    kernel = gp.kernels.Matern()        
    bayesOpt = BayesianOptimizer(model=svm, 
                                 hyperparams=svm_params, 
                                 kernel=kernel,                                  
                                 eval_func=clf_score)
    


    n_init_samples = 4    
    #mutation_noise = {'hidden_dim': 5}
    mutation_noise = {'C': 0.4, 'degree': 0.4}    
    svm_bounds = {'C':[0.1,5],'degree':[1,5]}    

    geneticOpt = GeneticOptimizer(svm, svm_params, clf_score, n_init_samples, 
                                 'RouletteWheel', mutation_noise)
    genetic_best_params, genetic_best_model = geneticOpt.fit(X_train, y_train, X_test, y_test, 50)    

    rand_best_params, rand_best_model = rand_search.fit(X_train, y_train, X_test, y_test, 50)
    bayes_best_params, bayes_best_model = bayesOpt.fit(X_train, y_train, X_test, y_test, 10)
    genetic_best_params, genetic_best_model = geneticOpt.fit(X_train, y_train, X_test, y_test, 50)    

    rand_best_model.fit(data, target)
    print("Random Search: {}".format(clf_score(y_test, rand_best_model.predict(X_test))))

    bayes_best_model.fit(data, target)
    print("Bayesian Optimisation: {}".format(clf_score(y_test, bayes_best_model.predict(X_test))))

    genetic_best_model.fit(data, target)    
    print("Genetic Algo: {}".format(clf_score(y_test, genetic_best_model.predict(X_test))))

    plt.plot([v[0] for v in bayesOpt.hyperparam_history], label='BayesOpt')
    plt.plot([v[0] for v in rand_search.hyperparam_history], label='Random Search')
    plt.plot([v[0] for v in geneticOpt.hyperparam_history], label='Genetic Algo')    
    plt.legend()
    plt.show()

    [np.max(np.array(geneticOpt.hyperparam_history)[:i,0]) for i in range(1,len(geneticOpt.hyperparam_history))]