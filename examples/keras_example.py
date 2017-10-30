import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation

from optml.random_search import RandomSearchOptimizer
from optml.bayesian_optimizer import BayesianOptimizer
from optml.genetic_optimizer import GeneticOptimizer
from optml.optimizer_base import Parameter
from optml.models import KerasModel

class NNModel(KerasModel):
    def __init__(self, input_dim, hidden_dim, train_epochs=100, batch_size=32): 
        self.epochs = train_epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = Sequential()
        self.model.add(Dense(units=int(hidden_dim), input_dim=input_dim))
        self.model.add(Activation('relu'))
        self.model.add(Dense(units=1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    def get_params(self, deep=False):
        return {'batch_size': self.batch_size, 
                'hidden_dim': self.hidden_dim, 
                'input_dim': self.input_dim, 
                'train_epochs': self.epochs}    


# build some artificial data for classification
data, target = make_classification(n_samples=100,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5,
                                   class_sep=1,
                                   n_clusters_per_class=4,
                                   flip_y=0.4)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# define three different classifiers
model = NNModel(input_dim=data.shape[1], hidden_dim=10, train_epochs=100, batch_size=32)

params = [Parameter(name='hidden_dim', param_type='integer', lower=10, upper=200)]

# define the score function
def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))

rand_search = RandomSearchOptimizer(model=model,
                                    eval_func=clf_score,
                                    hyperparams=params,
                                    grid_size=10)

kernel = gp.kernels.Matern()        
bayesOpt = BayesianOptimizer(model=model, 
                             hyperparams=params, 
                             kernel=kernel,                                  
                             eval_func=clf_score)
n_init_samples = 4    
mutation_noise = {'C': 0.4, 'degree': 0.4, 
                  'min_samples_split':0, 'min_weight_fraction_leaf':0,
                  'hidden_dim': 1}
geneticOpt = GeneticOptimizer(model, params, clf_score, n_init_samples, 
                             'RouletteWheel', mutation_noise)

# train and evaluate the models on the training data; alternatively we could score different sets
# of hyperparameters on validation data 
rand_best_params, rand_best_model = rand_search.fit(X_train=X_train, y_train=y_train, n_iters=50)
bayes_best_params, bayes_best_model = bayesOpt.fit(X_train=X_train, y_train=y_train, n_iters=50)
genetic_best_params, genetic_best_model = geneticOpt.fit(X_train=X_train, y_train=y_train, n_iters=50, n_tries=20)

rand_best_model.fit(X_test, y_test)
print("Random Search on test data: {}".format(clf_score(y_test, rand_best_model.predict(X_test))))

bayes_best_model.fit(X_test, y_test)
print("Bayesian Optimisation on test data: {}".format(clf_score(y_test, bayes_best_model.predict(X_test))))

genetic_best_model.fit(X_test, y_test)
print("Genetic Algo on test data: {}".format(clf_score(y_test, genetic_best_model.predict(X_test))))
