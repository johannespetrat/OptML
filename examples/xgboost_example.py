import numpy as np

import xgboost as xgb

from sklearn.datasets import make_classification
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split

from optml.random_search import RandomSearchOptimizer
from optml.bayesian_optimizer import BayesianOptimizer
from optml.hyperopt_optimizer import HyperoptOptimizer
from optml.genetic_optimizer import GeneticOptimizer
from optml import Parameter
from optml.models import KerasModel

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
model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, reg_lambda=1)


# define the list of hyperparameters to tune for each classifier
params = [Parameter(name='max_depth', param_type='integer', lower=1, upper=4),
          Parameter(name='learning_rate', param_type='continuous', lower=0.01, upper=0.5),
          Parameter(name='reg_lambda', param_type='continuous', lower=0.1, upper=10)]

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

hyperOpt = HyperoptOptimizer(model=model, 
                             hyperparams=params,                              
                             eval_func=clf_score)

n_init_samples = 4    
mutation_noise = {'max_depth': 0.4, 'learning_rate': 0.05, 
                  'reg_lambda':0.5}
geneticOpt = GeneticOptimizer(model, params, clf_score, n_init_samples, 
                             'RouletteWheel', mutation_noise)

# train and evaluate the models on the training data; alternatively we could score different sets
# of hyperparameters on validation data 
rand_best_params, rand_best_model = rand_search.fit(X_train=X_train, y_train=y_train, n_iters=50)
bayes_best_params, bayes_best_model = bayesOpt.fit(X_train=X_train, y_train=y_train, n_iters=50)
hyperopt_best_params, hyperopt_best_model = hyperOpt.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
genetic_best_params, genetic_best_model = geneticOpt.fit(X_train=X_train, y_train=y_train, n_iters=50, n_tries=20)

rand_best_model.fit(X_test, y_test)
print("Random Search on test data: {}".format(clf_score(y_test, rand_best_model.predict(X_test))))

bayes_best_model.fit(X_test, y_test)
print("Bayesian Optimisation on test data: {}".format(clf_score(y_test, bayes_best_model.predict(X_test))))

hyperopt_best_model.fit(X_test, y_test)
print("Hyperopt Algo on test data: {}".format(clf_score(y_test, hyperopt_best_model.predict(X_test))))

genetic_best_model.fit(X_test, y_test)
print("Genetic Algo on test data: {}".format(clf_score(y_test, genetic_best_model.predict(X_test))))
