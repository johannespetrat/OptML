import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split

from optml.random_search import RandomSearchOptimizer
from optml.bayesian_optimizer import BayesianOptimizer
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
rf = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_split=4)
svm = SVC(C=1, kernel='rbf', degree=3)

# define the list of hyperparameters to tune for each classifier
rf_params = [Parameter(name='min_samples_split', param_type='integer', lower=2, upper=6),
             Parameter(name='min_weight_fraction_leaf', param_type='continuous', lower=0, upper=0.5)]
svm_params = [Parameter(name='C', param_type='continuous', lower=0.1, upper=5),
              Parameter(name='degree', param_type='integer', lower=1, upper=5),
              Parameter(name='kernel', param_type='categorical', 
                        possible_values=['linear', 'poly', 'rbf', 'sigmoid'])]

model = svm
params = svm_params
# define the score function
def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))

rand_search = RandomSearchOptimizer(model=model,
                                    eval_func=clf_score,
                                    hyperparams=params)
      
bayesOpt = BayesianOptimizer(model=model, 
                             hyperparams=params,
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
