import numpy as np
import unittest
from optml.bayesian_optimizer import BayesianOptimizer
from optml import Parameter
from sklearn.linear_model import LogisticRegression
from  sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))

class TestBayesianOptimizer(unittest.TestCase):
    def test_bounds_arr(self):
        interval1 = [0,10]
        interval2 = [11,20]
        p1 = Parameter('test_integer1', 'integer', lower=interval1[0], upper=interval1[1])
        p2 = Parameter('test_integer2', 'integer', lower=interval2[0], upper=interval2[1])
        model = RandomForestClassifier()
        bayesOpt = BayesianOptimizer(model, [p1,p2],lambda x: x)

        self.assertTrue(bayesOpt.bounds_arr[0][0]>=interval1[0])
        self.assertTrue(bayesOpt.bounds_arr[0][1]<=interval1[1])
        self.assertTrue(bayesOpt.bounds_arr[1][0]>=interval2[0])
        self.assertTrue(bayesOpt.bounds_arr[1][1]>=interval2[1])

    def test_expected_improvement_tractable(self):
        np.random.seed(5)
        data, target = make_classification(n_samples=100,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5,
                                   class_sep=1,
                                   n_clusters_per_class=4,
                                   flip_y=0.4)
        model = RandomForestClassifier(max_depth=5)
        model.fit(data, target)
        start_score = clf_score(target, model.predict(data))
        p1 = Parameter('max_depth', 'integer', lower=1, upper=10)
        bayesOpt = BayesianOptimizer(model, [p1], clf_score, method='expected_improvement')
        best_params, best_model = bayesOpt.fit(X_train=data, y_train=target, n_iters=10)
        self.assertTrue(bayesOpt.success)
        best_model.fit(data, target)
        final_score = clf_score(target, best_model.predict(data))
        self.assertTrue(final_score>start_score)


    def test_probability_of_improvement_tractable(self):
        np.random.seed(5)
        data, target = make_classification(n_samples=100,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5,
                                   class_sep=1,
                                   n_clusters_per_class=4,
                                   flip_y=0.4)
        model = RandomForestClassifier(max_depth=5)
        model.fit(data, target)
        start_score = clf_score(target, model.predict(data))
        p1 = Parameter('max_depth', 'integer', lower=1, upper=10)
        bayesOpt = BayesianOptimizer(model, [p1], clf_score, method='probability_of_improvement')
        best_params, best_model = bayesOpt.fit(X_train=data, y_train=target, n_iters=10)
        self.assertTrue(bayesOpt.success)
        best_model.fit(data, target)
        final_score = clf_score(target, best_model.predict(data))
        self.assertTrue(final_score>start_score)

    def test_upper_confidence_bound_tractable(self):
        np.random.seed(5)
        data, target = make_classification(n_samples=100,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5,
                                   class_sep=1,
                                   n_clusters_per_class=4,
                                   flip_y=0.4)
        model = RandomForestClassifier(max_depth=5)
        model.fit(data, target)
        start_score = clf_score(target, model.predict(data))
        p1 = Parameter('max_depth', 'integer', lower=1, upper=10)
        bayesOpt = BayesianOptimizer(model, [p1], clf_score, method='upper_confidence_bound')
        best_params, best_model = bayesOpt.fit(X_train=data, y_train=target, n_iters=10)
        self.assertTrue(bayesOpt.success)
        best_model.fit(data, target)
        final_score = clf_score(target, best_model.predict(data))
        self.assertTrue(final_score>start_score)