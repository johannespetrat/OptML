import numpy as np
import unittest
from optml.random_search import RandomSearchOptimizer
from optml import Parameter
from  sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))

class TestRandomSearchOptimizer(unittest.TestCase):
    def test_improvement(self):
        np.random.seed(4)
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
        rand_search = RandomSearchOptimizer(model, [p1], clf_score)
        best_params, best_model = rand_search.fit(X_train=data, y_train=target, n_iters=10)
        best_model.fit(data, target)
        final_score = clf_score(target, best_model.predict(data))
        self.assertTrue(final_score>start_score)
