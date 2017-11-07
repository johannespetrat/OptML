import numpy as np
import unittest
from optml.hyperopt_optimizer import HyperoptOptimizer
from optml import Parameter
from sklearn.linear_model import LogisticRegression
from  sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))


class TestHyperoptOptimizer(unittest.TestCase):
    def test_param_space(self):
        interval = [0,10]
        p1 = Parameter('test_integer', 'integer', lower=interval[0], upper=interval[1])
        p2 = Parameter('test_categorical', 'categorical', possible_values=['A','B','C'])
        p3 = Parameter('test_boolean', 'boolean')
        p4 = Parameter('test_continuous', 'continuous', lower=interval[0], upper=interval[1])
        p5 = Parameter('test_continuous_array', 'continuous_array', lower=[interval[0]], upper=[interval[1]])
        model = RandomForestClassifier()
        hyperopt = HyperoptOptimizer(model, [p1,p2,p3,p4],lambda x: x)
        param_space = hyperopt.param_space

        with self.assertRaises(ValueError):
        	hyperopt = HyperoptOptimizer(model, [p1,p2,p3,p4,p5],lambda x: x)        

    def test_improvement(self):
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
        hyperopt = HyperoptOptimizer(model, [p1], clf_score)
        best_params, best_model = hyperopt.fit(X_train=data, y_train=target, n_iters=10)
        best_model.fit(data, target)
        final_score = clf_score(target, best_model.predict(data))
        self.assertTrue(final_score>start_score)

        for status in hyperopt.trials.statuses():
        	self.assertEqual(status, 'ok')