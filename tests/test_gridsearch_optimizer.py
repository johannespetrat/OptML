import numpy as np
import unittest
from optml.gridsearch_optimizer import GridSearchOptimizer, objective
from optml import Parameter
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from functools import partial

def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))

class TestGridsearchOptimizer(unittest.TestCase):
    def grid_spacing(self):
        interval = [1,10]
        p1 = Parameter('A', 'integer', lower=interval[0], upper=interval[1])
        p2 = Parameter('B', 'continuous', lower=interval[0], upper=interval[1])
        p3 = Parameter('C', 'categorical', possible_values=['Bla1', 'Bla2'])
        p4 = Parameter('D', 'boolean')
        grid_sizes = {'A': 5, 'B': 6}
        grid_search = GridSearchOptimizer(model, [p1, p2, p3, p4], clf_score, grid_sizes)
        grid = grid_search.grid
        for params in grid:
            self.assertIn(params['A'], range(*interval))
            self.assertIn(params['B']>=interval[0])
            self.assertIn(params['B']<=interval[1])
            self.assertIn(params['C'], ['Bla1', 'Bla2'])
            self.assertIn(params['D'], ['True', 'False'])
        lenA = len(np.unique([params['A'] for params in grid]))
        lenB = len(np.unique([params['B'] for params in grid]))
        lenC = len(np.unique([params['C'] for params in grid]))
        lenD = len(np.unique([params['D'] for params in grid]))
        self.assertTrue((lenA==grid_sizes['A']) or (lenA==grid_sizes['A']+1))
        self.assertTrue((lenB==grid_sizes['B']) or (lenB==grid_sizes['B']+1))
        self.assertTrue((lenC==grid_sizes['C']) or (lenC==grid_sizes['C']+1))
        self.assertTrue((lenD==grid_sizes['D']) or (lenD==grid_sizes['D']+1))

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
        grid_sizes = {'max_depth': 5}
        grid_search = GridSearchOptimizer(model, [p1], clf_score, grid_sizes)
        best_params, best_model = grid_search.fit(X_train=data, y_train=target)
        best_model.fit(data, target)
        final_score = clf_score(target, best_model.predict(data))
        self.assertTrue(final_score>start_score)

    def test_objective_function(self):
        np.random.seed(4)
        data, target = make_classification(n_samples=100,
                                   n_features=10,
                                   n_informative=10,
                                   n_redundant=0,
                                   class_sep=100,
                                   n_clusters_per_class=1,
                                   flip_y=0.0)
        model = RandomForestClassifier(max_depth=5)
        model.fit(data, target)
        fun = partial(objective, model, 
                                 'sklearn', 
                                 clf_score,
                                 data, target, data, target)
        # model should fit the data perfectly
        final_score = fun(model.get_params())[0]
        self.assertEqual(final_score,1)
