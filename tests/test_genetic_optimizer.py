import numpy as np
import unittest
from optml.genetic_optimizer import GeneticOptimizer
from optml import Parameter
from  sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))

class TestGeneticOptimizer(unittest.TestCase):
    def test_bounds_arr(self):
        interval1 = [0,10]
        interval2 = [11,20]
        p1 = Parameter('test_integer1', 'integer', lower=interval1[0], upper=interval1[1])
        p2 = Parameter('test_integer2', 'integer', lower=interval2[0], upper=interval2[1])
        mutation_noise = {'test_integer1': 0.4, 'test_integer2': 0.05}
        model = RandomForestClassifier()
        geneticOpt = GeneticOptimizer(model, [p1,p2],lambda x: x,4, 'RouletteWheel',mutation_noise)

        self.assertTrue(geneticOpt.bounds['test_integer1'][0]>=interval1[0])
        self.assertTrue(geneticOpt.bounds['test_integer1'][1]<=interval1[1])
        self.assertTrue(geneticOpt.bounds['test_integer2'][0]>=interval2[0])
        self.assertTrue(geneticOpt.bounds['test_integer2'][1]>=interval2[1])

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
        n_init_samples = 4    
        mutation_noise = {'max_depth': 0.4, 'learning_rate': 0.05, 
                          'reg_lambda':0.5}
        geneticOpt = GeneticOptimizer(model, [p1], clf_score, n_init_samples, 
                                     'RouletteWheel', mutation_noise)

        best_params, best_model = geneticOpt.fit(X_train=data, y_train=target, n_iters=30)
        best_model.fit(data, target)
        final_score = clf_score(target, best_model.predict(data))
        self.assertTrue(final_score>start_score)
