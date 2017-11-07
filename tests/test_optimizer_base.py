import unittest
from optml.optimizer_base import Optimizer, MissingValueException
from optml import Parameter
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from optml.models import KerasModel
from keras.models import Sequential
from keras.layers import Dense, Activation

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

class NotAModel(object):
    pass

class TestBaseOptimizer(unittest.TestCase):
    def test_model_detection(self):
        sklearn_model = LogisticRegression()
        pipeline_model = Pipeline([('log', sklearn_model)])
        xgb_model = XGBClassifier()
        nn_model = NNModel(100,10)
        sklearn_opt = Optimizer(sklearn_model,[], lambda x: x)
        pipeline_opt = Optimizer(pipeline_model,[], lambda x: x)
        xgb_opt = Optimizer(xgb_model,[], lambda x: x)
        nn_opt = Optimizer(nn_model,[], lambda x: x)

        self.assertEqual(sklearn_opt.model_module, 'sklearn')
        self.assertEqual(pipeline_opt.model_module, 'pipeline')
        self.assertEqual(xgb_opt.model_module, 'xgboost')
        self.assertEqual(nn_opt.model_module, 'keras')


    def test_build_new_model_sklearn(self):
        sklearn_model = LogisticRegression(C=1)
        sklearn_opt = Optimizer(sklearn_model,[], lambda x: x)
        new_model = sklearn_opt.build_new_model({'C':0.8})
        self.assertEqual(new_model.get_params()['C'], 0.8)

    def test_build_new_model_xgboost(self):
        xgb_model = XGBClassifier(max_depth=3)
        xgb_opt = Optimizer(xgb_model,[], lambda x: x)
        new_model = xgb_opt.build_new_model({'max_depth': 2})
        self.assertEqual(new_model.get_params()['max_depth'], 2)

    def test_build_new_model_keras(self):
        nn_model = NNModel(input_dim=100, hidden_dim=10)
        nn_opt = Optimizer(nn_model,[], lambda x: x)
        new_model = nn_opt.build_new_model({'input_dim': 90, 'hidden_dim': 9})
        self.assertEqual(new_model.get_params()['input_dim'], 90)
        self.assertEqual(new_model.get_params()['hidden_dim'], 9)

    def test_build_new_model_pipeline(self):
        sklearn_model = LogisticRegression(C=1)
        pipeline_model = Pipeline([('log', sklearn_model)])
        pipeline_opt = Optimizer(pipeline_model,[], lambda x: x)
        new_model = pipeline_opt.build_new_model({'log__C': 0.8})
        self.assertEqual(new_model.get_params()['log__C'], 0.8)

    def test_exception_unknown_model(self):
        with self.assertRaises(NotImplementedError):
            Optimizer(NotAModel, [], lambda x: x)

    @unittest.skip("not implemented for statsmodels yet")
    def test_build_new_model_statsmodels(self):
        pass

class TestParameter(unittest.TestCase):
    def test_categorical(self):
        vals = ['a','b','c']
        p = Parameter('test_categorical', 'categorical', possible_values=vals)
        self.assertIn(p.random_sample(), vals)
        with self.assertRaises(MissingValueException):
            Parameter('test_categorical', 'categorical')

    def test_continuous(self):
        interval = [0,10]
        p = Parameter('test_continuous', 'continuous', lower=interval[0], upper=interval[1])
        s = p.random_sample()
        self.assertTrue(s>=interval[0])
        self.assertTrue(s<=interval[1])
        with self.assertRaises(MissingValueException):
            Parameter('test_continuous', 'continuous')  

    def test_integer(self):
        interval = [0,10]
        p = Parameter('test_integer', 'integer', lower=interval[0], upper=interval[1])
        s = p.random_sample()
        self.assertTrue(s in range(*interval))
        self.assertTrue(isinstance(s, int))
        with self.assertRaises(MissingValueException):
            Parameter('test_integer', 'integer')

    def test_boolean(self):
        p = Parameter('test_bool', 'bool')
        s = p.random_sample()
        self.assertTrue(isinstance(s, bool))

    def test_int_array(self):
        lower = [0,10,20]
        upper = [5,15,25]
        p = Parameter('test_int_array', 'int_array', lower=lower, upper=upper)
        s = p.random_sample()
        for i,v in enumerate(s):
            self.assertTrue(v in range(lower[i],upper[i]))

        with self.assertRaises(ValueError):
            Parameter('test_int_array', 'int_array',lower=[1,2],upper=[3,4,5])

        with self.assertRaises(MissingValueException):
            Parameter('test_int_array', 'int_array')

    def test_continuous_array(self):
        lower = [0,10,20]
        upper = [5,15,25]
        p = Parameter('test_continuous_array', 'continuous_array', lower=lower, upper=upper)
        s = p.random_sample()

        for i,v in enumerate(s):
            self.assertTrue(v>=lower[i])
            self.assertTrue(v<=upper[i])

        with self.assertRaises(ValueError):
            Parameter('test_continuous_array', 'continuous_array',lower=[1,2],upper=[3,4,5])

        with self.assertRaises(MissingValueException):
            Parameter('test_continuous_array', 'continuous_array')
