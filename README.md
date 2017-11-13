# OptML     [![codecov](https://codecov.io/gh/johannespetrat/OptML/branch/master/graph/badge.svg)](https://codecov.io/gh/johannespetrat/OptML)
This package offers implementations of several black-box optimisation methods to tune hyperparameters of machine learning models. Its purpose is to enable data scientists to use optimization techniques for rapid protyping. Simply import OptML and supply it with a model and the parameters to optimize.

OptML offers a unified interface for models built with Scikit-Learn, Keras, XGBoost (and hopefully soon Statsmodels).

## Prerequisites
This package requires scikit-learn with version 0.19.0 or higher. If scikit-learn is not yet install run
```pip install scikit-learn==0.19.0```. If you want to make use of the ```HyperoptOptimizer``` then you also need to install [hyperopt](https://github.com/hyperopt/hyperopt) (e.g. by ```pip install hyperopt```).

In order to run with Keras and XGBoost models these libraries have to be install as well, of course.

## Installation

If scikit-learn is version 0.19 or higher simply install mlopt using `pip install optml` and you're ready to go.


## Usage
Specify your ML model and the parameters you want to optimize over. For the parameters you have to choose the type (such as integer, categorical, boolean, etc.) and the range of values it can take.
```python
model = SomeMLModel()
params = [Parameter(name='param1', param_type='continuous', lower=0.1, upper=5),
          Parameter(name='param2', param_type='integer', lower=1, upper=5),
          Parameter(name='param3', param_type='categorical', possible_values=['val1','val2','val3'])]
```
Then define the evaluation function. This can be anything from RMSE to crossentropy to custom functions. The first argument of the evaluation function is the array of true labels and the second argument is an array with model predictions.
```python
def clf_score(y_true,y_pred):
    return np.sum(y_true==y_pred)/float(len(y_true))
```
Import and initialize an optimizer and optimize the model for some training data.
```python
from optml.bayesian_optimizer import BayesianOptimizer
bayesOpt = BayesianOptimizer(model=model, 
                             hyperparams=params,                                  
                             eval_func=clf_score)
bayes_best_params, bayes_best_model = bayesOpt.fit(X_train=X_train, y_train=y_train, n_iters=50)
```

## Features
At the moment this library includes:
* Random Search
* Parallelized Gridsearch
* A simple Genetic Algorithm
* Bayesian Optimisation
* Hyperopt (using [hyperopt](https://github.com/hyperopt/hyperopt))


## TODOs
1. algorithms:
* more options for genetic algorithms
* meta heuristics/swarm optimisation (Ant Colony Optimization etc.)
2. functionality
* early stopping if there is no significant improvement after x iterations
3. usability
* add categorical parameters
* better documenation

## Author

* **Johannes Petrat** - *Initial work* - [johannespetrat](https://github.com/johannespetrat)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details