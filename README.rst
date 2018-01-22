OptML |codecov|
===============

This package offers implementations of several black-box optimisation methods to tune
hyperparameters of machine learning models. Its purpose is to enable data scientists to use
optimization techniques for rapid protyping. Simply import OptML and supply it with a model and the
parameters to optimize.

OptML offers a unified interface for models built with Scikit-Learn, Keras, XGBoost (and hopefully
soon Statsmodels).

Prerequisites
-------------

This package requires scikit-learn with version 0.19.0 or higher. If Scikit-Learn is not yet
installed run ``pip install scikit-learn==0.19.0``. If you want to make use of the
``HyperoptOptimizer`` then you also need to install
`hyperopt <https://github.com/hyperopt/hyperopt>`__ (e.g. by ``pip install hyperopt``).

In order to run with `Keras <https://github.com/fchollet/keras>`__ and
`XGBoost <https://github.com/dmlc/xgboost>`__ models these libraries have to be install as well, of
course.

Installation
------------

If Scikit-Learn is version 0.19 or higher simply install optml using ``pip install optml`` and
you're ready to go.

Usage
-----

Specify your ML model and the parameters you want to optimize over. For the parameters you have to
choose the type (such as integer, categorical, boolean, etc.) and the range of values it can take.

.. code:: python

    model = SomeMLModel()
    params = [Parameter(name='param1', param_type='continuous', lower=0.1, upper=5),
              Parameter(name='param2', param_type='integer', lower=1, upper=5),
              Parameter(name='param3', param_type='categorical', possible_values=['val1','val2','val3'])]

Then define the evaluation function. This can be anything from RMSE to crossentropy to custom
functions. The first argument of the evaluation function is the array of true labels and the second
argument is an array with model predictions.

.. code:: python

    def clf_score(y_true,y_pred):
        return np.sum(y_true==y_pred)/float(len(y_true))

Import and initialize an optimizer and optimize the model for some training data.

.. code:: python

    from optml.bayesian_optimizer import BayesianOptimizer
    bayesOpt = BayesianOptimizer(model=model, 
                                 hyperparams=params,                                  
                                 eval_func=clf_score)
    bayes_best_params, bayes_best_model = bayesOpt.fit(X_train=X_train, y_train=y_train, n_iters=50)

Features
--------

At the moment this library includes: \* Random Search \* Parallelized Gridsearch \* A simple Genetic
Algorithm \* Bayesian Optimisation (also supporting categorical parameters) \* Hyperopt (using
`hyperopt <https://github.com/hyperopt/hyperopt>`__)

How to Choose an Optimizer
--------------------------

OptML implements several optimization methods to address a range of requirements that can arise in
data science problems. One of the main concerns is the effort required to evaluate a model for a set
of parameters: If a model takes a long time to train we should choose an optimizer that maximises
the potential improvement with every new set of parameters. In this case Bayesian Optimization and
Hyperopt are more applicable. If a model is cheap to train then we can seek to parallelise the
evaluations.

Also consider the number of parameters and their ranges. Clearly, it is more difficult to optimize
over a large search space. It is advised to only include parameters in the optimization if they are
expected to improve the final model.

Please also note that all of OptML's optimizers require parameters to be bounded.

+--------------+------------------+--------------------+----------------+----------------------+-------------------+
|              | number of        | works with large   | can use        | handles categorical  | stochastic        |
|              | evaluations      | search space       | training in    | parameters           | optimisation      |
|              |                  |                    | parallel       |                      |                   |
+==============+==================+====================+================+======================+===================+
| Gridsearch   | high             | no                 | yes            | yes                  | no                |
+--------------+------------------+--------------------+----------------+----------------------+-------------------+
| Random       | high             | yes                | yes            | yes                  | yes               |
| Search       |                  |                    |                |                      |                   |
+--------------+------------------+--------------------+----------------+----------------------+-------------------+
| Genetic      | high             | yes                | not            | yes                  | yes               |
| Algorithm    |                  |                    | implemented    |                      |                   |
+--------------+------------------+--------------------+----------------+----------------------+-------------------+
| Bayesian     | low              | yes                | not            | yes                  | yes               |
| Optimizer    |                  |                    | implemented    |                      |                   |
+--------------+------------------+--------------------+----------------+----------------------+-------------------+
| Hyperopt     | low              | yes                | yes            | yes                  | yes               |
+--------------+------------------+--------------------+----------------+----------------------+-------------------+

TODOs
-----

1. algorithms:

-  implement more options for genetic algorithms
-  meta heuristics/swarm optimisation

2. functionality

-  early stopping if there is no significant improvement after x iterations

3. usability

-  better documenation

Author
------

-  **Johannes Petrat** - *Initial Release* - `johannespetrat <https://github.com/johannespetrat>`__

License
-------

This project is licensed under the MIT License - see the `LICENSE.md <LICENSE.md>`__ file for
details

.. |codecov| image:: https://codecov.io/gh/johannespetrat/OptML/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/johannespetrat/OptML
