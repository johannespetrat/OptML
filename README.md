# OptML
This package offers implementations of several black-box optimisation methods to tune hyperparameters of machine learning models. Its purpose is to enable data scientists to use optimization techniques for rapid protyping. Simply import OptML and supply it with a model and the parameters to optimize.

OptML offers a unified interface for models built with Scikit-Learn, Keras, XGBoost (and hopefully soon Statsmodels).

Author: Johannes Petrat

## Install
This package requires scikit-learn with version 0.19.0 or higher. If scikit-learn is not yet install run
`pip install scikit-learn==0.19.0`.

Afterwards install mlopt using `pip install optml` and you're ready to go.


## Features
At the moment this library includes:
* Random Search
* A simple Genetic Algorithm
* Bayesian Optimisation


## TODOs
1. algorithms:
* Hyperopt
* more options for genetic algorithms
* grid search
* meta heuristics/swarm optimisation (Ant Colony Optimization etc.)
2. functionality
* cross-validation for scoring; atm only optimises over training scores -> over-fitting
* early stopping if there is no significant improvement after x iterations
* parallelization??
* automatic detection if Keras, Scikit-learn, XGBoost or statsmodels
3. usability
* add categorical parameters
* unified APIs
* better documenation

## Assumptions
When developing I assumed that this library would be applied to models that are expensive to train i.e. that take a lot of computational resources and potentially take a long time to train. That's why I have put a focus on implementing as many (useful) algorithms as possible. Things like parallelisation and Cython implementations are not in the scope at the moment. 
There are many algorithms (including random search, grid search and genetic algorithms) that do benefit from parallelisation, though. So I may work on that in the future.
