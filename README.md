# HyperparameterOptimisation
Implementation of several black-box optimisation methods to tune hyperparameters of machine learning models.

The goal is to apply this to models built with Scikit-Learn, Statsmodels, Keras (and possibly other libraries) with an easy, unified interface.

Author: Johannes Petrat

## Features
At the moment this library includes:
* Random Search
* A very simple Genetic Algorithm
* Bayesian Optimisation
* 


## TODOs
1. algorithms:
* add Hyperopt
* add more options for genetic algorithms
* add grid search
* add meta heuristics/swarm optimisation (ant colony etc)
2. functionality
* add early stopping; no improvement after x iterations
* parallelization??
* add optional cross validation 
* automatic detection if Keras, Scikit-learn, XGBoost or statsmodels
3. usability
* unified APIs
* docstrings
* better documenation

## Assumption
When developing I assumed that this library would be applied to models that are expensive to train i.e. that take a lot of computational resources and potentially take a long time to train. That's why I have put a focus on implementing as many (useful) algorithms as possible. Things like parallelisation and Cython implementations are not in the scope at the moment. 
There are many algorithms (including random search, grid search and genetic algorithms) that do benefit from parallelisation, though. So I may work on that in the future.
