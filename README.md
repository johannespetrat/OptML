# HyperparameterOptimisation
Implementation of several black-box optimisation methods to tune hyperparameters of machine learning models.

The goal is to apply this to models built with Scikit-Learn, Statsmodels, Keras (and possibly other libraries) with an easy, unified interface.

Author: Johannes Petrat

## Install
Simply clone this repo and run 'pip install -e .' inside this directory

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
* meta heuristics/swarm optimisation (ant colony etc)
2. functionality
* cross-validation for scoring; atm only optimises over training scores -> over-fitting
* early stopping if there is no significant improvement after x iterations
* parallelization??
* add optional cross validation 
* automatic detection if Keras, Scikit-learn, XGBoost or statsmodels
3. usability
* add categorical parameters
* distinguish continuous, discrete and categorical parameters
* unified APIs
* docstrings
* better documenation

## Assumptions
When developing I assumed that this library would be applied to models that are expensive to train i.e. that take a lot of computational resources and potentially take a long time to train. That's why I have put a focus on implementing as many (useful) algorithms as possible. Things like parallelisation and Cython implementations are not in the scope at the moment. 
There are many algorithms (including random search, grid search and genetic algorithms) that do benefit from parallelisation, though. So I may work on that in the future.
