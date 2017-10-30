import numpy as np
import warnings

from optml.optimizer_base import Optimizer


class HyperoptOptimizer(Optimizer):
	def __init__(self, model, hyperparams, kernel, n_restarts_optimizer, eval_func, bounds):
		pass

	def fit(self, X_train, y_train, X_test, y_test, n_iters, start_vals):
		pass