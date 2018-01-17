from simanneal import Annealer

class MixedAnnealer(Annealer):
    """
    Simulated Annealing to maximize the acquisition function for mixed 
    optimization problems.

    Args:
        gaussian_process: a fitted scikit-learn gaussian process regressor

    Attributes:
        state: a dictionary with hyperparameter names and their values
        bayesian_optimizer: an instance of optml BayesianOptimizer
        gaussian_process: a fitted scikit-learn gaussian process regressor
        current_parameter_idx: an index between 0 and len(bayesian_optimizer.hyperparams).
                               used for cycling through all hyperparameters
    """
    def __init__(self, bayesian_optimizer, gaussian_process):
        start_vals = bayesian_optimizer.get_random_values_dict()
        super(MixedAnnealer, self).__init__(start_vals)
        self.gaussian_process = gaussian_process
        self.bayesian_optimizer = bayesian_optimizer
        self.current_parameter_idx = 0

    def update_parameter_idx(self):
        """
        Increments the index of the current parameter and resets it to 0 if 
        it becomes greater than len(self.bayesian_optimizer.hyperparams)

        Args:
            None

        Returns:
            None
        """
        self.current_parameter_idx += 1
        self.current_parameter_idx = self.current_parameter_idx % len(self.bayesian_optimizer.hyperparams)

    def move(self):
        """
        cycle through the parameters and randomly sample one at a time

        Args:
            None

        Returns:
            None
        """
        self.update_parameter_idx()
        hp = self.bayesian_optimizer.hyperparams[self.current_parameter_idx]
        self.state[hp.name] = hp.random_sample()

    def energy(self):
        """
        Energy function for simulated annealing. Uses the acquisition function of
        the bayesian optimizer. Note that values are multiplied by -1 to convert
        from a maximization problem to a minimization problem.

        Args:
            None

        Returns:
            a float with the energy of the current state
        """
        state_input = self.bayesian_optimizer._get_ordered_param_dict(self.state)
        if self.bayesian_optimizer.acquisition_function == 'expected_improvement':
            e = -1 * self.bayesian_optimizer.expected_improvement(self.gaussian_process, [state_input])
        elif self.bayesian_optimizer.acquisition_function == 'upper_confidence_bound':
            e = -1 * self.bayesian_optimizer.probability_of_improvement(self.gaussian_process, [state_input])
        elif self.acquisition_function == 'probability_of_improvement':
            e = -1 * self.bayesian_optimizer.probability_of_improvement(self.gaussian_process, [state_input])
        elif self.acquisition_function == 'generalized_expected_improvement':
            e = -1 * self.bayesian_optimizer.generalized_expected_improvement(self.gaussian_process, [state_input])
        return e