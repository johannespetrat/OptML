from simanneal import Annealer

class DiscreteAnnealer(Annealer):
    def __init__(self, start_vals, eval_func, categorical_params):
        super(DiscreteAnnealer, self).__init__(start_vals)
        self.categorical_params = categorical_params
        self.eval_func = eval_func

    def move(self):
        for hp in self.hyperparams:
            if hp.param_type == 'categorical':
                self.state[hp.name] = hp.random_sample()

    def energy(self):
        return eval_func(self.state) 