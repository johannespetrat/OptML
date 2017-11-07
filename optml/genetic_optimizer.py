import numpy as np
from optml.optimizer_base import Optimizer

class GeneticOptimizer(Optimizer):
    def __init__(self, model, hyperparams, eval_func, n_init_samples,
                 parent_selection_method, mutation_noise):
        super(GeneticOptimizer, self).__init__(model, hyperparams, eval_func)
        self.fitness_function = eval_func        
        self.bounds = {hp.name:[hp.lower, hp.upper] for hp in self.hyperparams}
        self.n_init_samples = n_init_samples
        self.parent_selection_method = parent_selection_method
        self.mutation_noise = mutation_noise

    def get_next_hyperparameters(self):
        pass

    def getParamType(self, parameter_name):
        return self.param_dict[parameter_name].param_type

    def _random_sample(self):
        sampled_params = {}
        for hp in self.hyperparams:
            v = np.random.uniform(hp.lower, hp.upper)
            if hp.param_type == 'integer':                
                sampled_params[hp.name] = int(round(v))
            else:
                sampled_params[hp.name] = v
        return sampled_params

    def init_population(self):
        return [{'params': self._random_sample()} for _ in range(self.n_init_samples)]

    def calculate_fitness(self, params, X_train, y_train, X_test=None, y_test=None):
        model = self.build_new_model(params)
        model.fit(X_train,y_train)
        if (X_test is not None) and (y_test is not None):
            score = self.fitness_function(y_test, model.predict(X_test))
        else:
            score = self.fitness_function(y_train, model.predict(X_train))
        return {'params': params, 'fitness': score}

    def cutoff_fitness(self, fitness):
        return fitness

    def select_parents(self, with_fitness, n_parents):
        fitness = [k['fitness'] for k in with_fitness]
        parents = []
        for i in range(n_parents):        
            if self.parent_selection_method=='RouletteWheel':
                r = np.random.uniform(0, sum(fitness))
                parent_idx = np.where(r<np.cumsum(fitness))[0][0]            
            elif self.parent_selection_method=='Max':
                parent_idx = np.argmax(fitness)
            parents.append(with_fitness[parent_idx])
        return parents

    def crossover(self, parents):
        final_params = {k:[] for k in parents[0]['params'].keys()}
        for parent in parents:
            for k in parent['params'].keys():
                final_params[k].append(parent['fitness'] * parent['params'][k])
        total_fitness = np.sum([p['fitness'] for p in parents])
        for k in final_params.keys():
            final_params[k] = np.sum(final_params[k])/total_fitness
        return final_params

    def mutate(self, params):
        for k in params.keys():
            with_noise = params[k] + self.mutation_noise[k] * np.random.randn(1)
            if self.getParamType(k) == 'integer':
                with_noise = int(round(with_noise))
            if with_noise < self.bounds[k][0]:
                with_noise = self.bounds[k][0]
            elif with_noise > self.bounds[k][1]:
                with_noise = self.bounds[k][1]
            params[k] = with_noise
        return params

    def select_best(self, fitnesses):
        best_idx = np.argmax([f['fitness'] for f in fitnesses])
        return fitnesses[best_idx]['params']

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_iters=10, n_tries=5):
        """
        n_tries: number of attempts to improve the parameters. stopping condition
        """
        population = self.init_population()
        fitnesses = [self.calculate_fitness(individual['params'], X_train, y_train, X_test, y_test) for individual in population]
        #self.hyperparam_history += zip([f['fitness'] for f in fitnesses],
        #                                [f['params'] for f in fitnesses])

        it = 0
        improvement_count = 0
        current_best = np.max([f['fitness'] for f in fitnesses])
        while (it<n_iters):
            fitnesses = self.cutoff_fitness(fitnesses)
            parents = self.select_parents(fitnesses, 3)
            params = self.crossover(parents)
            params = self.mutate(params)

            new_params_with_fitness = self.calculate_fitness(params, X_train, y_train, X_test, y_test)
            fitnesses.append(new_params_with_fitness)
            self.hyperparam_history.append((new_params_with_fitness['fitness'],
                                           new_params_with_fitness['params']))
            it += 1
            if new_params_with_fitness['fitness'] > current_best:
                current_best = new_params_with_fitness['fitness']
                improvement_count = 0 
            else:
                improvement_count +=1
            #if improvement_count>= n_tries:
            #    break
        
        best_params = self.select_best(fitnesses)
        best_model = self.build_new_model(params)
        return best_params, best_model
