import numpy as np
from optimizer_base import Optimizer

class GeneticOptimizer(Optimizer):
    def __init__(self, model, hyperparams, eval_func, bounds, n_init_samples, 
                 parent_selection_method, mutation_noise):
        self.model = model
        self.hyperparams = hyperparams
        self.hyperparam_history = []
        self.fitness_function = eval_func
        self.bounds = bounds        
        self.bounds_arr = np.array([bounds[hp] for hp in self.hyperparams])
        self.n_init_samples = n_init_samples
        self.parent_selection_method = parent_selection_method
        self.mutation_noise = mutation_noise

    def get_next_hyperparameters(self):
        pass

    def _sample_params(self):
        return {hp: np.random.uniform(*self.bounds[hp]) for hp in self.hyperparams}

    def init_population(self):
        return [{'params': self._sample_params()} for _ in range(self.n_init_samples)]

    def calculate_fitness(self, params, X_train, y_train, X_test, y_test):
        model = self.build_new_model(params)
        model.fit(X_train,y_train)
        score = self.fitness_function(y_test, model.predict(X_test))
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
            with_noise = self.mutation_noise[k] * np.random.randn(1)
            if with_noise < self.bounds[k][0]:
                with_noise = self.bounds[k][0]
            elif with_noise > self.bounds[k][1]:
                with_noise = self.bounds[k][1]
            params[k] += with_noise
        return params

    def select_best(self, fitnesses):
        best_idx = np.argmax([f['fitness'] for f in fitnesses])
        return fitnesses[best_idx]['params']

    def fit(self, X_train, y_train, X_test, y_test, max_iters, n_tries=5):
        """
        n_tries: number of attempts to improve the parameters. stopping condition
        """
        population = self.init_population()
        fitnesses = [self.calculate_fitness(individual['params'], X_train, y_train, X_test, y_test) for individual in population]
        self.hyperparam_history += zip([f['fitness'] for f in fitnesses],
                                        [f['params'] for f in fitnesses])

        it = 0
        improvement_count = 0
        current_best = np.max([f['fitness'] for f in fitnesses])
        while (it<max_iters):
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
            if improvement_count>= n_tries:
                break
        
        best_params = self.select_best(fitnesses)
        best_model = self.build_new_model(params)
        return best_params, best_model
