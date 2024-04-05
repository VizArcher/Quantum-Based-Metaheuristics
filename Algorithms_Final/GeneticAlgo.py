from random import uniform
from numpy.random import randint 
import matplotlib.pyplot as plt
from Kriging_Surrogate_Model import obj
from matplotlib.ticker import MaxNLocator

TOURNAMENT_SIZE = 20
CHROMOSOME_LENGTH = 2 

class Individual:

    def __init__(self):
        self.genes = [uniform(0.25, 1), uniform(1, 2), uniform(0.3, 1)]

    def get_fitness(self):
        fitness = obj(self.genes)
        return fitness 
    
    def set_gene(self, index, value):
        self.genes[index] = value 

    def __repr__(self):
        return ','.join(str(e) for e in self.genes)

class Population:

    def __init__(self, population_size):
        self.population_size = population_size
        self.individuals = [Individual() for _ in range(population_size)]

    def get_fittest(self):

        fittest = self.individuals[0]

        for individual in self.individuals[1:]:
            if individual.get_fitness() > fittest.get_fitness():
                fittest = individual
        
        return fittest
    
    def get_fittest_elitism(self, n):
        self.individuals.sort(key = lambda ind: ind.get_fitness(), reverse=True)
        #print(self.individuals)
        #print(self.individuals[:n])
        return self.individuals[:n]

    def get_size(self):
        return self.population_size

    def get_individual(self, index):
        return self.individuals[index]
    
    def save_individual(self, index, individual):
        self.individuals[index] = individual
    
class GeneticAlgorithm:

    def __init__(self, population_size=100, crossover_rate=0.65, mutation_rate=0.1, elitism_param=5):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_param = elitism_param
        self.convergence_monitor = []

    def run(self):

        pop = Population(self.population_size)
        generation_counter = 0

        while generation_counter < 100:
            generation_counter += 1
            print('Generation #%s - fittest is: %s with fitness value %s' %(
                generation_counter, pop.get_fittest(), pop.get_fittest().get_fitness()))
            self.convergence_monitor.append(pop.get_fittest().get_fitness())
            pop = algorithm.evolve_population(pop)

        print('Solution found...')
        print(pop.get_fittest())

    def evolve_population(self, population):

        next_population = Population(self.population_size)

        # elitism
        next_population.individuals.extend(population.get_fittest_elitism(self.elitism_param))

        # crossover
        for index in range(self.elitism_param, next_population.get_size()):
            first = self.random_selection(population)
            second = self.random_selection(population)
            next_population.save_individual(index, self.crossover(first, second))

        # mutation
        for individual in next_population.individuals:
            self.mutate(individual)
        
        print(next_population)
        return next_population
    
    def crossover(self, offspring1, offspring2):
        cross_individual = Individual()

        start = randint(CHROMOSOME_LENGTH)
        end = randint(CHROMOSOME_LENGTH)

        if start > end:
            start, end = end, start

        cross_individual.genes = offspring1.genes[:start] + offspring2.genes[start:end] + offspring1.genes[end:]

        return cross_individual
    
    def mutate(self, individual):
        for index in range(CHROMOSOME_LENGTH):
            if uniform(0,1) <= self.mutation_rate:
                individual.genes[index] = randint(CHROMOSOME_LENGTH)

    # this is called tournament selection
    def random_selection(self, actual_population):

        new_population = Population(TOURNAMENT_SIZE)

        for i in range(new_population.get_size()):
            random_index = randint(new_population.get_size())
            new_population.save_individual(i, actual_population.get_individual(random_index))

        return new_population.get_fittest()
    
    def display_convergence_plot(self):
        plt.plot(self.convergence_monitor, marker='o')
        plt.title('Convergence Monitor')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.show()
    
    def values(self):
        for i in range(100):
            print(self.convergence_monitor[i])
    
if __name__ == '__main__' :
    algorithm = GeneticAlgorithm(100, 0.85, 0.015)
    algorithm.run()
    algorithm.display_convergence_plot()
    algorithm.values()