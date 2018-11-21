# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 15:10
# @Author  : uhauha2929
import heapq
from copy import deepcopy
from operator import attrgetter

import numpy as np


class Population(object):
    def __init__(self, size=100, gene_length=10):
        self.size = size
        self.gene_length = gene_length
        self.individuals = [Individual(self.gene_length) for _ in range(size)]
        self.fitness = self.cal_fitness()

    def cal_fitness(self):
        return np.sum([individual.fitness for individual in self.individuals])

    def add(self, individual):
        self.individuals.remove(min(self.individuals, key=attrgetter('fitness')))
        self.individuals.append(individual)


class Individual(object):
    def __init__(self, gene_length=10):
        self.gene_length = gene_length
        self.genes = np.random.binomial(1, 0.5, self.gene_length)
        self.fitness = np.sum(self.genes)


class GA(object):

    def __init__(self):
        self.population = Population()

    def _selection(self):
        p = [individual.fitness / self.population.fitness for individual in self.population.individuals]
        first, second = np.random.choice(self.population.size, size=2, p=p)
        self.fittest = self.population.individuals[first]
        self.second_fittest = self.population.individuals[second]

    def _crossover(self):
        crossover_point = np.random.randint(1, self.population.gene_length)
        temp = deepcopy(self.fittest.genes[:crossover_point])
        self.fittest.genes[:crossover_point] = self.second_fittest.genes[:crossover_point]
        self.second_fittest.genes[:crossover_point] = temp

        self.fittest.fitness = np.sum(self.fittest.genes)
        self.second_fittest.fitness = np.sum(self.second_fittest.genes)

    def _mutation(self):
        mutation_point = np.random.randint(0, self.population.gene_length)
        self.fittest.genes[mutation_point] = not self.fittest.genes[mutation_point]
        mutation_point = np.random.randint(0, self.population.gene_length)
        self.second_fittest.genes[mutation_point] = not self.second_fittest.genes[mutation_point]

        self.fittest.fitness = np.sum(self.fittest.genes)
        self.second_fittest.fitness = np.sum(self.second_fittest.genes)

    def _get_fittest_offspring(self):
        if self.fittest.fitness > self.second_fittest.fitness:
            return self.fittest
        else:
            return self.second_fittest

    def evolve(self, n=200, mu=0.01):
        for i in range(n):
            self.population.fitness = self.population.cal_fitness()
            print(i, self.population.fitness)

            self._selection()
            self._crossover()

            if np.random.rand() < mu:
                self._mutation()

            self.population.add(self._get_fittest_offspring())


if __name__ == '__main__':
    ga = GA()
    ga.evolve()
