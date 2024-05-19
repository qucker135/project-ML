from sklearn.ensemble import RandomForestClassifier
from settings import domain
from settings import generate_random_params_from_domain, target_function
from format_helpers import ndarray_to_params_dict, params_dict_to_ndarray
from typing import Callable
from random import randint, random


class Genetic:
    def __init__(self,
                 population_size: int,
                 offspring_size: int,
                 n_generations: int,
                 mutation_prob: float,
                 crossover: Callable[[dict, dict], dict],
                 mutation: Callable[[dict], dict],
                 random_state=None):
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.n_generations = n_generations
        self.mutation_prob = mutation_prob
        self.crossover = crossover
        self.mutation = mutation
        self.best_params = None
        self.random_state = random_state

    def fit(self, X, y, model=RandomForestClassifier):
        # random population in the beginning
        population = [
            ndarray_to_params_dict(
                generate_random_params_from_domain(domain)
            )
            for _ in range(self.population_size)
        ]
        # evaluate the population
        print(f"{population=}")
        for i in range(len(population)):
            population[i] = (
                population[i],
                target_function(
                    params_dict_to_ndarray(population[i]), X, y, model
                )
            )
        population.sort(key=lambda x: x[1], reverse=True)
        self.best_params = population[0][0]
        best_score = population[0][1]
        # population = [x[0] for x in population]
        # iterate through generations
        for _ in range(self.n_generations):
            # crossover
            offspring = [
                self.crossover(
                    population[randint(0, len(population) - 1)][0],
                    population[randint(0, len(population) - 1)][0]
                )
                for _ in range(self.offspring_size)
            ]
            # mutation
            for i in range(len(offspring)):
                if random() < self.mutation_prob:
                    offspring[i] = self.mutation(offspring[i])
            # evaluate the offspring
            for i in range(len(offspring)):
                offspring[i] = (
                    offspring[i],
                    target_function(
                        params_dict_to_ndarray(offspring[i]), X, y, model
                    )
                )
            # selection
            tmp_population = population + offspring
            tmp_population.sort(key=lambda x: x[1], reverse=True)
            population = tmp_population[:self.population_size]
            # update best_params
            if population[0][1] > best_score:
                self.best_params = population[0][0]
                best_score = population[0][1]
        print(f"{population=}")

    def get_model(self) -> RandomForestClassifier:
        if self.best_params is None:
            raise ValueError("Model has not been trained yet")
        return RandomForestClassifier(**self.best_params)
