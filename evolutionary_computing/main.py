from importer import Importer
from util import Util
from selection import Selection
from crossover import Crossover
from mutation import Mutation

from statistics import stdev
import random
import time


def genetic_algorithmn(crossover, mutate, random_seed, path="misc/ncit30.dat"):
    #### Parâmetros fixos
    N = 50 # tamanho da população inicial
    generations_limit = 200 # limite de gerações
    tax_crossover = 0.75 # probabilidade de crossover
    tax_mutation = 0.1 # probabilidade de mutação

    cities = Importer.import_cities(path)

    # population = gerar população inicial (elements: cities, random_seed: int, population_size: int): [N chromossomos]
    population = Util.generate_population(cities, random_seed, N)
    population = Selection.sort_city_chromossomes(population)
    best_result = population[0]
    best_result_cost = Util.cities_costs(best_result)

    # loop com critério de parada do algoritmo genético
    for generation in range(1, generations_limit + 1):
        # aptidões = gerar função de aptidão (population: [N chromossomos]) : [N aptidões em percentual]
        fitness = Selection.generate_fitness_city(population)

        new_population = []

        while len(new_population) < N:
            #father, mother = selecionar pais para crossover pela roleta(population,  fitness): (chromossomo, chromossomo)
            father, mother = Selection.wheel_selection(population, fitness, select=2)

            if random.random() <= tax_crossover:
                sons = crossover(father, mother) : (chromossomo, chromossomo)
            else:
                sons = [father.copy(), mother.copy()]
            
            for son in sons:
                if random.random() <= tax_mutation:
                    mutate(son) : void

                new_population.append(son)
                
        # selecionar da população atual e da nova os N mais aptos e gerar a próxima população
        population = Selection.sort_city_chromossomes(population + new_population)[0:N:]
        
        population_0_cost = Util.cities_costs(population[0])
        
        if best_result_cost > population_0_cost:
            best_result = population[0]
            best_result_cost = population_0_cost
    
    return best_result, best_result_cost, generation


if __name__ == "__main__":
    runs = 30
    instances = [
        {"crossover": Crossover.OBX, "mutate": Mutation.position_based},
        {"crossover": Crossover.OBX, "mutate": Mutation.inversion},
        {"crossover": Crossover.OX, "mutate": Mutation.position_based},
        {"crossover": Crossover.OX, "mutate": Mutation.inversion}
    ]
    
    results = []
    best_result = None
    bests_cost_summ = 0
    
    for j, instance in enumerate(instances):
        result_instance = {
                "instance": instance,
                "answers": [],
                "best": None,
                "best_cost": float('inf'),
                "costs_summ": 0,
                "costs_mean": 0,
                "costs_standard_deviation": 0,
                "time_summ": 0,
                "time_mean": 0,
                "generation_summ": 0
                "generation_mean": 0
            }
            
        results.append(result_instance)
        
        for i in range(1, runs + 1):
            time_start = time.perf_counter()
            
            result, result_cost, generation = genetic_algorithmn(random_seed=i,**instance)
            
            time_delta = time.perf_counter() - time_start
            
            result_instance["time_summ"] += time_delta
            result_instance["costs_summ"] += result_cost
            result_instance["generation_summ"] += generation
            
            result_instance["answers"].append((i, result, result_cost))
            
            if not result_instance["best"] or result_instance["best_cost"] > result_cost:
                result_instance["best"] = result
                result_instance["best_cost"] = result_cost
        
        result_instance["time_mean"] = result_instance["time_summ"] / runs
        result_instance["costs_mean"] = result_instance["costs_summ"] / runs
        result_instance["generation_mean"] = result_instance["generation_summ"] / runs
        
        result_instance["costs_standard_deviation"] = stdev([i for i in result_instance["answers"][2]])
                
        bests_cost_summ += result_instance["best_cost"]
        
        if not best_result or best_result["best_cost"] > result_instance["best_cost"]:
                best_result = result_instance
