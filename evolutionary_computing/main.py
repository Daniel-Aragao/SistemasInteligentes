from importer import Importer
from util import Util
from selection import Selection
from crossover import Crossover
from mutation import Mutation

import matplotlib.pyplot as plt
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
    best_result_generation = 0
    
    cost_by_generation = {}

    # loop com critério de parada do algoritmo genético
    for generation in range(1, generations_limit + 1):
        # aptidões = gerar função de aptidão (population: [N chromossomos]) : [N aptidões em percentual]
        fitness = Selection.generate_fitness_city(population)

        new_population = []

        while len(new_population) < N:
            #father, mother = selecionar pais para crossover pela roleta(population,  fitness): [chromossomo, chromossomo]
            father, mother = Selection.wheel_selection(population, fitness, select=2)

            if random.random() <= tax_crossover:
                sons = crossover(father, mother)
            else:
                sons = [father.copy(), mother.copy()]
            
            for son in sons:
                if random.random() <= tax_mutation:
                    mutate(son)

                new_population.append(son)
                
        # selecionar da população atual e da nova os N mais aptos e gerar a próxima população
        population = Selection.sort_city_chromossomes(population + new_population)[0:N:]
        
        population_0_cost = Util.cities_costs(population[0])
        
        if best_result_cost > population_0_cost:
            best_result = population[0]
            best_result_cost = population_0_cost
            best_result_generation = generation
        
        cost_by_generation[generation] = best_result_cost
    
    return best_result, best_result_cost, best_result_generation, cost_by_generation


if __name__ == "__main__":
    runs = 30
    instances = [
        {"name": "OBX + Position based", "crossover": Crossover.OBX, "mutate": Mutation.position_based},
        {"name": "OBX + Inversion", "crossover": Crossover.OBX, "mutate": Mutation.inversion},
        {"name": "OX + Position based", "crossover": Crossover.OX, "mutate": Mutation.position_based},
        {"name": "OX + Inversion", "crossover": Crossover.OX, "mutate": Mutation.inversion}
    ]
    
    results = []
    best_result = None
    bests_cost_summ = 0
    
    for j, instance in enumerate(instances):
        print("----------", "Instance: " + instance["name"])
        
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
                "generation_summ": 0,
                "generation_mean": 0,
                "cost_by_generation": None
            }
            
        results.append(result_instance)
        
        for i in range(1, runs + 1):
            time_start = time.perf_counter()
            
            result, result_cost, generation, cost_by_generation = genetic_algorithmn(instance["crossover"], instance["mutate"], random_seed=i)
            
            time_delta = time.perf_counter() - time_start
            
            result_instance["time_summ"] += time_delta
            result_instance["costs_summ"] += result_cost
            result_instance["generation_summ"] += generation
            
            result_instance["answers"].append((i, result, result_cost))
            
            if not result_instance["cost_by_generation"]:
                result_instance["cost_by_generation"] = cost_by_generation.copy()
            else:
                for generation in cost_by_generation:
                    result_instance["cost_by_generation"][generation] += cost_by_generation[generation]
                    
            
            if not result_instance["best"] or result_instance["best_cost"] > result_cost:
                result_instance["best"] = result
                result_instance["best_cost"] = result_cost
                
                print("Instance best cost:", result_cost)
            
            
        for generation in result_instance["cost_by_generation"]:
            result_instance["cost_by_generation"][generation] = result_instance["cost_by_generation"][generation] / runs
        
        result_instance["time_mean"] = result_instance["time_summ"] / runs
        result_instance["costs_mean"] = result_instance["costs_summ"] / runs
        result_instance["generation_mean"] = result_instance["generation_summ"] / runs
        
        result_instance["costs_standard_deviation"] = stdev([i[2] for i in result_instance["answers"]])
                
        bests_cost_summ += result_instance["best_cost"]
        
        if not best_result or best_result["best_cost"] > result_instance["best_cost"]:
                best_result = result_instance
                print("General best cost:" + str(result_instance["best_cost"]))
        
        ri = result_instance["cost_by_generation"]
        plt.plot([i for i in ri], [ri[i] for i in ri])
        plt.show()
        
        print()
        print("Time:", result_instance["time_mean"])
        print("generation mean:", result_instance["generation_mean"])
        print("costs_mean:", result_instance["costs_mean"])
        print("costs_standard_deviation:", result_instance["costs_standard_deviation"])
        print("best_cost:", result_instance["best_cost"])
        print()