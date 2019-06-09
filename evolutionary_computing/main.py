from importer import Importer
from util import Util
from selection import Selection
import random


path = "misc/ncit30.dat"

def main():
    #### Parâmetros fixos
    N = 50 # tamanho da população inicial
    generations_limit = 200 # limite de gerações
    tax_crossover = 0.75 # probabilidade de crossover
    tax_mutation = 0.1 # probabilidade de mutação

    cities = Importer.import_cities(path)

    #### Parâmetros da rotina (ignorar por hora)
    #crossover = crossover_method # função de crossover escolhida pela rotina
    #mutate = mutate_method # função de mutação escolhida pela rotina
    random_seed = 1 # vamos usar quando formos criar as rotinas na primeira fase de gerar a população inicial

    #### Legenda
    ## [array] # no python lembra uma lista dinâmica, pois posso adicionar novos elementos eternamente
    ## (tuple) # um array imutável (não da pra mudar o valor nem o adicionar novos elementos)
    ## função(parâmetros :tipo do parâmetro) : tipo do retorno

    # population = gerar população inicial (elements: cities, random_seed: int, population_size: int): [N chromossomos]
    population = Util.generate_population(cities, random_seed, N)

    # loop com critério de parada do algoritmo genético
    for i in range(generations_limit):
        
        population = Selection.sort_city_chromossomes(population)
        
        # aptidões = gerar função de aptidão (population: [N chromossomos]) : [N aptidões em percentual]
        fitness = Selection.generate_fitness_city(population)

        new_population = []

        while len(new_population) < N:
            Selection.wheel_selection(population, fitness, select=2)
            # father, mother = selecionar pais para crossover pela roleta(population,  fitness): (chromossomo, chromossomo)
            father, mother = [], []

            if random.random() <= tax_crossover:
                # son1, son2 = crossover(father, mother) : (chromossomo, chromossomo)
                son1, son2 = [], []
            else:
                son1, son2 = father, mother

            # mutate(son1, tax_mutation) : void
            # mutate(son2, tax_mutation) : void

            new_population.append(son1)
            new_population.append(son2)

        # selecionar da população atual e da nova os N mais aptos e gerar a próxima população


if __name__ == "__main__":
    main()
