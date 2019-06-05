from importer import Importer
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
    #random_seed = 1 # vamos usar quando formos criar as rotinas na primeira fase de gerar a população inicial
    
    #### Legenda
    ## [array] # no python lembra uma lista dinâmica, pois posso adicionar novos elementos eternamente
    ## (tuple) # um array imutável (não da pra mudar o valor nem o adicionar novos elementos)
    ## função(parâmetros :tipo do parâmetro) : tipo do retorno
    
    # population = gerar população inicial (random_seed: int): [N chromossomos]
    population = []
    
    # loop com critério de parada do algoritmo genético
    for i in range(generations_limit):
        
        # aptidões = gerar função de aptidão (population: [N chromossomos]) : [N aptidões em percentual]
        aptitudes = []
        
        new_population = []
        
        while len(new_population) < N:
            if random.random() <= tax_crossover:
                # father, mother = selecionar pais para crossover pela roleta(population,  aptidões): (chromossomo, chromossomo)
                father, mother = [], []
                
                # son1, son2 = crossover(father, mother) : (chromossomo, chromossomo)
                son1, son2 = [], []
                
                # mutate(son1, tax_mutation) : void
                # mutate(son2, tax_mutation) : void
                
                population.append(son1)
                population.append(son2)
    
    


if __name__ == "__main__":
    main()
