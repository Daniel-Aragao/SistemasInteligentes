from util import Util
import random



class Selection:
    
    
    #@staticmethod
    #def cities_fitness(chromossomes):
    #    aptitudes = []
    #
    #    for chromossome in chromossomes
    #        aptitudes.append(city_chromossome_aptitude(chromossome))
    #
    #    return aptitudes
    
    @staticmethod
    def sort_city_chromossomes(population):
        return sorted(population, key=lambda e: Util.cities_costs(e))
        
    @staticmethod
    def generate_fitness_city(population, maxx=2, minn=0):
        fitness = []
        
        N = len(population)
        summ = 0
        
        for i, chromossome in enumerate(population):
            fitness_unit = minn + (maxx - minn) * ((N - (i + 1)) / (N - 1))
            
            summ += fitness_unit
            
            fitness.append(fitness_unit)
        
        
        return [i/summ for i in fitness]
    
    @staticmethod
    def wheel_selection(population, fitness, select=2):
        selecteds = []
        parents = []
        
        while len(selecteds)  < select:
            selection_number = random.random()
            summ = 0
            
            for i in range(len(fitness)):
                summ += fitness[i]
                
                if summ >= selection_number:
                    if not( i in selecteds):
                        selecteds.append(i)
                        parents.append(population[i])
                    
                    break
        
        return parents
