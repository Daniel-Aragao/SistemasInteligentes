from util import Util



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
        
        for chromossome in population:
            fitness.append(minn + (maxx - minn) * ((N - i) / (N - 1)))
        
        return fitness
    
    @staticmethod
    def wheel_selection(population, fitness, select=2):
        pass