import math
import random


class Util:
    
    @staticmethod
    def euclidian_distance(x1, y1, x2, y2):
        return math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
    
    @staticmethod
    def distance_between_cities(c1, c2):
        return euclidian_distance(c1[0], c1[1], c2[0], c2[1])
    
    @staticmethod
    def cities_costs(cities):
        summ = 0
        size = len(cities)
        
        for i in range(0, size - 1):
            summ += distance_between_cities(cities[i], cities[i + 1])
        
        
        return summ + distance_between_cities(cities[size - 1], cities[0])
    
    @staticmethod
    def generate_population(elements, seed, population_size):
        random.seed(seed)
        population = []
        
        for i in range(population_size):
            chromossome = [j for j in elements]
            random.shuffle(chromossome)
            
            population.append(chromossome)
        
        return population
        
    @staticmethod
    def generate_aptitudes_city(population):
        #aptitudes = []
        
        #costs = cities_costs(population)
        pass
    