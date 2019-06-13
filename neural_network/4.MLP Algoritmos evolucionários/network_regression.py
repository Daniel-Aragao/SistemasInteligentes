import random
import time
import math
from perceptron import Perceptron
from util import Normalize
from util import Randomize
from util import Selection
from crossover import Crossover
from mutation import Mutation


class MultiLayerPerceptron:
    def __init__(self, config_network, config_neuron, config_evolutionary, seed):
        self.config_network = config_network
        self.config_neuron = config_neuron
        self.config_evolutionary = config_evolutionary
        self.seed = seed

        self.normalize_function = config_neuron['normalize_function']

        self.scaler__normalize_output = None
        self.scaler__normalize_input = None

        self.printer = self.config_neuron['printer']
        self.precision = config_neuron["precision"]
        self.expected_outputs = config_neuron["expected_outputs"]
        self.shuffle = config_neuron['shuffle']

        self.layers_node = []
        self.last_layer_nodes = []

        self.training_samples = None

        self.__init_configuration()
    
    def __normalize_input(self, inputs):
        if not self.scaler__normalize_input:
            new_inputs, self.scaler__normalize_input = self.normalize_function(inputs)
            return new_inputs
        else:
            new_inputs, self.scaler__normalize_input = self.normalize_function(inputs, self.scaler__normalize_input)
            return new_inputs
    
    def normalize_output(self, output):
        if not self.scaler__normalize_output:
            new_inputs, self.scaler__normalize_output = self.normalize_function(output)
            return new_inputs
        else:
            new_inputs, self.scaler__normalize_output = self.normalize_function(output, self.scaler__normalize_output)
            return new_inputs
        
    def __unnormalize_input(self, inputs):
        return Normalize.unscale_data(inputs, self.scaler__normalize_input)

    def __unnormalize_output(self, outputs):
        return Normalize.unscale_data(outputs, self.scaler__normalize_output)
    
    def __init_configuration(self):
        config_neuron = self.config_neuron
        config_network = self.config_network
        layers = config_network["layers"]

        self.samples = self.__normalize_input(config_neuron['inputs'])
        self.samples = Perceptron.concatanate_threshold(self.samples)

        self.expected_outputs = self.normalize_output(config_neuron['expected_outputs'])

        for index, layer in enumerate(layers):
            self.layers_node.append([])

            # if layer['neuron_type'] == 'perceptron':
            for i in range(layer['quantity']):
                normalize = None

                expected_outputs = None

                if index == 0:
                    inputs = len(self.samples[0]) - 1
                    before_layer_nodes = None
                else:
                    before_layer_nodes = self.layers_node[index - 1]
                    inputs = len(before_layer_nodes)

                p = Perceptron(inputs, expected_outputs, config_neuron['learning_rate'],
                        normalize, is_random=True, activation_function=layer['activation_function'],
                        parents=before_layer_nodes, seed=self.seed)

                self.layers_node[index].append(p)
                p.network_recursion_activation_potential = [None for i in range(len(self.samples))]
                p.network_delta = [1 for i in self.samples]


                if index == len(layers) - 1:
                    self.last_layer_nodes.append(p)
        
    @staticmethod
    def output(node, samples, sample_index: int):
        activation_potential = MultiLayerPerceptron.get_activation_potential(node, samples, sample_index)
        return node.activation_function(activation_potential)

    @staticmethod
    def get_activation_potential(node, samples, sample_index: int, update_outputs=False):
        activation_potential = 0

        if not node.parents:
            for i, inputt in enumerate(samples[sample_index]):
                    activation_potential += node.weights[i] * inputt
        else:
            activation_potential += node.weights[0] * (- 1)

            for i, parent in enumerate(node.parents):
                activation_potential += node.weights[i + 1] * MultiLayerPerceptron.output(parent, samples, sample_index)
        
        return activation_potential

    def get_error(self, sample_index: int):
        error = 0

        for node in self.last_layer_nodes:
            d = self.expected_outputs[sample_index]
            aux = d - MultiLayerPerceptron.output(node, self.samples, sample_index)

            error += aux ** 2
        
        return error/2
    
    def calc_eqm(self, weights):
        error = 0
        
        self.vector_to_weights(weights)
        
        for sample_index, sample in enumerate(self.samples):

            error += self.get_error(sample_index)
        
        return error/len(self.samples)

    def vector_to_weights(self, weights):
        cursor = 0
        
        for layer in self.layers_node:
            for perceptron in layer:
                perceptron_weights_size = len(perceptron.weights)
                perceptron.weights = weights[cursor:cursor + perceptron_weights_size:]
                
                cursor += perceptron_weights_size
    
    def weights_to_vector(self):
        elements = []
        
        for layer in self.layers_node:
            for perceptron in layer:
                elements += perceptron.weights
                
        return elements
        
    def fitness(self, weights):
        return 1 / (1 + self.calc_eqm(weights))
    
    def AG(self, max_epoch):
        # self.config_evolutionary
        N = self.config_evolutionary["population"] # tamanho da população inicial
        generations_limit = int(max_epoch / N) # limite de gerações
        tax_crossover = self.config_evolutionary["crossover_tax"] # probabilidade de crossover
        tax_mutation = self.config_evolutionary["mutation_tax"] # probabilidade de mutação
        crossover = self.config_evolutionary["crossover"] # self.config_evolutionary["crossover"]
        pcrossover = self.config_evolutionary["p-crossover"]
        mutate = self.config_evolutionary["mutation"] # self.config_evolutionary["mutation"]
        pmutation = self.config_evolutionary["p-mutation"]
        
        best_eqm = float('inf')
        best_chromossome = None
        generation_to_best = 0
        
        elements = self.weights_to_vector()
        
        population = Randomize.generate_population(elements, self.seed, N)
        
        cost_by_generation = {}
        # population = Selection.sort_MLP_chromossomes(population, self.calc_eqm)
        
        for generation in range(1, generations_limit + 1):
            fitness = Selection.generate_fitness(population, self.calc_eqm)
    
            new_population = []
    
            while len(new_population) < N:
                father, mother = Selection.wheel_selection(population, fitness, select=2)
    
                if random.random() <= tax_crossover:
                    if pcrossover:
                        sons = crossover(father, mother, pcrossover)
                    else:
                        sons = crossover(father, mother)
                else:
                    sons = [father.copy(), mother.copy()]
                
                for son in sons:
                    #if random.random() <= tax_mutation:
                    if pmutation:
                        mutate(son, tax_mutation, pmutation)
                    else:
                        mutate(son, tax_mutation)
    
                    new_population.append(son)
                    
            population = Selection.sort_MLP_chromossomes(population + new_population, self.calc_eqm)[0:N:]
            
            eqm_current = self.calc_eqm(population[0])
            
            if not best_chromossome or best_eqm > eqm_current:
                best_eqm = eqm_current
                best_chromossome = population[0]
                generation_to_best += generation
                print("Melhor EQM", eqm_current)
                
            cost_by_generation[generation] = 1/(1 + eqm_current)
            print("Geração:",generation)
            
            # print("Geração:", generation, "EQM:", eqm_current)
        
        return best_chromossome, best_eqm, generation_to_best, cost_by_generation
    
    def PSO(self, max_epoch):
        population_size = self.config_evolutionary["population"]
        c1 = self.config_evolutionary["c1"]
        c2 = self.config_evolutionary["c2"]
        w = self.config_evolutionary["w"]
        topology = self.config_evolutionary["topology"]
        vmax = 0.5
        vmin = -0.5
        
        generations_limit = int(max_epoch / population_size)
        
        elements = self.weights_to_vector()
        
        aceleration1 = [random.uniform(0, c1) for i in range(len(elements))]
        aceleration2 = [random.uniform(0, c2) for i in range(len(elements))]
        
        population = Randomize.generate_population(elements, self.seed, population_size)
        p = population.copy()
        p_fitness = [self.fitness(g) for g in p]
        
        pg = population.copy()
        pg_fitness = [self.fitness(g) for g in pg]
        v = [[random.uniform(vmin, vmax) for j in range(0,len(elements))] for i in range(0, population_size)]
        
        best_fitness = 0
        best_particle = None
        generation_to_best = 0
        cost_by_generation = {}
        
        for generation in range(1, generations_limit + 1):
            for i, particle in enumerate(population):
                particle_fitness = self.fitness(particle)
                neighbours = []
                
                if particle_fitness > p_fitness[i]:
                    p[i] = particle
                    p_fitness[i] = particle_fitness
                    
                    if best_fitness < particle_fitness:
                        best_fitness = particle_fitness
                        best_particle = particle
                        generation_to_best = generation
                        print("Melhor fitness")
                
                if topology == "star":
                    neighbours = population.copy()
                    
                elif topology == "ring":
                    j1 = i-1 if i >= 1 else population_size - 1
                    j2 = i+1 if i < population_size - 1 else 0
                    
                    neighbours = [population[j1], population[j2]]
                    
                for j in range(len(neighbours)):
                    if self.fitness(neighbours[j]) > pg_fitness[i]:
                        pg[i] = neighbours[j]
                        pg_fitness[i] = self.fitness(neighbours[j])
                
                if w:
                    v[i] = [w * k for k in v[i]]
                    
                v[i] = [q * (p[i][k] - particle[k]) for k, q in enumerate(aceleration1)]
                v[i] = [q * (pg[i][k] - particle[k]) for k, q in enumerate(aceleration2)]
                
                v[i] = [max(vmin, min(vmax, k)) for k in v[i]]
                
                population[i] = [q + particle[k] for k, q in enumerate(v[i])]
            
            cost_by_generation[generation] = best_fitness
            print("Geração:", generation)
        
        return best_particle, best_fitness, generation_to_best, cost_by_generation
        
        
    
    def EE(self, max_epoch):
        parents_size = self.config_evolutionary["parents"]
        sons_size = self.config_evolutionary["sons"]
        c = self.config_evolutionary["c"]
        sigma = self.config_evolutionary["sigma"]
        mode, scope = self.config_evolutionary["crossover"].split(" ")
        substitutions = self.config_evolutionary["substitution"].split(" ")
        k = self.config_evolutionary["k-generations"]
        
        generations_limit = int(max_epoch / (sons_size + parents_size))
        
        best_eqm = float('inf')
        best_chromossome = None
        generation_to_best = 0
        
        mutation_parents = []
        
        elements = self.weights_to_vector()
        
        population = Randomize.generate_population(elements, self.seed, parents_size)
        
        cost_by_generation = {}
        
        for generation in range(1, generations_limit + 1):
            new_population = []
    
            while len(new_population) < sons_size:
    
                sons = Crossover.dynamic(population, scope, mode)
                
                for son in sons:
                    #if random.random() <= tax_mutation:
                    Mutation.litte_disturbance(son, sigma)
                    
                    new_population.append(son)
            
            substitution_dict = {"old": population, "new": new_population}
            substitution = [i for key in substitution_dict for i in substitution_dict[key] if key in substitutions]
            
            population = Selection.sort_MLP_chromossomes(substitution, self.calc_eqm)[0:parents_size:]
            
            mutation_parents.append(population[0])
            
            if len(mutation_parents) == k ** 2:
                better = 0
                for i in range(0, k):
                    if self.calc_eqm(mutation_parents[i]) > self.calc_eqm(mutation_parents[i+k]):
                        better += 1
                
                if better / k > 1 / 5:
                    sigma = sigma / c
                elif better / k < 1 / 5:
                    sigma = sigma * c
            
            eqm_current = self.calc_eqm(population[0])
            
            if not best_chromossome or best_eqm > eqm_current:
                best_eqm = eqm_current
                best_chromossome = population[0]
                generation_to_best = generation
                print("Melhor EQM", eqm_current)
                
            cost_by_generation[generation] = 1/(1 + eqm_current)
            print("Geração:",generation)
            
            # print("Geração:", generation, "EQM:", eqm_current)
        
        return best_chromossome, best_eqm, generation_to_best, cost_by_generation
        
        
    def train(self, max_epoch=10000):
        #precision = self.config_neuron['precision']
        #learning_rate = self.config_neuron['learning_rate']
        time_begin = time.time()

        epochs = 0

        eqm_current = math.inf
        eqm_before = 0

        epochs_eqm = []
        
        weights = []
        best_eqm = 0
        generation_to_best = 0
        cost_by_generation = []
        
        if self.config_evolutionary["evolutionary_algorithmn"] == "AG":
            weights, best_eqm, generation_to_best, cost_by_generation = self.AG(max_epoch)
            
        elif self.config_evolutionary["evolutionary_algorithmn"] == "PSO":
            weights, best_eqm, generation_to_best, cost_by_generation = self.PSO(max_epoch)
            
        elif self.config_evolutionary["evolutionary_algorithmn"] == "EE":
            weights, best_eqm, generation_to_best, cost_by_generation = self.EE(max_epoch)
            
        else:
            raise Exception("Invalid evolutionary algorithmn try AG, PSO or EE: " + str(self.config_evolutionary["evolutionary_algorithmn"]))
            

        time_end = time.time()
        time_delta = time_end - time_begin
        
        self.training_samples = self.samples

        #self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        #self.printer.print_msg("EQM Final: " + str(best_eqm))
        #self.printer.print_msg("Épocas: " + str(epochs))

        return best_eqm, generation_to_best, time_delta, cost_by_generation

    def classify(self, samples):
        samples = self.__normalize_input(samples)
        samples = Perceptron.concatanate_threshold(samples)

        self.samples = samples

        outputs = []

        for sample_index, sample in enumerate(samples):
            for node in self.last_layer_nodes:
                outputs.append(MultiLayerPerceptron.output(node, samples, sample_index))
        
        # eqm = self.calc_eqm()

        # self.printer.print_msg("EQM: " + str(eqm))
        
        return outputs
        # return self.__unnormalize_output(outputs)
