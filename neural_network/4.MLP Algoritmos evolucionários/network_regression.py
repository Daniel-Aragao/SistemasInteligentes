import random
import time
import math
from perceptron import Perceptron
from util import Normalize
from util import Randomize
from util import Selection


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
                        parents=before_layer_nodes)

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
        
        self.vector_to_weights(weights):
        
        for sample_index, sample in enumerate(self.samples):

            error += self.get_error(sample_index)
        
        return error/len(self.samples)

    def vector_to_weights(self, weights):
        cursor = 0
        
        for layer in self.layers_node:
            for perceptron in layer:
                perceptron_weights_size = len(perceptron.weights)
                perceptron.weights = weights[cursor:perceptron_weights_size:]
                
                cursor += perceptron_weights_size
    
    def weights_to_vector(self):
        elements = []
        
        for layer in self.layers_node:
            for perceptron in layer:
                elements += perceptron.weights
                
        return elements
    
    def AG(self, max_epoch):
        # self.config_evolutionary
        generations_limit = max_epoch # limite de gerações
        N = self.config_evolutionary["population"] # tamanho da população inicial
        tax_crossover = self.config_evolutionary["crossover_tax"] # probabilidade de crossover
        tax_mutation = self.config_evolutionary["mutation_tax"] # probabilidade de mutação
        crossover = None # self.config_evolutionary["crossover"]
        mutate = None # self.config_evolutionary["mutation"]
        
        elements = self.weights_to_vector()
        
        population = Randomize.generate_population(elements, self.seed, N)
        population = Selection.sort_MLP_chromossomes(population, self.calc_eqm)
        
        for generation in range(1, generations_limit + 1):
            fitness = Selection.generate_fitness(population)
    
            new_population = []
    
            while len(new_population) < N:
                father, mother = Selection.wheel_selection(population, fitness, select=2)
    
                if random.random() <= tax_crossover:
                    sons = crossover(father, mother)
                else:
                    sons = [father.copy(), mother.copy()]
                
                for son in sons:
                    if random.random() <= tax_mutation:
                        mutate(son)
    
                    new_population.append(son)
                    
            population = Selection.sort_MLP_chromossomes(population + new_population, self.calc_eqm)[0:N:]
        
        return population[0]
    
    def PSO(self, max_epoch):
        pass
    
    def EE(self, max_epoch):
        pass
        
    def train(self, max_epoch=10000):
        #precision = self.config_neuron['precision']
        #learning_rate = self.config_neuron['learning_rate']
        time_begin = time.time()

        epochs = 0

        eqm_current = math.inf
        eqm_before = 0

        epochs_eqm = []
        
        weights = []
        if self.config_evolutionary["evolutionary_algorithmn"] == "AG":
            weights = self.AG(max_epoch)
        elif self.config_evolutionary["evolutionary_algorithmn"] == "PSO":
            weights = self.PSO(max_epoch)
        elif self.config_evolutionary["evolutionary_algorithmn"] == "EE":
            weights = self.EE(max_epoch)
        else:
            raise Exception("Invalid evolutionary algorithmn try AG, PSO or EE: " + str(self.config_evolutionary["evolutionary_algorithmn"]))
            
        
        # while(abs(eqm_current - eqm_before) > self.precision):
        #     if epochs > max_epoch:
        #        break

            #if self.shuffle:
            #    self.__shuffle()

        #    eqm_before = self.calc_eqm()

            
            #for sample_index, sample in enumerate(self.samples):
            #    self.update_recursion_output(sample_index)
            #
            #    for node in self.last_layer_nodes:
            #        delta = self.get_node_delta_output_layer(node, sample_index)
            #        node.network_delta[sample_index] = delta
            #
            #        for index_parent, parent in enumerate(node.parents):
            #            node.weights[index_parent + 1] += node.learning_rate * delta *  MultiLayerPerceptron.output(parent, self.samples, sample_index)

            #        node.weights[0] += node.learning_rate * delta *  (- 1)
            #
            #    # for layer in enumerate(self.layers_node[1:len(self.layers_node) - 1:]):
            #    #     for node_index, node in enumerate(layer):
            #    #         pass
            #
            #    for node_index, node in enumerate(self.layers_node[0]):
            #        delta = self.get_node_delta_first_layer(node, node_index, sample_index)
            #        node.network_delta[sample_index] = delta
            #
            #        for index_weights, weight in enumerate(node.weights):
            #            node.weights[index_weights] = weight + node.learning_rate * delta *  sample[index_weights]
            
        #    eqm_current = self.calc_eqm()

        #    epochs_eqm.append((epochs, eqm_current))

        #    epochs += 1

        time_end = time.time()
        time_delta = time_end - time_begin

        self.training_samples = self.samples

        self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        #self.printer.print_msg("EQM Final: " + str(eqm_current))
        #self.printer.print_msg("Épocas: " + str(epochs))

        return self.calc_eqm(weights)
        #return epochs, epochs_eqm, eqm_current

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
