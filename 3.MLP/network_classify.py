import random
import time
import math
from perceptron import Perceptron
from util import Normalize


class MultiLayerPerceptron:
    def __init__(self, config_network, config_neuron):
        self.config_network = config_network
        self.config_neuron = config_neuron

        self.normalize_function = config_neuron['normalize_function']
        self.codification = config_neuron['codification']
        self.momentum = config_neuron['momentum']

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

        # self.expected_outputs = self.normalize_output(config_neuron['expected_outputs'])
        self.expected_outputs = config_neuron['expected_outputs']

        last_layer = layers[len(layers)-1]

        if self.codification == "sequencial":
            last_layer["quantity"] = config_neuron['output_classes'].bit_length()
        elif (self.codification == "oneofc"):
            last_layer["quantity"] = config_neuron['output_classes']

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
        if not update_outputs and node.network_recursion_activation_potential[sample_index]:
            return node.network_recursion_activation_potential[sample_index]

        else:
            activation_potential = 0

            if not node.parents:
                for i, inputt in enumerate(samples[sample_index]):
                        activation_potential += node.weights[i] * inputt
            else:
                activation_potential += node.weights[0] * (- 1)

                for i, parent in enumerate(node.parents):
                    activation_potential += node.weights[i + 1] * MultiLayerPerceptron.output(parent, samples, sample_index)    
            
            return activation_potential

    def clean_recursion_output(self):
        for layer in self.layers_node:
            for node in layer:
                node.network_recursion_activation_potential = [None for i in range(len(self.samples))]
                
    def update_recursion_output(self, sample_index):
        self.clean_recursion_output()

        for node in self.last_layer_nodes:
                MultiLayerPerceptron.output(node, self.samples, sample_index)
    
    def get_expected_node_output(self, output, node_index):
        if self.codification == "sequencial":
            binar = bin(int(output))
            binar = binar if len(binar) > 3 else "0b01"

            node_output = int(binar[node_index+2])

            return 1 if node_output else -1

        elif self.codification == "oneofc":
            node_output = 1 if node_index + 1 == output else -1
            # print(node_output, node_index + 1, output)
            return node_output
    
    def get_error(self, sample_index: int):
        error = 0

        for index, node in enumerate(self.last_layer_nodes):
            d = self.get_expected_node_output(self.expected_outputs[sample_index], index)
            aux = d - MultiLayerPerceptron.output(node, self.samples, sample_index)

            error += aux ** 2
        
        return error/2
    
    def calc_eqm(self):
        error = 0
        self.clean_recursion_output()
        
        for sample_index, sample in enumerate(self.samples):

            error += self.get_error(sample_index)
        
        return error/len(self.samples)

    
    def get_node_delta_output_layer(self, node, sample_index):
        node_index = self.last_layer_nodes.index(node)

        d = self.get_expected_node_output(self.expected_outputs[sample_index], node_index)
        y = MultiLayerPerceptron.output(node, self.samples, sample_index)
        u = MultiLayerPerceptron.get_activation_potential(node, self.samples, sample_index)
        g_ = node.activation_function(u, is_derivative=True)

        return ((d - y) * g_)

    def get_node_delta_hidden_layers(self, node, node_index, sample_index, next_layer):
        summ = 0

        for children_node in self.layers_node[next_layer]:
            summ += children_node.network_delta[sample_index] * children_node.weights[node_index + 1]

        u = MultiLayerPerceptron.get_activation_potential(node, self.samples, sample_index)

        return (summ * node.activation_function(u, is_derivative=True))
    
    def __shuffle(self):
        shuffle_map = [i for i in range(len(self.samples))]
        random.shuffle(shuffle_map)

        for i, new_i in enumerate(shuffle_map):
            a = self.samples[i]
            self.samples[i] = self.samples[new_i]
            self.samples[new_i] = a

            b = self.expected_outputs[i]
            self.expected_outputs[i] = self.expected_outputs[new_i]
            self.expected_outputs[new_i] = b
        
    def train(self, max_epoch=10000, offline=False):
        precision = self.config_neuron['precision']
        learning_rate = self.config_neuron['learning_rate']
        time_begin = time.time()

        epochs = 0

        eqm_current = math.inf
        eqm_before = 0

        epochs_eqm = []

        while(abs(eqm_current - eqm_before) > self.precision):
            if epochs > max_epoch:
                break

            if self.shuffle:
                self.__shuffle()                

            eqm_before = self.calc_eqm()

            if offline:
                # for sample_index, sample in enumerate(self.samples):
                #     self.update_recursion_output(sample_index)

                    for node in self.last_layer_nodes:
                        aux = [0 for i in node.parents]
                        aux_threshold = 0

                        self.clean_recursion_output()

                        for samp_index, samp in enumerate(self.samples):

                            delta = self.get_node_delta_output_layer(node, samp_index)
                            node.network_delta[samp_index] = delta

                            for index_parent, parent in enumerate(node.parents):
                                aux[index_parent] += delta * MultiLayerPerceptron.output(parent, self.samples, samp_index)
                            
                            aux_threshold += delta * -1
                                                    
                        
                        for index_parent, parent in enumerate(node.parents):
                            if not self.momentum:
                                momentum = 0
                            else:
                                momentum = self.momentum * (node.weights[index_parent + 1] - node.before_weights[index_parent + 1])
                                node.before_weights[index_parent + 1] = node.weights[index_parent + 1]
                                
                            node.weights[index_parent + 1] += momentum + node.learning_rate/len(self.samples) * aux[index_parent]

                        if not self.momentum:
                            momentum = 0
                        else:
                            momentum = self.momentum * (node.weights[0] - node.before_weights[0])
                            node.before_weights[0] = node.weights[0]
                        node.weights[0] += momentum + node.learning_rate/len(self.samples) *  aux_threshold

                    for layer_index, layer in enumerate(self.layers_node[1:len(self.layers_node) - 1:]):
                        for node_index, node in enumerate(layer):
                            aux = [0 for i in node.parents]
                            aux_threshold = 0

                            self.clean_recursion_output()

                            for samp_index, samp in enumerate(self.samples):

                                delta = self.get_node_delta_hidden_layers(node, node_index, samp_index, layer_index + 1)
                                node.network_delta[samp_index] = delta

                                for index_parent, parent in enumerate(node.parents):
                                    aux[index_parent] += delta * MultiLayerPerceptron.output(parent, self.samples, samp_index)

                            for index_parent, parent in enumerate(node.parents):
                                if not self.momentum:
                                    momentum = 0
                                else:
                                    momentum = self.momentum * (node.weights[index_parent + 1] - node.before_weights[index_parent + 1])
                                    node.before_weights[index_parent + 1] = node.weights[index_parent + 1]

                                node.weights[index_parent + 1] += momentum + node.learning_rate/len(self.samples) * aux[index_parent]

                            if not self.momentum:
                                momentum = 0
                            else:
                                momentum = self.momentum * (node.weights[0] - node.before_weights[0])
                                node.before_weights[0] = node.weights[0]
                            node.weights[0] += momentum + node.learning_rate/len(self.samples) *  aux_threshold
                    
                    for node_index, node in enumerate(self.layers_node[0]):
                        aux = [0 for i in node.weights]

                        for samp_index, samp in enumerate(self.samples):
                            delta = self.get_node_delta_hidden_layers(node, node_index, samp_index, 1)
                            node.network_delta[samp_index] = delta

                            for index_weights, weight in enumerate(node.weights):
                                aux[index_weights] = delta * samp[index_weights]
                                                
                        for index_weights, weight in enumerate(node.weights):
                            if not self.momentum:
                                momentum = 0
                            else:
                                momentum = self.momentum * (node.weights[index_weights] - node.before_weights[index_weights])
                                node.before_weights[index_weights] = node.weights[index_weights]

                            node.weights[index_weights] = weight + momentum + node.learning_rate/len(self.samples) * aux[index_weights]

            else:
                for sample_index, sample in enumerate(self.samples):
                    self.update_recursion_output(sample_index)

                    for node in self.last_layer_nodes:
                        delta = self.get_node_delta_output_layer(node, sample_index)
                        node.network_delta[sample_index] = delta

                        for index_parent, parent in enumerate(node.parents):
                            node.weights[index_parent + 1] += node.learning_rate * delta *  MultiLayerPerceptron.output(parent, self.samples, sample_index)

                        node.weights[0] += node.learning_rate * delta *  (- 1)
                    
                    for layer_index, layer in enumerate(self.layers_node[1:len(self.layers_node) - 1:]):
                        for node_index, node in enumerate(layer):
                            delta = self.get_node_delta_hidden_layers(node, node_index, sample_index, layer_index + 1)
                            node.network_delta[sample_index] = delta

                            for index_parent, parent in enumerate(node.parents):
                                node.weights[index_parent + 1] += node.learning_rate * delta *  MultiLayerPerceptron.output(parent, self.samples, sample_index)

                            node.weights[0] += node.learning_rate * delta *  (- 1)
                    
                    for node_index, node in enumerate(self.layers_node[0]):
                        delta = self.get_node_delta_hidden_layers(node, node_index, sample_index, 1)
                        node.network_delta[sample_index] = delta
                        
                        for index_weights, weight in enumerate(node.weights):
                            node.weights[index_weights] = weight + node.learning_rate * delta *  sample[index_weights]
            
            eqm_current = self.calc_eqm()

            epochs_eqm.append((epochs, eqm_current))

            epochs += 1

        time_end = time.time()
        time_delta = time_end - time_begin

        self.training_samples = self.samples

        self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        self.printer.print_msg("EQM Final: " + str(eqm_current))
        self.printer.print_msg("Épocas: " + str(epochs))

        if epochs > max_epoch:
            self.printer.print_msg(
                "Máximo de épocas atingido ("+str(max_epoch)+")")

        return epochs, epochs_eqm, eqm_current

    def classify(self, samples):
        samples = self.__normalize_input(samples)
        samples = Perceptron.concatanate_threshold(samples)

        self.samples = samples

        outputs = []

        self.clean_recursion_output()

        for sample_index, sample in enumerate(samples):
            bigger = -math.inf
            bigger_class = 0
            sequencial_bits = ""

            for index, node in enumerate(self.last_layer_nodes):
                output = MultiLayerPerceptron.output(node, samples, sample_index)
                
                if self.codification == "sequencial":
                    sequencial_bits += str(math.ceil(MultiLayerPerceptron.output(node, samples, sample_index)))

                elif (self.codification == "oneofc"):
                    if bigger < output:
                        bigger = output
                        bigger_class = index + 1
            
            if self.codification == "sequencial":
                outputs.append(int(sequencial_bits, 2))
            elif (self.codification == "oneofc"):
                outputs.append(bigger_class)
        # eqm = self.calc_eqm()

        # self.printer.print_msg("EQM: " + str(eqm))
        
        return outputs
        # return self.__unnormalize_output(outputs)
