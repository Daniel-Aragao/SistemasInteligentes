# config_network = {
#     "layers" :[
#         {
#             "neuron_type": perceptron,
#             "activation_funcion" : ActivationFunctions.hyperbolic_tangent,
#             "quantity": 2
#         },
#         {
#             "neuron_type": perceptron,
#             "activation_funcion" : ActivationFunctions.linear,
#             "quantity": 1
#         }
#     ]
# }

import time
import math
from perceptron import Perceptron
from util import Normalize


class MultiLayerPerceptron:
    def __init__(self, config_network, config_neuron):
        self.config_network = config_network
        self.config_neuron = config_neuron

        self.scaler__normalize_output = None
        self.scaler__normalize_input = None

        self.printer = self.config_neuron['printer'] 
        self.precision = config_neuron["precision"]
        self.expected_outputs = config_neuron["expected_outputs"]

        self.layers_node = []
        self.last_layer_nodes = []

        self.__init_configuration()
    
    def __normalize_input(self, inputs):
        # new_inputs = Normalize.min_max(-0.5, 0.5, inputs)
        if not self.scaler__normalize_input:
            new_inputs, self.scaler__normalize_input = Normalize.scale_data(inputs)
            return new_inputs
        else:
            return self.scaler__normalize_input.transform(inputs)
    
    def __normalize_output(self, output):
        # new_inputs = Normalize.min_max(-0.5, 0.5, inputs)
        if not self.scaler__normalize_output:
            new_inputs, self.scaler__normalize_output = Normalize.scale_data(output)
            return new_inputs
        else:
            new_inputs, self.scaler__normalize_output = Normalize.scale_data(self.scaler__normalize_output)
            return new_inputs
    
    def __init_configuration(self):
        config_neuron = self.config_neuron
        config_network = self.config_network
        layers = config_network["layers"]

        normalize = config_neuron['normalize']

        self.samples = self.__normalize_input(config_neuron['inputs'])
        self.samples = Perceptron.concatanate_threshold(self.samples)

        self.expected_outputs = self.__normalize_output(config_neuron['expected_outputs'])

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

    def clean_recursion_output(self, sample_index):
        for layer in self.layers_node:
            for node in layer:
                node.network_recursion_activation_potential = [None for i in range(len(self.samples))]
                
    def update_recursion_output(self, sample_index):
        self.clean_recursion_output(sample_index)

        for node in self.last_layer_nodes:
                MultiLayerPerceptron.output(node, self.samples, sample_index)
    
    def get_error(self, sample_index: int):
        error = 0

        for node in self.last_layer_nodes:
            d = self.expected_outputs[sample_index]
            aux = d - MultiLayerPerceptron.output(node, self.samples, sample_index)

            error += aux ** 2
        
        return error/2
    
    def calc_eqm(self):
        error = 0
        
        for sample_index, sample in enumerate(self.samples):
            self.clean_recursion_output(sample_index)

            error += self.get_error(sample_index)
        
        return error/len(self.samples)

    
    def get_node_delta_output_layer(self, node, sample_index):
        d = self.expected_outputs[sample_index]
        y = MultiLayerPerceptron.output(node, self.samples, sample_index)
        u = MultiLayerPerceptron.get_activation_potential(node, self.samples, sample_index)
        g_ = node.activation_function(u, is_derivative=True)

        return -((d - y) * g_)

    def get_node_delta_first_layer(self, node, node_index, sample_index):
        summ = 0

        for children_node in self.layers_node[1]:
            summ += children_node.network_delta[sample_index] * children_node.weights[node_index + 1]

        u = MultiLayerPerceptron.get_activation_potential(node, self.samples, sample_index)

        return summ * node.activation_function(u, is_derivative=True)
        
        
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

            eqm_before = self.calc_eqm()

            for sample_index, sample in enumerate(self.samples):
                self.update_recursion_output(sample_index)

                if offline:
                    for node in self.last_layer_nodes:
                        aux = [0 for i in node.parents]
                        aux_threshold = 0

                        for samp_index, samp in enumerate(self.samples):
                            delta = self.get_node_delta_output_layer(node, samp_index)
                            node.network_delta[samp_index] = delta

                            for index_parent, parent in enumerate(node.parents):
                                aux[index_parent] += delta * MultiLayerPerceptron.output(parent, self.samples, samp_index)
                            
                            aux_threshold += delta * -1
                                                    
                        
                        for index_parent, parent in enumerate(node.parents):
                            node.weights[index_parent + 1] += node.learning_rate/len(self.samples) * aux[index_parent]

                        node.weights[0] += node.learning_rate/len(self.samples) *  aux_threshold

                    
                    for node_index, node in enumerate(self.layers_node[0]):
                        aux = [0 for i in node.weights]

                        for samp_index, samp in enumerate(self.samples):
                            delta = self.get_node_delta_first_layer(node, node_index, samp_index)
                            node.network_delta[samp_index] = delta

                            for index_weights, weight in enumerate(node.weights):
                                aux[index_weights] = delta * samp[index_weights]
                                                
                        for index_weights, weight in enumerate(node.weights):
                            node.weights[index_weights] = weight + node.learning_rate/len(self.samples) * aux[index_weights]

                else:
                    for node in self.last_layer_nodes:
                        delta = self.get_node_delta_output_layer(node, sample_index)
                        node.network_delta[sample_index] = delta

                        for index_parent, parent in enumerate(node.parents):
                            node.weights[index_parent + 1] += node.learning_rate * delta *  MultiLayerPerceptron.output(parent, self.samples, sample_index)

                        node.weights[0] += node.learning_rate * delta *  (- 1)
                    
                    for node_index, node in enumerate(self.layers_node[0]):
                        delta = self.get_node_delta_first_layer(node, node_index, sample_index)
                        node.network_delta[sample_index] = delta
                        
                        for index_weights, weight in enumerate(node.weights):
                            node.weights[index_weights] = weight + node.learning_rate * delta *  sample[index_weights]
            
            eqm_current = self.calc_eqm()

            epochs_eqm.append((epochs, eqm_current))

            epochs += 1

        time_end = time.time()
        time_delta = time_end - time_begin


        self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        self.printer.print_msg("EQM Final: " + str(eqm_current))
        self.printer.print_msg("Épocas: " + str(epochs))

        if epochs > max_epoch:
            self.printer.print_msg(
                "Máximo de épocas atingido ("+str(max_epoch)+")")

        return epochs, epochs_eqm

    def classify(self, samples):
        samples = self.__normalize_input(samples)
        samples = Perceptron.concatanate_threshold(samples)

        outputs = []

        for sample_index, sample in enumerate(samples):
            self.clean_recursion_output(sample_index)
            # outputs.append([])

            for node in self.last_layer_nodes:
                # outputs[sample_index].append(MultiLayerPerceptron.output(node, samples, sample_index))
                outputs.append(MultiLayerPerceptron.output(node, samples, sample_index))
        
        return outputs
