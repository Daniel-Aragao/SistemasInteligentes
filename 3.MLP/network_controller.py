"""
config_network = {
    "layers" :[
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 2
        },
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.linear,
            "quantity": 1
        }
    ]
}
"""
import time
import math
from perceptron import Perceptron as Perceptron
from IO_Operations import Printer
from util import Normalize


class MultiLayerPerceptron:
    def __init__(self, config_network, config_neuron):
        self.config_network = config_network
        self.config_neuron = config_neuron

        self.scaler__normalize = None

        self.printer = self.conf_neuron['printer'] if self.conf_neuron['printer'] else Printer

        self.nodes = []
        self.last_layer_nodes = []

        self.__init_configuration()
    
    def __normalize(self, inputs):
        # new_inputs = Normalize.min_max(-0.5, 0.5, inputs)
        if not self.scaler__normalize:
            new_inputs, self.scaler__normalize = Normalize.scale_data(inputs)
            return new_inputs
        else:
            return self.scaler__normalize.transform(inputs)
    
    def __init_configuration(self):
        conf_neuron = self.config_neuron
        config_network = self.config_network
        layers = config_network["layers"]

        normalize = conf_neuron['normalize']

        self.samples = self.__normalize(conf_neuron['inputs'])
        self.samples = Perceptron.concatanate_threshold(self.samples)

        self.expected_outputs = conf_neuron['expected_outputs']

        for index, layer in enumerate():
            self.nodes.append([])

            # if layer['neuron_type'] == 'perceptron':
            for i in range(layer['quantity']):
                normalize = None

                expected_outputs = None

                if index == 0:
                    inputs = len(self.samples)
                else:
                    inputs = layers[index - 1]['quantity']

                p = Perceptron(inputs, expected_outputs, conf_neuron['learning_rate'], 
                        normalize, is_random=True, activation_function=layer['activation_function'])

                self.nodes[index].append(p)
                p.network_recursion_activation_potential = [None for i in range(len(self.samples))]

                if index > 0:
                    before_layer_nodes = self.nodes[index - 1]
                    
                    p.parents = before_layer_nodes

                if index == len(layers) - 1:
                    self.last_layer_nodes.append(p)
    
    @staticmethod
    def output(samples, node, sample_index: int):
        activation_potential = MultiLayerPerceptron.get_activation_potential(samples, node, sample_index)
        return node.activation_function(activation_potential)

    @staticmethod
    def get_activation_potential(samples, node, sample_index: int, update_outputs=False):
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
                    activation_potential += node.weights[i + 1] * MultiLayerPerceptron.output(parent, sample_index)    
            
            return activation_potential

    def clean_recursion_output(self, sample_index):
        for layer in self.nodes:
            for node in layer:
                node.network_recursion_activation_potential[sample_index] = [None for i in range(len(self.samples))]
                
    def update_recursion_output(self, sample_index):
        self.clean_recursion_output(sample_index)

        for node in self.last_layer_nodes:
                MultiLayerPerceptron.output(node, sample_index)
    
    def get_error(self, sample_index: int):
        error = 0

        for node in self.last_layer_nodes:
            d = node.expected_outputs[sample_index]
            aux = d - MultiLayerPerceptron.output(node, sample_index)

            error += aux ** 2
        
        return error/2
    
    def calc_eqm(self, sample_index: int):
        error = 0
        
        for index, sample in enumerate(self.samples):
            error += self.get_error(index)
        
        return error/len(self.samples)
        
    def train(self, max_epoch=10000):
        precision = self.conf_neuron['precision']
        learning_rate = self.conf_neuron['learning_rate']
        time_begin = time.time()

        epochs = 0

        eqm_current = math.inf
        eqm_before = 0

        epochs_eqm = []

        while(abs(eqm_current - eqm_before) > self.precision):
            if epochs > max_epoch:
                break

            eqm_before = self.calc_eqm()
            outputs = []

            for sample_index, sample in enumerate(self.samples):
                self.update_recursion_output(sample_index)

                for node in self.last_layer_nodes:
                    d = node.expected_outputs[sample_index]
                    y = MultiLayerPerceptron.output(node, sample_index)
                    u = MultiLayerPerceptron.get_activation_potential(self.samples, node, sample_index)
                    g_ = node.activation_function(u, is_derivative=True)

                    delta = -(d - y) * g_
                    node.network_delta = delta

                    for index_parent, parent in enumerate(node.parents):
                        node.weights[index_parent + 1] += node.learning_rate * delta *  MultiLayerPerceptron.output(parent, sample_index)

                    node.weights[0] += node.learning_rate * delta *  (- 1)
                
                for node_index, node in enumerate(self.nodes[0]):
                    summ = 0

                    for children_node in self.nodes[1]:
                        summ += children_node.network_delta * children_node.weights[node_index]

                    u = MultiLayerPerceptron.get_activation_potential(self.samples, node, sample_index)

                    delta = summ * node.activation_function(u, is_derivative=True)
                    node.network_delta = delta
                    
                    for index_weights, weight in enumerate(node.weights):
                        node.weights[index_weights] = weight + node.learning_rate * delta *  sample[index_weights]
            
                outputs.append(MultiLayerPerceptron.output(self.last_layer_nodes[0], sample_index))

            self.update_recursion_output()

            eqm_current = self.calc_eqm()

            epochs += 1

        time_end = time.time()
        time_delta = time_end - time_begin


        self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        self.printer.print_msg("EQM Final: " + str(eqm_current))
        self.printer.print_msg("Épocas: " + str(epochs))

        if epochs > max_epoch:
            self.printer.print_msg(
                "Máximo de épocas atingido ("+str(max_epoch)+")")
                
        self.printer.print_msg("Random seed: " + str(self.seed))

        return epochs, epochs_eqm

    def classify(self, samples):
        samples = self.__normalize(samples)
        samples = Perceptron.concatanate_threshold(samples)

        outputs = []

        for sample_index, sample in enumerate(samples):
            self.clean_recursion_output(sample_index)
            # outputs.append([])

            for node in self.last_layer_nodes:
                # outputs[sample_index].append(MultiLayerPerceptron.output(samples, node, sample_index))
                outputs.append(MultiLayerPerceptron.output(samples, node, sample_index))
        
        return outputs
