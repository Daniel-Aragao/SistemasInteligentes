from util import Randomize
from IO_Operations import Printer
from sample import Sample
from activation_functions import ActivationFunctions as AF
import time
from util import Normalize
import math


class Neuron:
    seed_count = 0

    def __init__(self, inputs, expected_outputs: list, learning_rate: float = 1,
                 normalize=None, is_random: bool = True,
                 activation_function=AF.signal, 
                 parents: list = None, printer=Printer):
        

        self.seed = Neuron.seed_count
        Neuron.seed_count += 1

        self.normalize_function = normalize

        self.expected_outputs = expected_outputs
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        self.create_weights(inputs, is_random)

        self.scaler__normalize_input = None
        self.scaler__normalize_output = None

        if(normalize):
            inputs = self.__normalize_input(inputs)
            self.expected_outputs = self.normalize_output(expected_outputs)

        self.normalize = normalize

        self.inputs = None
        self.parents = None

        self.parents = parents

        if not (inputs is None) and type(inputs) == type([]):
            self.inputs = Neuron.concatanate_threshold(inputs)  # [-1] + inputs

        self.__samples: Sample = []

        self.printer = printer()
    
    def create_weights(self, inputs, is_random=True):
        if type(inputs) != type([]):
            weights_size = inputs
        else:
            weights_size = len(inputs[0])

        if is_random:
            self.weights = Randomize.get_random_vector((weights_size), self.seed)
            self.before_weights = self.weights.copy()
            self.__threshold = Randomize.get_random()
        else:
            self.weights = [1 for i in range(weights_size)]
            self.before_weights = self.weights.copy()
            self.__threshold = 0
        
        self.weights = [self.__threshold] + self.weights

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

    def __param_validation(self):
        import types

        if self.inputs is None or type(self.inputs) != type([]):
            raise Exception("\"Inputs\" can't be None and must be a list")

        if self.weights is None or type(self.weights) != type([]):
            raise Exception("\"Weights\" can't be None and must be a list")

        if self.expected_outputs is None or type(self.expected_outputs) != type([]):
            raise Exception("\"Expected outputs\" can't be None and must be a list")

        if len(self.inputs[0]) != len(self.weights):
            print(self.inputs[0], self.weights)
            raise Exception(
                "Inputs and Weights arrays must have the same size")

        if len(self.inputs) != len(self.expected_outputs):
            raise Exception(
                "Inputs and Expected outputs arrays must have the same size")

        # if self.__threshold is None:
        #     raise Exception("Threshold must not be None")

        if self.activation_function is None or not isinstance(self.activation_function, types.FunctionType):
            raise Exception(
                "activation function can't be None and must be a function")

    @staticmethod
    def concatanate_threshold(inputs):
        return [[-1] + list(inputt) for inputt in inputs]

    @staticmethod
    def __associate_samples(inputs, outputs, weights):
        samples: Sample = []

        for i, value in enumerate(inputs):
            sample = Sample(value, outputs[i], weights)

            samples.append(sample)

        return samples

    def calc_eqm(self):
        summ = 0
        samples_size = len(self._Neuron__samples)

        for sample in self._Neuron__samples:
            activation_potential = sample.get_activation_potential()

            summ += ((sample.expected_output - activation_potential) ** 2)

        return summ/samples_size

    def train(self, max_epoch=10000):
        pass
    
    def classify(self, inputs):
        if self.normalize:
            inputs = self.__normalize_input(inputs)

        inputs = self.concatanate_threshold(inputs)

        samples = [Sample(inputt, None, self.weights) for inputt in inputs]
        outputs = []


        for sample in samples:
            activation_potential = sample.get_activation_potential()

            output = self.activation_function(activation_potential)
            outputs.append(output)

        self.__samples = samples
        # eqm = self.calc_eqm()
        # self.printer.print_msg("EQM Classify: " + str(eqm))

        return outputs, inputs

    def __str__(self):
        string = "\nThreshold: " + str(self.__threshold) + " "
        string += "Weight: " + str(self.weights) + " "
        string += "Activation Function method: " + \
            str(self.activation_function)
        string += "Random seed:" + str(self.seed)
        return string
    
    def __repr__(self):
        return str(self.parents)
