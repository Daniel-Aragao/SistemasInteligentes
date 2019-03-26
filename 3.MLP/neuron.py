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

        self.expected_outputs = expected_outputs
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        self.create_weights(inputs, is_random)

        self.scaler__normalize = None

        if(normalize):
            inputs = self.__normalize(inputs)

        self.is_normalize = normalize

        self.inputs = None
        self.parents = None

        self.parents = parents

        if not inputs is None and type(inputs) == type([]):
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
            self.__threshold = Randomize.get_random()
        else:
            self.weights = [1 for i in range(weights_size)]
            self.__threshold = 0
        
        self.weights = [self.__threshold] + self.weights

    def __normalize(self, inputs):
        # new_inputs = Normalize.min_max(-0.5, 0.5, inputs)
        if not self.scaler__normalize:
            new_inputs, self.scaler__normalize = Normalize.scale_data(inputs)
            return new_inputs
        else:
            return self.scaler__normalize.transform(inputs)

    def __param_validation(self):
        import types

        if self.inputs is None or type(self.inputs) != type([]):
            raise Exception("\"Inputs\" can't be None and must be a function")

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

    def train(self, max_epoch=10000):
        pass
    
    def get_neuron_output(self, inputt):
        inputt = self.concatanate_threshold([inputt])[0]
        sample = Sample(inputt, None, self.weights)

        activation_potential = sample.get_activation_potential()

        return self.activation_function(activation_potential)

    def classify(self, inputs, input_normalized=False):
        if self.is_normalize and not input_normalized:
            inputs = self.__normalize(inputs)
        
        return [self.get_neuron_output(inputt) for inputt in inputs], inputs

    def __str__(self):
        string = "\nThreshold: " + str(self.__threshold) + " "
        string += "Weight: " + str(self.weights) + " "
        string += "Activation Function method: " + \
            str(self.activation_function)
        string += "Random seed:" + str(self.seed)
        return string
    
    def __repr__(self):
        return str(self.parents)
