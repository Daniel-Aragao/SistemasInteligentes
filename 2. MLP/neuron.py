from util import Randomize
from IO_Operations import Printer
from sample import Sample
from activation_functions import ActivationFunctions as AF
import time
from util import Normalize
import math

from sklearn import preprocessing


class Neuron:
    def __init__(self, inputs: list, expected_outputs: list, learning_rate: float = 1,
                 normalize: bool = False, is_random: bool = True, activation_function=AF.signal, printer=Printer):

        self.expected_outputs = expected_outputs
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        if is_random:
            self.weights = Randomize.get_random_vector(len(inputs[0]))
            self.__threshold = Randomize.get_random()
        else:
            self.weights = [1 for i in range(len(inputs[0]))]
            self.__threshold = 0

        self.weights = [self.__threshold] + self.weights

        if(normalize):
            inputs = self.__normalize(inputs)

        self.inputs = Neuron.__concatanate_threshold(
            inputs)  # [-1] + inputs

        self.__samples: Sample = []

        self.printer = printer()

    @staticmethod
    def __normalize(inputs):
        new_inputs = Normalize.min_max(-1, 1, inputs)

        # new_inputs = [i for i in inputs]
        # scaler = preprocessing.StandardScaler().fit(new_inputs)
        # new_inputs = scaler.transform(new_inputs)
        return new_inputs

    def __param_validation(self):
        import types

        if self.inputs is None or type(self.inputs) != type([]):
            raise Exception("Inputs can't be None and must be a function")

        if self.weights is None or type(self.weights) != type([]):
            raise Exception("Weights can't be None and must be a list")

        if len(self.inputs[0]) != len(self.weights):
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
    def __concatanate_threshold(inputs):
        return [(-1, t[0], t[1]) for t in inputs]

    @staticmethod
    def __associate_samples(inputs, outputs):
        samples: Sample = []

        for i, value in enumerate(inputs):
            sample = Sample(value, outputs[i])

            samples.append(sample)

        return samples
    
    @staticmethod
    def calc_eqm(weights):
        pass

    def train(self, max_epoch=50000):
        pass

    def classify(self, inputs):
        pass

    def __str__(self):
        string = "\nThreshold: " + str(self.__threshold) + " "
        string += "Weight: " + str(self.weights) + " "
        string += "Activation Function method: " + str(self.activation_function)
        return string
