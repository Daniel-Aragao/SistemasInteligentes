from util import Randomize
from IO_Operations import Printer
from sample import Sample
from activation_functions import ActivationFunctions as AF
import time
from util import Normalize
import math


class Neuron:
    seed_count = 0

    def __init__(self, inputs: list, expected_outputs: list, learning_rate: float = 1,
                 normalize: bool = False, is_random: bool = True, 
                 activation_function=AF.signal, printer=Printer):

        self.seed = Neuron.seed_count
        Neuron.seed_count += 1

        self.expected_outputs = expected_outputs
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        if is_random:
            self.weights = Randomize.get_random_vector(len(inputs[0]), self.seed)
            self.__threshold = Randomize.get_random()
        else:
            self.weights = [1 for i in range(len(inputs[0]))]
            self.__threshold = 0

        self.weights = [self.__threshold] + self.weights

        self.scaler = None
        if(normalize):
            inputs = self.__normalize(inputs)
        
        self.is_normalize = normalize

        self.inputs = Neuron.__concatanate_threshold(
            inputs)  # [-1] + inputs

        self.__samples: Sample = []

        self.printer = printer()

    def __normalize(self,inputs):
        # new_inputs = Normalize.min_max(-0.5, 0.5, inputs)
        if not self.scaler:
            new_inputs, self.scaler = Normalize.scale_data(inputs)
            return new_inputs
        else:
            return self.scaler.transform(inputs)


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
    def __associate_samples(inputs, outputs, weights):
        samples: Sample = []

        for i, value in enumerate(inputs):
            sample = Sample(value, outputs[i], weights)

            samples.append(sample)

        return samples

    def train(self, max_epoch=50000):
        pass

    def classify(self, inputs, input_normalized=False):
        if self.is_normalize and not input_normalized:
            inputs = self.__normalize(inputs)
            
        inputs = self.__concatanate_threshold(inputs)

        samples = [Sample(inputt, None, self.weights) for inputt in inputs]
        outputs = []

        for sample in samples:
            activation_potential = sample.get_activation_potential()

            output = self.activation_function(activation_potential)
            outputs.append(output)

        return outputs, inputs

    def __str__(self):
        string = "\nThreshold: " + str(self.__threshold) + " "
        string += "Weight: " + str(self.weights) + " "
        string += "Activation Function method: " + str(self.activation_function)
        string += "Random seed:" + str(self.seed)
        return string
