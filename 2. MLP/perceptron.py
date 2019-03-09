from util import Randomize
from IO_Operations import Printer
from sample import Sample
from activation_functions import ActivationFunctions as AF
import time

class Perceptron:
    def __init__(self, inputs: list, expected_outputs: list, learning_rate: float = 1,
                 normalize: bool = False, is_random: bool = True, activation_function=AF.signal):

        self.__expected_outputs = expected_outputs
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        if is_random:
            self.__weights = Randomize.get_random_vector(len(inputs[0]))
            self.__threshold = Randomize.get_random()
        else:
            self.__weights = [1 for i in range(len(inputs[0]))]
            self.__threshold = 0

        self.__weights = [self.__threshold] + self.__weights
        self.__inputs = Perceptron.__concatanate_threshold(inputs) #[-1] + inputs

        self.__samples: Sample = []

        self.printer = Printer

        if(normalize):
            self.__normalize()

    def __normalize():
        raise Exception("TODO")

    def __param_validation(self):
        import types

        if self.__inputs is None or type(self.__inputs) != type([]):
            raise Exception("Inputs can't be None and must be a function")

        if self.__weights is None or type(self.__weights) != type([]):
            raise Exception("Weights can't be None and must be a list")

        if len(self.__inputs[0]) != len(self.__weights):
            raise Exception(
                "Inputs and Weights arrays must have the same size")

        if len(self.__inputs) != len(self.__expected_outputs):
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

    def train(self, max_epoch=0):
        time_begin = time.time()

        self.__param_validation()
        self.__samples = Perceptron.__associate_samples(self.__inputs, self.__expected_outputs)
        outputs = []
        epochs = 0

        have_error = True

        while(have_error):
            # self.printer.print_msg("Época:" + str(epochs) + " Pesos " + str(self.__weights))
            outputs = []

            if max_epoch and epochs >= max_epoch:
                break
            
            have_error = False

            for sample in self.__samples:
                activation_potential = 0

                for i, inputt in enumerate(sample.inputs):
                    activation_potential += self.__weights[i] * inputt

                output = self.activation_function(activation_potential)
                outputs.append(output)

                if output != sample.expected_output:
                    for i, inputt in enumerate(sample.inputs):
                        self.__weights[i] += self.learning_rate * (sample.expected_output - output) * inputt
                        
                    have_error = True

            epochs += 1
        time_end = time.time()
        time_delta = time_end - time_begin

        self.printer.print_msg("Duração(sec): " + str(time_delta))
        self.printer.print_msg("Pesos: " + str(self.__weights[1::]))
        self.printer.print_msg("Limiar: " + str(self.__weights[0]))
        self.printer.print_msg("Épocas: " + str(epochs))

        return self.__weights, outputs
    
    def classify(self, inputs):
        inputs = Perceptron.__concatanate_threshold(inputs)

        samples = [Sample(inputt, None) for inputt in inputs]
        outputs = []

        for sample in samples:
            activation_potential = 0

            for i, inputt in enumerate(sample.inputs):
                activation_potential += self.__weights[i] * inputt

            output = self.activation_function(activation_potential)
            outputs.append(output)
        
        return outputs

    # def __str__(self):
    #     string = "\nThreshold: " + str(self.__threshold) + " "
    #     string += "Inputs: " + str(self.inputs) + " "
    #     string += "Weight: " + str(self.weights) + " "
    #     string += "Activation Function method: " + str(self.activation_function)
    #     return string
