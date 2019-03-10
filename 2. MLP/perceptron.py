from util import Randomize
from IO_Operations import Printer
from sample import Sample
from activation_functions import ActivationFunctions as AF
import time
from util import Normalize

# from sklearn import preprocessing


class Perceptron:
    def __init__(self, inputs: list, expected_outputs: list, learning_rate: float = 1,
                 normalize: bool = False, is_random: bool = True, activation_function=AF.signal, printer=Printer):

        self.__expected_outputs = expected_outputs
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
            self.__normalize(inputs)

        self.inputs = Perceptron.__concatanate_threshold(
            inputs)  # [-1] + inputs

        self.__samples: Sample = []

        self.printer = printer()

    @staticmethod
    def __normalize(inputs):
        Normalize.min_max(-1, 1, inputs)
        # axis = []
        # normalized_inputs = []

        # scaler = preprocessing.StandardScaler().fit(inputs)
        # inputs = scaler.transform(inputs)

        # for i in range(len(inputs[0])):
        #     xs = [inputt[i] for inputt in inputs]

        #     for ix, x in enumerate(xs):
        #         inputs[ix][i] = Normalize.min_max(x, xs)

    def __param_validation(self):
        import types

        if self.inputs is None or type(self.inputs) != type([]):
            raise Exception("Inputs can't be None and must be a function")

        if self.weights is None or type(self.weights) != type([]):
            raise Exception("Weights can't be None and must be a list")

        if len(self.inputs[0]) != len(self.weights):
            raise Exception(
                "Inputs and Weights arrays must have the same size")

        if len(self.inputs) != len(self.__expected_outputs):
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

    def train(self, max_epoch=10000):
        time_begin = time.time()

        self.__param_validation()
        self.__samples = Perceptron.__associate_samples(
            self.inputs, self.__expected_outputs)
        outputs = []
        epochs = 0

        have_error = True

        while(have_error):
            # self.printer.print_msg("Época:" + str(epochs) + " Pesos " + str(self.__weights))
            if epochs >= max_epoch:
                break

            outputs = []
            
            have_error = False

            for sample in self.__samples:
                activation_potential = 0

                for i, inputt in enumerate(sample.inputs):
                    activation_potential += self.weights[i] * inputt

                output = self.activation_function(activation_potential)
                outputs.append(output)

                if output != sample.expected_output:
                    for i, inputt in enumerate(sample.inputs):
                        self.weights[i] += self.learning_rate * \
                            (sample.expected_output - output) * inputt

                    have_error = True

            epochs += 1

        time_end = time.time()
        time_delta = time_end - time_begin

        if epochs >= max_epoch:
            self.printer.print_msg("Máximo de épocas atingido ("+str(max_epoch)+")")

        self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        self.printer.print_msg("Pesos: " + str(self.weights[1::]))
        self.printer.print_msg("Limiar: " + str(self.weights[0]))
        self.printer.print_msg("Épocas: " + str(epochs))

        return self.weights, outputs, epochs

    def classify(self, inputs):
        inputs = Perceptron.__concatanate_threshold(inputs)

        samples = [Sample(inputt, None) for inputt in inputs]
        outputs = []

        for sample in samples:
            activation_potential = 0

            for i, inputt in enumerate(sample.inputs):
                activation_potential += self.weights[i] * inputt

            output = self.activation_function(activation_potential)
            outputs.append(output)

        return outputs, inputs

    # def __str__(self):
    #     string = "\nThreshold: " + str(self.__threshold) + " "
    #     string += "Inputs: " + str(self.inputs) + " "
    #     string += "Weight: " + str(self.weights) + " "
    #     string += "Activation Function method: " + str(self.activation_function)
    #     return string
