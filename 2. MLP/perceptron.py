from util import Randomize
from IO_Operations import Printer
from sample import Sample
from activation_functions import ActivationFunctions as AF
import time
from util import Normalize

from sklearn import preprocessing

from neuron import Neuron

class Perceptron(Neuron):
    def __init__(self, inputs: list, expected_outputs: list, learning_rate: float = 1,
                 normalize: bool = False, is_random: bool = True, activation_function=AF.signal, printer=Printer):

        super().__init__(inputs, expected_outputs, learning_rate,
                 normalize, is_random, activation_function, printer)

    
    def train(self, max_epoch=50000):
        self._Neuron__param_validation()

        time_begin = time.time()
        self._Neuron__samples = self._Neuron__associate_samples(
            self.inputs, self.expected_outputs)
        outputs = []
        epochs = 0

        have_error = True

        while(have_error):
            # self.printer.print_msg("Época:" + str(epochs) + " Pesos " + str(self.__weights))
            if epochs > max_epoch:
                break

            outputs = []

            have_error = False

            for sample in self._Neuron__samples:
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

        if epochs > max_epoch:
            self.printer.print_msg(
                "Máximo de épocas atingido ("+str(max_epoch)+")")

        self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        self.printer.print_msg("Pesos: " + str(self.weights[1::]))
        self.printer.print_msg("Limiar: " + str(self.weights[0]))
        self.printer.print_msg("Épocas: " + str(epochs))

        return self.weights, outputs, epochs

    def classify(self, inputs):
        inputs = self._Neuron__concatanate_threshold(inputs)

        samples = [Sample(inputt, None) for inputt in inputs]
        outputs = []

        for sample in samples:
            activation_potential = 0

            for i, inputt in enumerate(sample.inputs):
                activation_potential += self.weights[i] * inputt

            output = self.activation_function(activation_potential)
            outputs.append(output)

        return outputs, inputs
