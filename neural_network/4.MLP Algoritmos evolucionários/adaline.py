from util import Randomize
from IO_Operations import Printer
from sample import Sample
from activation_functions import ActivationFunctions as AF
import time
from util import Normalize
import math

from sklearn import preprocessing

from neuron import Neuron


class Adaline(Neuron):
    def __init__(self, inputs, expected_outputs: list, learning_rate: float = 1,
                 precision: float = 0.1, is_offline=False, is_random: bool = True,
                 activation_function=AF.signal, printer=Printer, normalize=Normalize.standard_scale_data):

        super().__init__(inputs, expected_outputs, learning_rate,
                         normalize, is_random, activation_function, printer=printer)

        self.precision = precision
        self.is_offline = is_offline


    def train(self, max_epoch=10000):
        self._Neuron__param_validation()

        time_begin = time.time()

        self._Neuron__samples = self._Neuron__associate_samples(
            self.inputs, self.expected_outputs, self.weights)

        outputs = []
        epochs = 0

        eqm_current = math.inf
        eqm_before = self.calc_eqm()

        epochs_eqm = []

        while(abs(eqm_current - eqm_before) > self.precision):
            if epochs > max_epoch:
                break
            # print(epochs)

            eqm_before = self.calc_eqm()

            if self.is_offline:
                for i, sample in enumerate(self._Neuron__samples):
                    aux_input = [0 for i in self.weights]                    

                    for samp in self._Neuron__samples:
                        activation_potential = samp.get_activation_potential()

                        for index, inputt in enumerate(samp.inputs):
                            aux_input[index] +=  (samp.expected_output - activation_potential) * inputt

                    for index in range(len(self.weights)):
                        self.weights[index] += aux_input[index] * self.learning_rate/len(self._Neuron__samples)

            else:
                for i, sample in enumerate(self._Neuron__samples):
                    activation_potential = sample.get_activation_potential()

                    aux = self.learning_rate * (sample.expected_output - activation_potential) 

                    for index, weight in enumerate(self.weights):
                        self.weights[index] = weight + aux * sample.inputs[index]

            outputs = []
            for i, sample in enumerate(self._Neuron__samples):
                    activation_potential = sample.get_activation_potential()
                    outputs.append(self.activation_function(activation_potential))

            epochs += 1

            eqm_current = self.calc_eqm()

            epochs_eqm.append((epochs, eqm_current))


        time_end = time.time()
        time_delta = time_end - time_begin

        self._Neuron__training_samples = self._Neuron__samples

        if epochs > max_epoch:
            self.printer.print_msg(
                "\nMáximo de épocas atingido ("+str(max_epoch)+")")

        self.printer.print_msg("\nDuração(sec): " + str(time_delta))
        self.printer.print_msg("Pesos: " + str(self.weights[1::]))
        self.printer.print_msg("Limiar: " + str(self.weights[0]))
        self.printer.print_msg("Épocas: " + str(epochs))
        self.printer.print_msg("Random seed: " + str(self.seed))
        self.printer.print_msg("EQM Final: " + str(eqm_current))

        return self.weights, outputs, epochs, epochs_eqm
