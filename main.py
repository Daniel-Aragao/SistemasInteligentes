from neuron import Neuron
from activation_functions import exp

neuron = Neuron()
neuron.activation_function = exp

print(neuron.activation_function(2))


