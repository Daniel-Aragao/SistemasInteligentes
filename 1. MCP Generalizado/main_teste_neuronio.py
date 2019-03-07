from neuron import Neuron
from activation_functions import PartiallyDiff

neuron = Neuron(2, [1, 1], [1, 1], PartiallyDiff.hard_limiter)

print(neuron)
print("Result: " + str(neuron.output()))

neuron.inputs = [1,0]
print(neuron)
print("Result: " + str(neuron.output()))

neuron.inputs = [0,1]
print(neuron)
print("Result: " + str(neuron.output()))

neuron.inputs = [0,0]
print(neuron)
print("Result: " + str(neuron.output()))