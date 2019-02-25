import matplotlib.pyplot as plt
# import numpy as np

import activation_functions as af


data = [x * 0.1 for x in range(-200, 201)]

# data_step = list(map(af.PartiallyDiff.step, data))
# data_step_derivative = list(map(af.PartiallyDiff.step_derivative, data))
# plt.plot(data, data_step, label='Degrau')
# plt.plot(data, data_step_derivative, label='Derivada Degrau')

# data_signal = list(map(af.PartiallyDiff.signal, data))
# data_signal_derivative = list(map(af.PartiallyDiff.signal_derivative, data))
# plt.plot(data, data_signal, label='Degrau bipolar')
# plt.plot(data, data_signal_derivative, label='Derivada Degrau bipolar')

# data_symmetrical_ramp = list(map(af.PartiallyDiff.symmetrical_ramp, data))
# data_symmetrical_ramp_derivative = list(map(af.PartiallyDiff.symmetrical_ramp_derivative, data))
# plt.plot(data, data_symmetrical_ramp, label='Rampa simétrica')
# plt.plot(data, data_symmetrical_ramp_derivative, label='Derivada Rampa simétrica')

# data_rectified_linear_unit = list(map(af.PartiallyDiff.rectified_linear_unit, data))
# data_rectified_linear_unit_derivative = list(map(af.PartiallyDiff.rectified_linear_unit_derivative, data))
# plt.plot(data, data_rectified_linear_unit, label='RELU')
# plt.plot(data, data_rectified_linear_unit_derivative, label='Derivada RELU')

# data_parametric_rectified_linear_unit = list(map(af.PartiallyDiff.parametric_rectified_linear_unit, data))
# data_parametric_rectified_linear_unit_derivative = list(map(af.PartiallyDiff.parametric_rectified_linear_unit_derivative, data))
# plt.plot(data, data_parametric_rectified_linear_unit, label='PRELU')
# plt.plot(data, data_parametric_rectified_linear_unit_derivative, label='Derivada PRELU')

##

# data_logistic = list(map(af.FullyDiff.logistic, data))
# data_logistic_derivative = list(map(af.FullyDiff.logistic_derivative, data))
# plt.plot(data, data_logistic, label='Logísitca')
# plt.plot(data, data_logistic_derivative, label='Derivada Logística')

# data_hyperbolic_tangent = list(map(af.FullyDiff.hyperbolic_tangent, data))
# data_hyperbolic_tangent_derivative = list(map(af.FullyDiff.hyperbolic_tangent_derivative, data))
# plt.plot(data, data_hyperbolic_tangent, label='Tangente Hiperbólica')
# plt.plot(data, data_hyperbolic_tangent_derivative, label='Derivada Tangente Hiperbólica')

# data_gaussian = list(map(af.FullyDiff.gaussian, data))
# data_gaussian_derivative = list(map(af.FullyDiff.gaussian_derivative, data))
# plt.plot(data, data_gaussian, label='Gaussiana')
# plt.plot(data, data_gaussian_derivative, label='Derivada Gaussiana')

# data_linear = list(map(af.FullyDiff.linear, data))
# data_linear_derivative = list(map(af.FullyDiff.linear_derivative, data))
# plt.plot(data, data_linear, label='Linear')
# plt.plot(data, data_linear_derivative, label='Derivada Linear')

# data_exponential_linear_unit = list(map(af.FullyDiff.exponential_linear_unit, data))
# data_exponential_linear_unit_derivative = list(map(af.FullyDiff.exponential_linear_unit_derivative, data))
# plt.plot(data, data_exponential_linear_unit, label='ELU')
# plt.plot(data, data_exponential_linear_unit_derivative, label='Derivada ELU')

##

# data_sinusoid = list(map(af.OthersFunctions.sinusoid, data))
# data_sinusoid_derivative = list(map(af.OthersFunctions.sinusoid_derivative, data))
# plt.plot(data, data_sinusoid, label='Senóide')
# plt.plot(data, data_sinusoid_derivative, label='Derivada Senóide')

# data_arc_tangente = list(map(af.OthersFunctions.arc_tangente, data))
# data_arc_tangente_derivative = list(map(af.OthersFunctions.arc_tangente_derivative, data))
# plt.plot(data, data_arc_tangente, label='Arco tangente')
# plt.plot(data, data_arc_tangente_derivative, label='Derivada Arco tangente')

# data_soft_plus = list(map(af.OthersFunctions.soft_plus, data))
# data_soft_plus_derivative = list(map(af.OthersFunctions.soft_plus_derivative, data))
# plt.plot(data, data_soft_plus, label='SoftPlus')
# plt.plot(data, data_soft_plus_derivative, label='Derivada SoftPlus')

plt.legend()
plt.show()