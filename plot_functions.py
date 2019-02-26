import matplotlib.pyplot as plt
# import numpy as np

from activation_functions import ActivationFunctions as af

range_value = 100

data = [x * 0.1 for x in range(-range_value, range_value + 1)]

# activation_function = "Degrau"
# activation_function = "Degrau bipolar"
# activation_function = "Rampa simétrica"
# activation_function = "RELU"
# activation_function = "PRELU"

# activation_function = "Logísitca"
# activation_function = "Tangente Hiperbólica"
# activation_function = "Gaussiana"
activation_function = "Linear"
# activation_function = "ELU"

# activation_function = "Senóide"
# activation_function = "Arco tangente"
# activation_function = "SoftPlus"
# activation_function = "Softsign"
# activation_function = "SQLN"


if activation_function == "Degrau":
    data_step = list(map(af.step, data))
    data_step_derivative = list(map(af.step_derivative, data))
    plt.plot(data, data_step, label='Degrau')
    plt.plot(data, data_step_derivative, label='Derivada Degrau')

elif(activation_function == "Degrau bipolar"):
    data_signal = list(map(af.signal, data))
    data_signal_derivative = list(map(af.signal_derivative, data))
    plt.plot(data, data_signal, label='Degrau bipolar')
    plt.plot(data, data_signal_derivative, label='Derivada Degrau bipolar')

elif(activation_function == "Rampa simétrica"):
    data_symmetrical_ramp = list(map(af.symmetrical_ramp, data))
    data_symmetrical_ramp_derivative = list(map(af.symmetrical_ramp_derivative, data))
    plt.plot(data, data_symmetrical_ramp, label='Rampa simétrica')
    plt.plot(data, data_symmetrical_ramp_derivative, label='Derivada Rampa simétrica')

elif(activation_function == "RELU"):
    data_rectified_linear_unit = list(map(af.rectified_linear_unit, data))
    data_rectified_linear_unit_derivative = list(map(af.rectified_linear_unit_derivative, data))
    plt.plot(data, data_rectified_linear_unit, label='RELU')
    plt.plot(data, data_rectified_linear_unit_derivative, label='Derivada RELU')

elif(activation_function == "PRELU"):
    data_parametric_rectified_linear_unit = list(map(af.parametric_rectified_linear_unit, data))
    data_parametric_rectified_linear_unit_derivative = list(map(af.parametric_rectified_linear_unit_derivative, data))
    plt.plot(data, data_parametric_rectified_linear_unit, label='PRELU')
    plt.plot(data, data_parametric_rectified_linear_unit_derivative, label='Derivada PRELU')

##
elif(activation_function == "Logísitca"):
    data_logistic = list(map(af.logistic, data))
    data_logistic_derivative = list(map(af.logistic_derivative, data))
    plt.plot(data, data_logistic, label='Logísitca')
    plt.plot(data, data_logistic_derivative, label='Derivada Logística')

elif(activation_function == "Tangente Hiperbólica"):
    data_hyperbolic_tangent = list(map(af.hyperbolic_tangent, data))
    data_hyperbolic_tangent_derivative = list(map(af.hyperbolic_tangent_derivative, data))
    plt.plot(data, data_hyperbolic_tangent, label='Tangente Hiperbólica')
    plt.plot(data, data_hyperbolic_tangent_derivative, label='Derivada Tangente Hiperbólica')

elif(activation_function == "Gaussiana"):
    data_gaussian = list(map(af.gaussian, data))
    data_gaussian_derivative = list(map(af.gaussian_derivative, data))
    plt.plot(data, data_gaussian, label='Gaussiana')
    plt.plot(data, data_gaussian_derivative, label='Derivada Gaussiana')

elif(activation_function == "Linear"):
    data_linear = list(map(af.linear, data))
    data_linear_derivative = list(map(af.linear_derivative, data))
    plt.plot(data, data_linear, label='Linear')
    plt.plot(data, data_linear_derivative, label='Derivada Linear')

elif(activation_function == "ELU"):
    data_exponential_linear_unit = list(map(af.exponential_linear_unit, data))
    data_exponential_linear_unit_derivative = list(map(af.exponential_linear_unit_derivative, data))
    plt.plot(data, data_exponential_linear_unit, label='ELU')
    plt.plot(data, data_exponential_linear_unit_derivative, label='Derivada ELU')

##

elif(activation_function == "Senóide"):
    data_sinusoid = list(map(af.sinusoid, data))
    data_sinusoid_derivative = list(map(af.sinusoid_derivative, data))
    plt.plot(data, data_sinusoid, label='Senóide')
    plt.plot(data, data_sinusoid_derivative, label='Derivada Senóide')

elif(activation_function == "Arco tangente"):
    data_arc_tangente = list(map(af.arc_tangente, data))
    data_arc_tangente_derivative = list(map(af.arc_tangente_derivative, data))
    plt.plot(data, data_arc_tangente, label='Arco tangente')
    plt.plot(data, data_arc_tangente_derivative, label='Derivada Arco tangente')

elif(activation_function == "SoftPlus"):
    data_soft_plus = list(map(af.soft_plus, data))
    data_soft_plus_derivative = list(map(af.soft_plus_derivative, data))
    plt.plot(data, data_soft_plus, label='SoftPlus')
    plt.plot(data, data_soft_plus_derivative, label='Derivada SoftPlus')

elif(activation_function == "Softsign"):
    data_soft_sign = list(map(af.soft_sign, data))
    data_soft_sign_derivative = list(map(af.soft_sign_derivative, data))
    plt.plot(data, data_soft_sign, label='Softsign')
    plt.plot(data, data_soft_sign_derivative, label='Derivada Softsign')

elif(activation_function == "SQLN"):
    data_square_nonlinearity = list(map(af.square_nonlinearity, data))
    data_square_nonlinearity_derivative = list(map(af.square_nonlinearity_derivative, data))
    plt.plot(data, data_square_nonlinearity, label='SQLN')
    plt.plot(data, data_square_nonlinearity_derivative, label='Derivada SQLN')

plt.legend()
plt.show()