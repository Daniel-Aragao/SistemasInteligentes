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

activation_function = "Logísitca"
# activation_function = "Tangente Hiperbólica"
# activation_function = "Gaussiana"
# activation_function = "Linear"
# activation_function = "ELU"

# activation_function = "Senóide"
# activation_function = "Arco tangente"
# activation_function = "SoftPlus"
# activation_function = "Softsign"
# activation_function = "SQLN"

def plot_funcs(func, data, label):
    data_func = af.apply_function(func, data)
    data_func_derivative = af.apply_function(func, data, True)
    plt.plot(data, data_func, label=label)
    plt.plot(data, data_func_derivative, label='Derivada '+ label)


if activation_function == "Degrau":
    plot_funcs(af.step, data, activation_function)

elif(activation_function == "Degrau bipolar"):
    plot_funcs(af.signal, data, activation_function)

elif(activation_function == "Rampa simétrica"):
    plot_funcs(af.symmetrical_ramp, data, activation_function)

elif(activation_function == "RELU"):
    plot_funcs(af.rectified_linear_unit, data, activation_function)

elif(activation_function == "PRELU"):
    plot_funcs(af.parametric_rectified_linear_unit, data, activation_function)

##
elif(activation_function == "Logísitca"):
    plot_funcs(af.logistic, data, activation_function)

elif(activation_function == "Tangente Hiperbólica"):
    plot_funcs(af.hyperbolic_tangent, data, activation_function)

elif(activation_function == "Gaussiana"):
    plot_funcs(af.gaussian, data, activation_function)

elif(activation_function == "Linear"):
    plot_funcs(af.linear, data, activation_function)

elif(activation_function == "ELU"):
    plot_funcs(af.exponential_linear_unit, data, activation_function)

##

elif(activation_function == "Senóide"):
    plot_funcs(af.sinusoid, data, activation_function)

elif(activation_function == "Arco tangente"):
    plot_funcs(af.arc_tangente, data, activation_function)

elif(activation_function == "SoftPlus"):
    plot_funcs(af.soft_plus, data, activation_function)

elif(activation_function == "Softsign"):
    plot_funcs(af.soft_sign, data, activation_function)

elif(activation_function == "SQLN"):
    plot_funcs(af.square_nonlinearity, data, activation_function)

plt.legend()
plt.show()