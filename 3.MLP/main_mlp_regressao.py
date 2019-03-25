from IO_Operations import Importer
from util import Classification as testc
from util import DistanceCalcs
from show_graphics import Ploter
from IO_Operations import PrinterFile as PrintOnFileEather
from IO_Operations import Printer as PrintOnlyConsole
from IO_Operations import Exporter
from activation_functions import ActivationFunctions

from perceptron_multiple_layer import MLPerceptron as perceptron

######################################################### PARAMETRIZAÇÃO #########################################################
train_inputs = Importer.import_input('misc/xtrain_bodyfat.txt')
train_outputs = Importer.import_output('misc/dtrain_bodyfat.txt')

test_inputs = Importer.import_input('misc/xtest_bodyfat.txt')
test_outputs = Importer.import_output('misc/dtest_bodyfat.txt')

save_image = False
avoid_plot_it_all = False
save_data = False
######################################################### PRÉ ROTINAS #########################################################
if save_data:
    Printer = PrintOnFileEather
else:
    Printer = PrintOnlyConsole


def print_epoch_average(exec, epochs, qtd):
    Printer.print_msg("\nMédia de épocas para execução " +
                      exec + ": " + str(epochs/qtd) + "\n\n")

    if save_data:
        Exporter.add_result_entry(exec, epochs/qtd)


def end_results_file():
    if save_data:
        Exporter.end_results_line()


def ploting_inputs_class(name, inputs, expected_outputs, outputs, weights):
    Ploter.plot_results(inputs, expected_outputs, outputs)
    Ploter.plot_line(inputs, weights)
    title = "Execução MLP " + name

    if save_image:
        Ploter.savefig(title)
    else:
        Ploter.show(title)

def ploting_eqm_epoch(name, epochs_eqm):
    Ploter.plot_eqm_epoch(epochs_eqm)
    title = "Execução MLP " + name + "EQM x ÉPOCA "

    if save_image:
        Ploter.savefig(title)
    else:
        Ploter.show(title)

PMC1 = {
    "layers" :[
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 2
        },
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.linear,
            "quantity": 1
        }
    ]
}

PMC2 = {
    "layers" :[
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 5
        },
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.linear,
            "quantity": 1
        }
    ]
}

PMC3 = {
    "layers" :[
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 10
        },
        {
            "neuron_type": perceptron,
            "activation_funcion" : ActivationFunctions.linear,
            "quantity": 1
        }
    ]
}

# Bodyfat
# uma camada escondida nas 3 topologias
#   PMC1: 2 neurônios escondidos, 1 saída
#   PMC2: 5 neurônios escondidos, 1 saída
#   PMC3: 10 neurônios escondidos, 1 saída
# escondidos utilizarão g = tangente hiperbólica
# saída usarão g = linear
# max épocas = 10000
# e = 0.000001
# pesos aleatórios entre -0.5 e 0.5

############# 1 #############
# normalização via padronização
# taxa de aprendizado variando entre 0.01, 0.1, 0.2, 0.5, 0.7, 1.0
# 5 execuções online para cada topologia e adaline
# embaralhar amostras a cada época para PMC
# anotar: 
#   o número de épocas
#   EQM final
#   EQM dos testes

############# 2 #############
# 1 sendo offline e sem embaralhar amostras a cada época

############# 3 #############
# 1 e 2 somente para PMC considerando normalização min-max([-0.5,0.5])

############# 4 #############
# qual PMC foi mais eficaz (menor erro de aproximação médio)
# alguma configuração levou ao overfitting?