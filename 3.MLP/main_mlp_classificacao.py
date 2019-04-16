from IO_Operations import Importer
from util import Classification as testc
from util import DistanceCalcs
from util import Normalize
from show_graphics import Ploter
from IO_Operations import PrinterFileMLP as PrinterFileMLP
from IO_Operations import Printer as PrintOnlyConsole
from IO_Operations import Exporter
from activation_functions import ActivationFunctions
import time

import traceback

from network_classify import MultiLayerPerceptron as MLP

######################################################### PARAMETRIZAÇÃO #########################################################
train_inputs = Importer.import_input('misc/xtrain_3spirals.txt')
train_outputs = Importer.import_output('misc/dtrain_3spirals.txt')

test_inputs = Importer.import_input('misc/xtest_3spirals.txt')
test_outputs = Importer.import_output('misc/dtest_3spirals.txt')

save_image = False
avoid_plot_it_all = False
save_data = False
######################################################### PRÉ ROTINAS #########################################################
if save_data:
    Printer = PrinterFileMLP
else:
    Printer = PrintOnlyConsole

tempo_inicio = time.time()
tempo_inicio_local = time.localtime()
Printer.print_msg(str(tempo_inicio_local.tm_hour)+":"+str(tempo_inicio_local.tm_min)+":"+str(tempo_inicio_local.tm_sec))

def print_epoch_average(exec, epochs, qtd):
    Printer.print_msg("\nMédia de épocas para execução " +
                      exec + ": " + str(epochs/qtd) + "\n\n")

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
    title = "Execução MLP " + name + " EQM x ÉPOCA "

    if save_image:
        Ploter.savefig(title)
    else:
        Ploter.show(title)

def executar_MLP(execution_name, mlp, is_offline):
    epochs, epochs_eqm, eqm_final = mlp.train(max_epoch=10000, offline=is_offline)

    train_results = mlp.classify(train_inputs)

    test_results = mlp.classify(test_inputs)

    testc.test_outputs("Execução MLP " + execution_name + " Treino",
                       train_results, train_outputs, printer=Printer)

    testc.test_outputs("Execução MLP " + execution_name + " Teste",
                        test_results, test_outputs, printer=Printer)

    if not avoid_plot_it_all:
        ploting_eqm_epoch(execution_name, epochs_eqm)

    return epochs

def routine_adaline(execution_name, PMC, config_neuron, is_offline=False):
    epochs = 0
    try:
        
        ####### 1 #######
        epochs += executar_MLP(execution_name + "_1", MLP(PMC, config_neuron), is_offline)

        ####### 2 #######
        epochs += executar_MLP(execution_name + "_2", MLP(PMC, config_neuron), is_offline)

        ####### 3 #######
        # epochs += executar_MLP(execution_name + "_3", MLP(PMC, config_neuron), is_offline)

        # ####### 4 #######
        # epochs += executar_MLP(execution_name + "_4", MLP(PMC, config_neuron), is_offline)

        # ####### 5 #######
        # epochs += executar_MLP(execution_name + "_5", MLP(PMC, config_neuron), is_offline)

    except Exception as e:
        Printer.print_msg(traceback.format_exc())
    print_epoch_average(execution_name, epochs, 5)


PMC1 = {
    "name": "PMC1",
    "layers" :[
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 6
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 0
        }
    ]
}

PMC2 = {
    "name": "PMC2",
    "layers" :[
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 12
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 0
        }
    ]
}

PMC3 = {
    "name": "PMC3",
    "layers" :[
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 6
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 2
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 0
        }
    ]
}

PMC4 = {
    "name": "PMC4",
    "layers" :[
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 4
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 3
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 0
        }
    ]
}

PMCs = [PMC1, PMC2, PMC3, PMC4]

##################### GERAL #####################
learning_rates = [0.01, 0.1, 0.2, 0.5]
momentuns = [0.5, 0.7, 0.9]

# learning_rates = [0.01]
config_neuron = {
    "learning_rate": 0.1,
    "precision": 0.000001,
    "inputs": train_inputs,
    "expected_outputs": train_outputs,
    "normalize_function": Normalize.min_max_scale_data,
    "printer": Printer,
    "shuffle": True,
    "momentum": None,
    "codification": "sequencial",
    "output_classes": 3
}
for PMC in PMCs:
    for learning_rate in learning_rates:
############# 1 #############
        # config_neuron["codification"] = "sequencial"
        # routine_adaline("1_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron, True)
# ############# 2 #############
        config_neuron["codification"] = "oneofc"
        routine_adaline("2_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron, True)
############# 3 #############]
        for momentum in momentuns:
            config_neuron["momentum"] = momentum
            routine_adaline("3_2_"+PMC["name"]+"_"+str(learning_rate)+"_"+str(momentum), PMC, config_neuron, True)


tempo_fim = time.time()
tempo_fim_local = time.localtime()
Printer.print_msg(str(tempo_fim_local.tm_hour)+":"+str(tempo_fim_local.tm_min)+":"+str(tempo_fim_local.tm_sec))

delta = tempo_fim - tempo_inicio

Printer.print_msg("Delta tempo: " + str(delta))