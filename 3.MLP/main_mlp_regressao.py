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

from network_controller import MultiLayerPerceptron as MLP

######################################################### PARAMETRIZAÇÃO #########################################################
train_inputs = Importer.import_input('misc/xtrain_bodyfat.txt')
train_outputs = Importer.import_output('misc/dtrain_bodyfat.txt')

test_inputs = Importer.import_input('misc/xtest_bodyfat.txt')
test_outputs = Importer.import_output('misc/dtest_bodyfat.txt')

save_image = True
avoid_plot_it_all = False
save_data = True
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

    testc.test_regression_outputs("Execução MLP " + execution_name + " Treino",
                       train_results, mlp.normalize_output(train_outputs), printer=Printer)

    testc.test_regression_outputs("Execução MLP " + execution_name + " Teste",
                        test_results, mlp.normalize_output(test_outputs), printer=Printer)

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
        epochs += executar_MLP(execution_name + "_3", MLP(PMC, config_neuron), is_offline)

        ####### 4 #######
        epochs += executar_MLP(execution_name + "_4", MLP(PMC, config_neuron), is_offline)

        ####### 5 #######
        epochs += executar_MLP(execution_name + "_5", MLP(PMC, config_neuron), is_offline)

    except Exception :
        pass
    print_epoch_average(execution_name, epochs, 5)


PMC1 = {
    "name": "PMC1",
    "layers" :[
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 2
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.linear,
            "quantity": 1
        }
    ]
}

PMC2 = {
    "name": "PMC2",
    "layers" :[
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 5
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.linear,
            "quantity": 1
        }
    ]
}

PMC3 = {
    "name": "PMC3",
    "layers" :[
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.hyperbolic_tangent,
            "quantity": 10
        },
        {
            "neuron_type": 'perceptron',
            "activation_function" : ActivationFunctions.linear,
            "quantity": 1
        }
    ]
}

PMCs = [PMC1, PMC2, PMC3]

##################### GERAL #####################
learning_rates = [0.01, 0.1, 0.2, 0.5, 0.7, 1.0]

config_neuron = {
    "learning_rate": 0.1,
    "precision": 0.000001,
    "inputs": train_inputs,
    "expected_outputs": train_outputs,
    "normalize_function": Normalize.standard_scale_data,
    "printer": Printer,
    "shuffle": True
}
############# 1 #############
for PMC in PMCs:
    for learning_rate in learning_rates[3:]:
        config_neuron["learning_rate"] = learning_rate
        routine_adaline("1_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron)

############# 2 #############
config_neuron["shuffle"] = False

for PMC in PMCs:
    for learning_rate in learning_rates:
        config_neuron["learning_rate"] = learning_rate
        routine_adaline("2_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron, True)

############# 3 #############
config_neuron["normalize_function"] = Normalize.min_max_scale_data

######### 1 #########
config_neuron["shuffle"] = True

for PMC in PMCs:
    for learning_rate in learning_rates:
        config_neuron["learning_rate"] = learning_rate
        routine_adaline("3_1_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron)

######### 2 #########
config_neuron["shuffle"] = False
for PMC in PMCs:
    for learning_rate in learning_rates:
        config_neuron["learning_rate"] = learning_rate
        routine_adaline("3_2_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron, True)

############# 4 #############
config_neuron["normalize_function"] = Normalize.standard_scale_data

config_neuron["shuffle"] = True

for PMC in PMCs:
    for learning_rate in learning_rates:
        config_neuron["learning_rate"] = learning_rate
        routine_adaline("4_2_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron, True)

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

tempo_fim = time.time()
tempo_fim_local = time.localtime()
Printer.print_msg(str(tempo_fim_local.tm_hour)+":"+str(tempo_fim_local.tm_min)+":"+str(tempo_fim_local.tm_sec))

delta = tempo_fim - tempo_inicio

Printer.print_msg("Delta tempo: " + str(delta))