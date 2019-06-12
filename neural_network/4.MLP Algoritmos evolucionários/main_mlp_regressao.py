from IO_Operations import Importer
from IO_Operations import PrinterFileMLP as PrinterFileMLP
from IO_Operations import Printer as PrintOnlyConsole
from IO_Operations import Exporter
from util import Classification as testc
from util import DistanceCalcs
from util import Normalize
from show_graphics import Ploter
from activation_functions import ActivationFunctions
from mutation import Mutation
from crossover import Crossover

import time

from network_regression import MultiLayerPerceptron as MLP

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

def executar_MLP(execution_name, mlp):
    epochs, epochs_eqm, eqm_final = mlp.train(max_epoch=10000)

    train_results = mlp.classify(train_inputs)

    test_results = mlp.classify(test_inputs)

    testc.test_regression_outputs("Execução MLP " + execution_name + " Treino",
                       train_results, mlp.normalize_output(train_outputs), printer=Printer)

    testc.test_regression_outputs("Execução MLP " + execution_name + " Teste",
                        test_results, mlp.normalize_output(test_outputs), printer=Printer)

    #if not avoid_plot_it_all:
    #    ploting_eqm_epoch(execution_name, epochs_eqm)

    #return epochs

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

EEsGeneral = {
    "parents": 20,
    "sons": 140,
    "c": 0.8,
    "evolutionary_algorithmn": "EE"
}
EE1 = {
    "name": "EE-1",
    "crossover": "intermediaria local",
    "substitution": "old+new",
    **EEsGeneral
}
EE2 = {
    "name": "EE-2",
    "crossover": "discreta local",
    "substitution": "old+new",
    **EEsGeneral
}
EE3 = {
    "name": "EE-3",
    "crossover": "intermediaria global",
    "substitution": "old+new",
    **EEsGeneral
}
EE4 = {
    "name": "EE-4",
    "crossover": "discreta global",
    "substitution": "old+new",
    **EEsGeneral
}
EE5 = {
    "name": "EE-5",
    "crossover": "intermediaria local",
    "substitution": "new",
    **EEsGeneral
}
EE6 = {
    "name": "EE-6",
    "crossover": "discreta local",
    "substitution": "new",
    **EEsGeneral
}
EE7 = {
    "name": "EE-7",
    "crossover": "intermediaria global",
    "substitution": "new",
    **EEsGeneral
}
EE8 = {
    "name": "EE-8",
    "crossover": "discreta global",
    "substitution": "new",
    **EEsGeneral
}

AGsGeneral = {
    "population": 20,
    "crossover": Crossover.BLX,
    "p-crossover": 0.5,
    "mutation": Mutation.gaussian,
    "p-mutation": 0.1,
    "selection": "wheel",
    "substitution": "old+new",
    "evolutionary_algorithmn": "AG"
}
AG1 = {
    "name": "AG-1",
    "crossover_tax": 0.7,
    "mutation_tax": 0.1,
    **AGsGeneral
}
AG2 = {
    "name": "AG-2",
    "crossover_tax": 0.9,
    "mutation_tax": 0.1,
    **AGsGeneral
}
AG3 = {
    "name": "AG-3",
    "crossover_tax": 0.7,
    "mutation_tax": 0.01,
    **AGsGeneral
}
AG4 = {
    "name": "AG-4",
    "crossover_tax": 0.9,
    "mutation_tax": 0.01,
    **AGsGeneral
}

PSOsGeneral = {
    "population": 20,
    "c1": 2.05,
    "c2": 2.05,
    "evolutionary_algorithmn": "PSO"
}
PSO1 = {
    "name": "PSO-1",
    "model": "canonical",
    "topology": "star",
    **PSOsGeneral
}
PSO2 = {
    "name": "PSO-2",
    "model": "canonical",
    "topology": "ring",
    **PSOsGeneral
}
PSO3 = {
    "name": "PSO-3",
    "model": "inercia",
    "w": 0.9,
    "topology": "star",
    **PSOsGeneral
}
PSO4 = {
    "name": "PSO-4",
    "model": "inercia",
    "w": 0.9,
    "topology": "ring",
    **PSOsGeneral
}


EEs = [EE1, EE2, EE3, EE4, EE5, EE6, EE7, EE8]
AGs = [AG1, AG2, AG3, AG4]
PSOs = [PSO1, PSO2, PSO3, PSO4]

#EA = EEs + AGs + PSOs
EA = AGs
PMCs = [PMC1, PMC2, PMC3]
runs = 10

##################### GERAL #####################
learning_rates = [0.01, 0.1, 0.2, 0.5, 0.7, 1.0]
# learning_rates = [0.01]
config_neuron = {
    "learning_rate": 0.1,
    "precision": 0.000001,
    "inputs": train_inputs,
    "expected_outputs": train_outputs,
    "normalize_function": Normalize.min_max_scale_data,
    "printer": Printer,
    "shuffle": False
}
config_AG = {
    "crossover_tax": 0.7,
    "crossover_tax": 0.1
}


############# 1 #############
for ea in EA:
    for run in range(1, runs + 1):
        for PMC in PMCs:
            execution_name = str(run) + ". " + PMC["name"] + " " + ea["name"]
            
            best_eqm, generation_to_best, time_delta = mlp.train(max_epoch=10000)
            #executar_MLP(execution_name, MLP(PMC, config_neuron, ea, run))
    
    #for learning_rate in learning_rates:
        # routine_adaline("1_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron)



tempo_fim = time.time()
tempo_fim_local = time.localtime()
Printer.print_msg(str(tempo_fim_local.tm_hour)+":"+str(tempo_fim_local.tm_min)+":"+str(tempo_fim_local.tm_sec))

delta = tempo_fim - tempo_inicio

Printer.print_msg("Delta tempo: " + str(delta))