from IO_Operations import Importer
from IO_Operations import PrinterFileMLP as PrinterFileMLP
from IO_Operations import Printer as PrintOnlyConsole
from IO_Operations import PrinterFile
from IO_Operations import Exporter
from util import Classification as testc
from util import DistanceCalcs
from util import Normalize
from show_graphics import Ploter
from activation_functions import ActivationFunctions
from mutation import Mutation
from crossover import Crossover
from network_regression import MultiLayerPerceptron as MLP

import time
import matplotlib.pyplot as plt


######################################################### PARAMETRIZAÇÃO #########################################################
train_inputs = Importer.import_input('misc/xtrain_bodyfat.txt')
train_outputs = Importer.import_output('misc/dtrain_bodyfat.txt')

test_inputs = Importer.import_input('misc/xtest_bodyfat.txt')
test_outputs = Importer.import_output('misc/dtest_bodyfat.txt')

save_image = False
avoid_plot_it_all = False
save_data = True
######################################################### PRÉ ROTINAS #########################################################
if save_data:
    Printer = PrinterFile
else:
    Printer = PrintOnlyConsole

tempo_inicio = time.time()
tempo_inicio_local = time.localtime()

def savefig(title):
        plt.savefig("./output/img/" + title+ ".png", format="PNG")
        plt.close()


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
    "sigma": 0.1,
    "k-generations": 5,
    "evolutionary_algorithmn": "EE"
}
EE1 = {
    "name": "EE-1",
    "crossover": "intermediary local",
    "substitution": "old new",
    **EEsGeneral
}
EE2 = {
    "name": "EE-2",
    "crossover": "discreet local",
    "substitution": "old new",
    **EEsGeneral
}
EE3 = {
    "name": "EE-3",
    "crossover": "intermediary global",
    "substitution": "old new",
    **EEsGeneral
}
EE4 = {
    "name": "EE-4",
    "crossover": "discreet global",
    "substitution": "old new",
    **EEsGeneral
}
EE5 = {
    "name": "EE-5",
    "crossover": "intermediary local",
    "substitution": "new",
    **EEsGeneral
}
EE6 = {
    "name": "EE-6",
    "crossover": "discreet local",
    "substitution": "new",
    **EEsGeneral
}
EE7 = {
    "name": "EE-7",
    "crossover": "intermediary global",
    "substitution": "new",
    **EEsGeneral
}
EE8 = {
    "name": "EE-8",
    "crossover": "discreet global",
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
    "w": 0,
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


#EEs = [EE1, EE2, EE3, EE4, EE5, EE6, EE7, EE8]
EEs = [EE3, EE4, EE5, EE6, EE7, EE8]
AGs = [AG1, AG2, AG3, AG4]
PSOs = [PSO1, PSO2, PSO3, PSO4]

#EA = EEs + AGs + PSOs
EA = PSOs
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
    
    for PMC in PMCs:
        eqm_summ = 0
        eqms = []
        generations_summ = 0
        time_summ = 0
        PMC_cost_by_generation = None
        pmc_name = ea["name"] + " " + PMC["name"]
        
        for run in range(1, runs + 1):
            execution_name = str(run) + ". " + pmc_name
            print(execution_name)
            
            mlp = MLP(PMC, config_neuron, ea, run)
            best_eqm, generation_to_best, time_delta, cost_by_generation = mlp.train(max_epoch=10000)
            
            eqms.append(best_eqm)
            eqm_summ += best_eqm
            generations_summ += generation_to_best
            time_summ += time_delta
            
            if not PMC_cost_by_generation:
                PMC_cost_by_generation = cost_by_generation
            else:
                for generation in cost_by_generation:
                    PMC_cost_by_generation[generation] += cost_by_generation[generation]
        
            #executar_MLP(execution_name, MLP(PMC, config_neuron, ea, run))
        
        Printer.println_msg(pmc_name + " ----------------------")
        Printer.println_msg("EQM médio:" + str(eqm_summ / runs))
        Printer.println_msg("Número de avaliações médio:" + str(generations_summ / runs))
        Printer.println_msg("Tempo médio:" + str(time_summ / runs))
        
        plt.plot([i for i in PMC_cost_by_generation], [PMC_cost_by_generation[i] for i in PMC_cost_by_generation])
        savefig(pmc_name)
    #for learning_rate in learning_rates:
        # routine_adaline("1_"+PMC["name"]+"_"+str(learning_rate), PMC, config_neuron)



tempo_fim = time.time()
tempo_fim_local = time.localtime()
Printer.print_msg(str(tempo_fim_local.tm_hour)+":"+str(tempo_fim_local.tm_min)+":"+str(tempo_fim_local.tm_sec))

delta = tempo_fim - tempo_inicio

Printer.print_msg("Delta tempo: " + str(delta))
