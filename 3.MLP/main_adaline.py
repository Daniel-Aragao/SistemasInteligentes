from adaline import Adaline
from IO_Operations import Importer
from util import Classification as testc
from util import DistanceCalcs
from util import Normalize
from show_graphics import Ploter
from IO_Operations import PrinterFile as PrintOnFileEather
from IO_Operations import Printer as PrintOnlyConsole
from IO_Operations import Exporter
import time

######################################################### PARAMETRIZAÇÃO #########################################################
train_inputs = Importer.import_input('misc/xtrain_bodyfat.txt')
train_outputs = Importer.import_output('misc/dtrain_bodyfat.txt')

test_inputs = Importer.import_input('misc/xtest_bodyfat.txt')
test_outputs = Importer.import_output('misc/dtest_bodyfat.txt')

# train_inputs = Importer.import_input('misc/xtrain.txt')
# train_outputs = Importer.import_output('misc/dtrain.txt')

# test_inputs = Importer.import_input('misc/xtest.txt')
# test_outputs = Importer.import_output('misc/dtest.txt')

save_image = True
avoid_plot_it_all = False
save_data = True
######################################################### PRÉ ROTINAS #########################################################
if save_data:
    Printer = PrintOnFileEather
else:
    Printer = PrintOnlyConsole

tempo_inicio = time.time()
tempo_inicio_local = time.localtime()
Printer.print_msg(str(tempo_inicio_local.tm_hour)+":"+str(tempo_inicio_local.tm_min)+":"+str(tempo_inicio_local.tm_sec))

def print_epoch_average(exec, epochs, qtd):
    Printer.print_msg("\nMédia de épocas para execução " +
                      exec + ": " + str(epochs/qtd) + "\n\n")


def ploting_eqm_epoch(name, epochs_eqm):
    Ploter.plot_eqm_epoch(epochs_eqm)
    title = "Execução Adaline " + name + "EQM x ÉPOCA "

    if save_image:
        Ploter.savefig(title)
    else:
        Ploter.show(title)

######################################################### ADALINE #########################################################


def executar_adaline(name, adaline: Adaline):
    weights, resulted_outputs, epochs, epochs_eqm = adaline.train()
    classify_outputs, classify_inputs = adaline.classify(test_inputs)    

    testc.test_regression_outputs("Execução Adaline " + name + " Treino",
                       resulted_outputs, adaline.normalize_output(train_outputs), printer=Printer)

    testc.test_regression_outputs("Execução Adaline " + name + " Teste",
                    classify_outputs, adaline.normalize_output(test_outputs), printer=Printer)

    if not avoid_plot_it_all:
        ploting_eqm_epoch(name, epochs_eqm)

    return epochs


def get_adaline(learning_rate, precision, is_offline, is_random, normalize):
    return Adaline(train_inputs, train_outputs, learning_rate, precision, is_offline, is_random=is_random, printer=Printer, normalize=normalize)


def routine_adaline(execution_name, learning_rate, precision, is_offline, normalize=Normalize.standard_scale_data):
    epochs = 0
    is_random = True

    ####### 1 #######
    epochs += executar_adaline(execution_name + "_1", get_adaline(
        learning_rate, precision, is_offline, is_random, normalize))

    ####### 2 #######
    epochs += executar_adaline(execution_name + "_2", get_adaline(
        learning_rate, precision, is_offline, is_random, normalize))

    ####### 3 #######
    epochs += executar_adaline(execution_name + "_3", get_adaline(
        learning_rate, precision, is_offline, is_random, normalize))

    ####### 4 #######
    epochs += executar_adaline(execution_name + "_4", get_adaline(
        learning_rate, precision, is_offline, is_random, normalize))

    ####### 5 #######
    epochs += executar_adaline(execution_name + "_5", get_adaline(
        learning_rate, precision, is_offline, is_random, normalize))

    print_epoch_average(execution_name, epochs, 5)

##################### GERAL #####################
learning_rates = [0.01, 0.1, 0.2, 0.5, 0.7, 1.0]
precision = 0.000001

############# 1 #############

for learning_rate in learning_rates:
    routine_adaline("1_"+str(learning_rate), learning_rate, precision, False)

############# 2 #############

for learning_rate in learning_rates:
    routine_adaline("2_"+str(learning_rate), learning_rate, precision, True)

############# 3 #############
######### 1 #########

for learning_rate in learning_rates:
    routine_adaline("3_1_"+str(learning_rate), learning_rate, precision, False, normalize=Normalize.min_max_scale_data)

######### 2 #########

for learning_rate in learning_rates:
    routine_adaline("3_2_"+str(learning_rate), learning_rate, precision, True, normalize=Normalize.min_max_scale_data)

# precision = 0.001
# learning_rate = 0.01
# routine_adaline("3", learning_rate, precision, True)
# ############ 4 #############
# routine_adaline("4", learning_rate, precision, False)

# ############ 5 #############
# precision = 0.01
# routine_adaline("5_3", learning_rate, precision, True)
# routine_adaline("5_4", learning_rate, precision, False)
# ############ 6 #############
# precision = 0.00001
# routine_adaline("6_3", learning_rate, precision, True)
# routine_adaline("6_4", learning_rate, precision, False)
# ############ 7 #############
# indicar qual foi o mais eficiente (menor qtd de épocas)
# indicar qual foi o mais eficaz (maior taxa de acerto sob treinamento de teste)


tempo_fim = time.time()
tempo_fim_local = time.localtime()
Printer.print_msg(str(tempo_fim_local.tm_hour)+":"+str(tempo_fim_local.tm_min)+":"+str(tempo_fim_local.tm_sec))

delta = tempo_fim - tempo_inicio

Printer.print_msg("Delta tempo: " + str(delta))
