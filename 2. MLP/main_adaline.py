from adaline import Adaline
from IO_Operations import Importer
from util import Classification as testc
from util import DistanceCalcs
from show_graphics import Ploter
from IO_Operations import PrinterFile as PrintOnFileEather
from IO_Operations import Printer as PrintOnlyConsole
from IO_Operations import Exporter

######################################################### PARAMETRIZAÇÃO #########################################################
train_inputs = Importer.import_input('misc/xtrain.txt')
train_outputs = Importer.import_output('misc/dtrain.txt')

test_inputs = Importer.import_input('misc/xtest.txt')
test_outputs = Importer.import_output('misc/dtest.txt')

save_image = False
avoid_plot_it_all = True
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


def ploting(name, inputs, outputs, weights):
    Ploter.plot_results(inputs, outputs)
    Ploter.plot_line(inputs, weights)

    if save_image:
        Ploter.savefig("Execução " + name)
    else:
        Ploter.show("Execução " + name)

######################################################### ADALINE #########################################################


def executar_adaline(name, adaline: Adaline, test=True):
    weights, outputs, epochs = adaline.train()
    classify_outputs, classify_inputs = adaline.classify(test_inputs)

    testc.test_outputs("Execução " + name + " Treino",
                       outputs, train_outputs, printer=Printer)

    if test:
        testc.test_outputs("Execução " + name + " Teste",
                        classify_outputs, test_outputs, printer=Printer)

    if not avoid_plot_it_all:
        ploting(name, adaline.inputs, outputs, adaline.weights)

        if test:
            ploting(name + "_teste", classify_inputs,
                    classify_outputs, adaline.weights)

    return epochs


def get_adaline(learning_rate, normalize, is_random, outputs=train_outputs):
    return Adaline(train_inputs, outputs, learning_rate, normalize, is_random, printer=Printer)


def routine_adaline(exec, learning_rate, normalize, test=True, outputs=train_outputs):
    epochs = 0

    ####### 1 #######
    is_random = False
    epochs += executar_adaline(exec + "_1", get_adaline(
        learning_rate, normalize, is_random, outputs), test=test)

    ####### 2 #######
    is_random = True
    epochs += executar_adaline(exec + "_2", get_adaline(
        learning_rate, normalize, is_random, outputs), test=test)

    ####### 3 #######
    epochs += executar_adaline(exec + "_3", get_adaline(
        learning_rate, normalize, is_random, outputs), test=test)

    ####### 4 #######
    epochs += executar_adaline(exec + "_4", get_adaline(
        learning_rate, normalize, is_random, outputs), test=test)

    ####### 5 #######
    epochs += executar_adaline(exec + "_5", get_adaline(
        learning_rate, normalize, is_random, outputs), test=test)

    print_epoch_average(exec, epochs, 5)


############# 1 #############
normalized = True
learning_rate = 1
precision = 0.1
# 5 execuções offline
# 1 pesos e limiar nulos
# 2 a 5 aleatório
# se não convergir explique o motivo
############# 2 #############
# 1, porém online
############# 3 #############
learning_rate = 0.01
# calcular taxa de acerto quanto as amostras de treinamento
# verificar se há alterações nas fronteiras
# calcular taxa de acerto quanto as amostras de teste
############# 4 #############
# 3, porém online
############# 5 #############
# 3 e 4
precision = 0.01
############# 6 #############
# 3 e 4
precision = 0.00001
############# 7 #############
# indicar qual foi o mais eficiente (menor qtd de épocas)
# indicar qual foi o mais eficaz (nauir taxa de acerto sob treinamento de teste)


end_results_file()
