from perceptron import Perceptron
from IO_Operations import Importer
from util import TestClassification as testc
from show_graphics import Ploter
from IO_Operations import PrinterFile as PrintOnFileEather
from IO_Operations import Printer as PrintOnlyConsole
from IO_Operations import Exporter

######### PARAMETRIZAÇÃO #########
train_inputs = Importer.import_input('misc/xtrain.txt')
train_outputs = Importer.import_output('misc/dtrain.txt')

test_inputs = Importer.import_input('misc/xtest.txt')
test_outputs = Importer.import_output('misc/dtest.txt')

save_image = False
avoid_plot_it_all = False
save_data = False

######### FIM PARAMETRIZAÇÃO #########

if save_data:
    Printer = PrintOnFileEather
else:
    Printer = PrintOnlyConsole


def print_epoch_average(exec, epochs, qtd):
    Printer.print_msg("\nMédia de épocas para execução " + exec + ": " + str(epochs/qtd) +"\n\n")
    
    if save_data:
        Exporter.add_result_entry(exec, epochs/qtd)

def end_results():
    if save_data:
        Exporter.end_results_line()

def ploting(name, inputs, outputs, weights):
    Ploter.plot_results(inputs, outputs)
    Ploter.plot_line(inputs, weights)
    
    if save_image:
        Ploter.savefig("Execução " + name)
    else:
        Ploter.show("Execução " + name)


################### Perceptron ###################
def executar_perceptron(name, perceptron: Perceptron):
    weights, outputs, epochs = perceptron.train()
    classify_outputs, classify_inputs = perceptron.classify(test_inputs)

    testc.test_outputs("Execução " + name + " Treino", outputs, train_outputs, printer=Printer)
    testc.test_outputs("Execução " + name + " Teste", classify_outputs, classify_outputs, printer=Printer)

    if not avoid_plot_it_all:
        ploting(name, perceptron.inputs, outputs, perceptron.weights)
        ploting(name + "_teste", classify_inputs, classify_outputs, perceptron.weights)

    return epochs

def routine_perceptron(exec, learning_rate, normalize):
    epochs = 0

    ####### 1 #######
    is_random = False
    epochs += executar_perceptron(exec + "_1", Perceptron(train_inputs, train_outputs, learning_rate, normalize, is_random, printer=Printer))
    
    ####### 2 #######
    is_random = True
    epochs += executar_perceptron(exec + "_2", Perceptron(train_inputs, train_outputs, learning_rate, normalize, is_random, printer=Printer))
   
    ####### 3 #######
    epochs += executar_perceptron(exec + "_3", Perceptron(train_inputs, train_outputs, learning_rate, normalize, is_random, printer=Printer))
   
    ####### 4 #######
    epochs += executar_perceptron(exec + "_4", Perceptron(train_inputs, train_outputs, learning_rate, normalize, is_random, printer=Printer))
   
    ####### 5 #######
    epochs += executar_perceptron(exec + "_5", Perceptron(train_inputs, train_outputs, learning_rate, normalize, is_random, printer=Printer))

    print_epoch_average(exec, epochs,5)

############# 1 #############
routine_perceptron("1", 1, False)

############# 2 #############
routine_perceptron("2", 0.1, False)

############# 3 #############
routine_perceptron("3", 0.01, False)

############# 4 #############
####### 1 #######
routine_perceptron("4_1", 1, True)
####### 2 #######
routine_perceptron("4_2", 0.1, True)
####### 3 #######
routine_perceptron("4_3", 0.01, True)

############# 5 #############
# indiciar qual foi o mais eficiente pela média de épocas
############# 6 #############
# trocar 1 da classe A com a classe B que estejam bem próximos
# e concluir se ainda converge para cada uma das rotinas acima

end_results()