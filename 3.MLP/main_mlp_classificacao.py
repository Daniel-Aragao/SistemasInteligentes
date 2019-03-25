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