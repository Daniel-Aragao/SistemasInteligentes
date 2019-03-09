from perceptron import Perceptron
from IO_Operations import Importer
from util import TestClassification as testc

################### Perceptron ###################
############# 1 #############
# não normalizado
# 𝜂 = 1

# 5 execuções
# 1ª pesos e limiar devem começar nulos
# 2ª a 5ª pesos e limiar devem ser aleatórios entre 0 e 1

#saídas desejadas:
#   valores de pesos
#   valor do limiar
#   número de épocas

#obs.: testar com os arquivos de teste
inputs = Importer.import_input('misc/xtrain.txt')
outputs = Importer.import_output('misc/dtrain.txt')

test_inputs = Importer.import_input('misc/xtest.txt')
test_outputs = Importer.import_output('misc/dtest.txt')

exec1_1 = Perceptron(inputs, outputs, is_random=False)
weights, train_outputs = exec1_1.train()

testc.test_outputs("Execução 1_1 Treino", train_outputs, outputs)
testc.test_outputs("Execução 1_1 Teste", exec1_1.classify(test_inputs), test_outputs)

# exec1_2 = Perceptron(inputs, outputs)
# exec1_2.train()


############# 2 #############
# repitir o 1, porém com 𝜂 0.1
############# 3 #############
# repitir o 1, porém com 𝜂 0.01
############# 4 #############
# repitir do 1 ao 3, porém com dados normalizados
############# 5 #############
# indiciar qual foi o mais eficiente pela média de épocas
############# 6 #############