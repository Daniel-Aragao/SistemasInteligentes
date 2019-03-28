import random
from IO_Operations import Printer
import math
from sklearn import preprocessing
import numpy as np

class Randomize:
    @staticmethod
    def get_random(interval_start=-0.5, interval_end=0.5):
        return random.uniform(interval_start, interval_end)

    @staticmethod
    def get_random_vector(length, seed=0, interval_start=-0.5, interval_end=0.5):
        if seed:
            random.seed(seed)
        
        return [Randomize.get_random(interval_start, interval_end) for i in range(length)]


class Classification:
    @staticmethod
    def get_class_distribution(inputs,outputs):
        A = ([], [])
        B = ([], [])

        for i, inputt in enumerate(inputs):
            if len(inputt) == 2:
                x, y = inputt
            elif len(inputt) == 3:
                teta, x, y = inputt

            C = (B if outputs[i] == -1 else A)

            C[0].append(x)
            C[1].append(y)
        
        return A, B

    @staticmethod
    def test_outputs(name, perceptron_output, test_output, printer=Printer):
        qtd_errors = 0
        fail = False
        for index, p_o in enumerate(perceptron_output):
            fail_local = False

            if p_o != test_output[index]:
                fail = True
                fail_local = True
                qtd_errors += 1

        samples_size = len(perceptron_output)
        hits = (((samples_size - qtd_errors)/samples_size)*100)

        if fail:            
            printer.print_msg(
                " => {" + name + "} deu "+str(qtd_errors)+" erro(s) com "+str(hits)+" de taxa de acerto --------------------------------------")
        else:
            printer.print_msg(
                " => {" + name + "} deu bom                                       ")
        
        return hits
    
    @staticmethod
    def test_regression_outputs(name, results_output, test_output, printer=Printer):
        error= Classification.calc_eqm(results_output, test_output)

        printer.print_msg(
            " => {" + name + "} EQM: "+str(error))

    @staticmethod
    def get_pairs_from_class_distribution(L): 
        return [[L[0][index], L[1][index]] for index in range(len(L[0]))]
        
    @staticmethod
    def change_nearest_points_classes(inputs, outputs):
        A, B = Classification.get_class_distribution(inputs, outputs, no_teta=True)

        A = Classification.get_pairs_from_class_distribution(A)
        B = Classification.get_pairs_from_class_distribution(B)

        dist, a_pair, b_pair = DistanceCalcs.nearest_points(A, B)

        filter(lambda e: e[0] == A[a_pair][0] and e[1] == A[a_pair][1] , inputs)
        
        new_outputs = [i for i in outputs]

        new_outputs[inputs.index(a_pair)] *= -1
        new_outputs[inputs.index(b_pair)] *= -1

        return new_outputs
    
    @staticmethod
    def calc_eqm(results, outputs):
        summ = 0

        for i, output in enumerate(outputs):
            summ += ((results[i] - output) ** 2)

        return summ/len(outputs)


class Normalize:
    @staticmethod
    def reshape(data_points):
        if type(data_points[0]) != type([]):
            return True, [[i] for i in data_points]

        return False, data_points
    
    @staticmethod
    def unshape(data_points):
        return [i[0] for i in data_points]
        
    @staticmethod
    def unscale_data(data_points, scaler):
        reshape, data_points = Normalize.reshape(data_points)

        result = scaler.inverse_transform(data_points)

        if reshape:
            result = Normalize.unshape(result)
        
        return list(result)
    
    @staticmethod
    def min_max_scale_data(data_points, scaler=None, min=-0.5, max=0.5):
        reshape, data_points = Normalize.reshape(data_points)

        if not scaler:
            scaler = preprocessing.MinMaxScaler((min,max)).fit(data_points)
            
        transformed = scaler.transform(data_points)

        if reshape:
            transformed = Normalize.unshape(transformed)

        return list(transformed), scaler

    @staticmethod
    def standard_scale_data(data_points, scaler=None):
        reshape, data_points = Normalize.reshape(data_points)

        if not scaler:
            scaler = preprocessing.StandardScaler().fit(data_points)
            
        transformed = scaler.transform(data_points)

        if reshape:
            transformed = Normalize.unshape(transformed)

        return list(transformed), scaler


class DistanceCalcs:
    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    @staticmethod
    def nearest_points(points_list1, points_list2):
        best_points = (math.inf, None)

        for i, point1 in enumerate(points_list1):
            for j, point2 in enumerate(points_list2):
                dist = DistanceCalcs.euclidean_distance(
                    point1[0], point1[1], point2[0], point2[1])

                if dist < best_points[0]:
                    best_points = (dist, point1, point2)

        return best_points
    