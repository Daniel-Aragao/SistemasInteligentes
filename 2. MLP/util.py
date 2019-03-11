import random
from IO_Operations import Printer
import math


class Randomize:

    @staticmethod
    def get_random():
        return random.random()

    @staticmethod
    def get_random_vector(length):
        return [random.random() for i in range(length)]


class Classification:

    @staticmethod
    def get_class_distribution(inputs,outputs, no_teta=False):
        A = ([], [])
        B = ([], [])

        for i, inputt in enumerate(inputs):
            if no_teta:
                x, y = inputt
            else:
                teta, x, y = inputt

            C = (A if outputs[i] == 1 else B)

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

            # print(str(index).zfill(2) + ".", p_o, test_output[index], "Erro" if fail_local else "")
        
        samples_size = len(perceptron_output)
        hits = (((samples_size - qtd_errors)/samples_size)*100)

        if fail:            
            printer.print_msg(
                " => {" + name + "} deu "+str(qtd_errors)+" erro(s) com "+str(hits)+" de taxa de acerto --------------------------------------")
        else:
            printer.print_msg(
                " => {" + name + "} deu bom                                       ")
        
        return hits
        
    def change_nearest_points_classes(inputs, outputs):
        A, B = Classification.get_class_distribution(inputs, outputs, no_teta=True)
        dist, a_index, b_index = DistanceCalcs.nearest_points(A, B)

        new_outputs = [i for i in outputs]

        new_outputs[a_index] *= -1
        new_outputs[b_index] *= -1

        return new_outputs


class Normalize:

    @staticmethod
    def min_max(new_min, new_max, inputs):
        new_inputs = [i for i in inputs]
        all_inputs = [j for i in inputs for j in i]

        old_max = max(all_inputs)
        old_min = min(all_inputs)

        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                new_inputs[i][j] = (inputs[i][j] - old_min) / (old_max - old_min)
                # new_inputs[i][j] = ((inputs[i][j] - old_min) /
                #                 (old_max - old_min)) * (new_max - new_min) + new_min
        
        return new_inputs


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
                    best_points = (dist, i, j)

        return best_points
    