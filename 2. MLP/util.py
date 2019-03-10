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


class TestClassification:

    @staticmethod
    def test_outputs(name, perceptron_output, test_output, printer=Printer):

        fail = False
        for index, p_o in enumerate(perceptron_output):
            fail_local = False

            if p_o != test_output[index]:
                fail = True
                fail_local = True

            # print(str(index).zfill(2) + ".", p_o, test_output[index], "Erro" if fail_local else "")

        if fail:
            print(" => {" + name + "} deu erro --------------------------------------")
        else:
            print(" => {" + name + "} deu bom                                       ")

class Normalize:
    
    @staticmethod
    def min_max(x, inputs):
        xMax = max([i for i in inputs])
        xMin = min([i for i in inputs])

        return (x - xMin)/(xMax - xMin)

# class Distance:

#     @staticmethod
#     def euclidean_distance(x1,y1, x2, y2):
#         return math.sqrt((x1-x2)**2 + (y1-y2)**2)
    
#     @staticmethod
#     def nearest_points(points_list1, points_list2):
#         best_points = ()

#         for point1 in points_list1:
#             for point2 in points_list2
