import matplotlib.pyplot as plt
from util import Classification


class Ploter:
    @staticmethod
    def plot_results(inputs,outputs):
        A, B = Classification.get_class_distribution(inputs,outputs)

        plt.scatter(A[0], A[1], c='blue', label="Classe A (1)")
        plt.scatter(B[0], B[1], c='red', label="Classe B (-1)")

    @staticmethod
    def plot_line(inputs, weights):
        xMax = max([i[1] for i in inputs])
        xMin = min([i[1] for i in inputs])

        x = []

        plane = []

        for inputt in inputs:
            x.append(inputt[1])
            # -((w1*x1 - teta)/w2) = x2
            plane.append(-(inputt[1] * weights[1] - weights[0])/weights[2])
        
        plt.plot(x, plane, c="green", label="Hiper plano")
    
    @staticmethod
    def plot_eqm_epoch(epochs_eqm):
        x = []
        y = []

        for i in epochs_eqm:
            x.append(i[0])
            y.append(i[1])

        plt.plot(x, y)

    @staticmethod
    def show(title=None):
        if title:
            plt.title(title, fontsize=15)

        plt.legend()
        plt.show()
    
    @staticmethod
    def savefig(title):
        plt.savefig("./output/img/" + title+ ".png", format="PNG")
        plt.close()