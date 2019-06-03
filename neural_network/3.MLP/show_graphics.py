import matplotlib.pyplot as plt
from util import Classification


class Ploter:
    @staticmethod
    def plot_results(inputs, expected_outputs, outputs):
        A_expected, B_expected = Classification.get_class_distribution(inputs,expected_outputs)
        A_resulting, B_resulting = Classification.get_class_distribution(inputs,outputs)

        fig, subplt = plt.subplots()

        subplt.scatter(A_expected[0], A_expected[1], c='blue', label="Classe A (1)")
        subplt.scatter(B_expected[0], B_expected[1], c='red', label="Classe B (-1)")


        def annotate(vector, text):
            for i, x in enumerate(vector[0]):
                y = vector[1][i]
                subplt.annotate(text, (x, y))
        
        annotate(A_resulting, "A")
        annotate(B_resulting, "B")
        # annotate(A_expected, "A")
        # annotate(B_expected, "B")


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