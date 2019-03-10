import matplotlib.pyplot as plt


class Ploter:
    @staticmethod
    def plot_results(inputs,outputs):
        A = ([], [])
        B = ([], [])

        for i, inputt in enumerate(inputs):
            teta, x, y = inputt
            C = (A if outputs[i] == 1 else B)

            C[0].append(x)
            C[1].append(y)

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
    def show(title=None):
        if title:
            plt.title(title, fontsize=15)

        plt.legend()
        plt.show()
    
    @staticmethod
    def savefig(title):
        plt.savefig("./output/" + title+ ".png", format="PNG")
        plt.close()