class Neuron:
    def __init__(self):
        self.inputs = []
        self.output = None
        self.activation_function = None
        self.weights = []
        self.__threshold = 0

    def set_threshold(self, value):
        self.threshold = value
