class Neuron:
    def __init__(self, threshold: float, inputs: list, weights: list, activation_function):
        self.inputs = inputs
        # self.output = None
        self.activation_function = activation_function
        self.weights = weights
        self.__threshold = threshold
    
    def __param_validation(self):
        import types

        if self.inputs is None or type(self.inputs) != type([]):
            raise Exception("Inputs can't be None and must be a function")
        
        if self.weights is None or type(self.weights) != type([]):
            raise Exception("Weights can't be None and must be a function")

        if len(self.inputs) != len (self.weights):
            raise Exception("Inputs and Weights arrays must have the same size")

        if self.__threshold is None:
            raise Exception("Threshold must not be None")
        
        if self.activation_function is None or not isinstance(self.activation_function, types.FunctionType):
            raise Exception("activation function can't be None and must be a function")

    def output(self):
        self.__param_validation()

        summm = 0

        for index, input_signal in enumerate(self.inputs):
            summm += input_signal * self.weights[index]
        
        activation_potential = summm - self.__threshold

        return self.activation_function(activation_potential)

    def __str__(self):
        string = "\nThreshold: " + str(self.__threshold) + " "
        string += "Inputs: " + str(self.inputs) + " "
        string += "Weight: " + str(self.weights) + " "
        string += "Activation Function method: " + str(self.activation_function)
        return string