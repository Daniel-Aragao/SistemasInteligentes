class Sample:
    def __init__(self, inputs: tuple, expected_output, weights):
        self.inputs = inputs
        self.expected_output = expected_output
        self.weights = weights

    def get_activation_potential(self):
        activation_potential = 0

        for i, inputt in enumerate(self.inputs):
            activation_potential += self.weights[i] * inputt
        
        return activation_potential
    
    def __str__(self):
        return self.inputs

    def __repr__(self):
        return str(self.inputs)
