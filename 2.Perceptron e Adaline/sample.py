class Sample:
    def __init__(self, inputs: tuple, expected_output):
        self.inputs = inputs
        self.expected_output = expected_output
    
    def __str__(self):
        return self.inputs

    def __repr__(self):
        return str(self.inputs)
