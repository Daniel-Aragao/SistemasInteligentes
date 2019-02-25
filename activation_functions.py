import math

class PartiallyDiff:

    @staticmethod
    def step(u):
        return 0 if u < 0 else 1
    
    @staticmethod
    def step_derivative(u):
        return 0

    @staticmethod
    def signal(u):
        return 0 if u == 0 else (1 if u > 0 else -1)

    @staticmethod
    def signal_derivative(u):
        return 0

    @staticmethod
    def symmetrical_ramp(u, a=2):
        return a if u > a else (-a if u < -a else u)

    @staticmethod
    def symmetrical_ramp_derivative(u, a=2):
        return 0 if u > a else (0 if u < -a else 1)

    @staticmethod
    def rectified_linear_unit(u):
        return u if u > 0 else 0

    @staticmethod
    def rectified_linear_unit_derivative(u):
        return PartiallyDiff.step(u)

    @staticmethod
    def parametric_rectified_linear_unit(u, alpha=0.2):
        if alpha <= 0:
            raise Exception("alpha > 0")
        
        return u if u > 0 else alpha*u
    
    @staticmethod
    def parametric_rectified_linear_unit_derivative(u, alpha=0.2):
        if alpha <= 0:
            raise Exception("alpha > 0")
        
        return 1 if u > 0 else alpha

class FullyDiff:

    @staticmethod
    def logistic(u, beta=0.5):
        if beta <= 0:
            raise Exception("beta > 0")

        return 1/(1 + math.e ** (-beta * u))
    
    @staticmethod
    def logistic_derivative(u):
        return FullyDiff.logistic(u) * (1 - FullyDiff.logistic(u))

    @staticmethod
    def hyperbolic_tangent(u, beta=0.4):
        if beta <= 0:
            raise Exception("beta > 0")
        
        return (1 - math.e ** (-beta * u))/(1 + math.e ** (-beta * u))
    
    @staticmethod
    def hyperbolic_tangent_derivative(u):
        return 1 - (FullyDiff.hyperbolic_tangent(u))**2
    
    @staticmethod
    def gaussian(u, c=1, sigma=2):
        if sigma == 0:
            raise Exception("sigma == 0")
            
        return math.e ** -(((u - c) ** 2)/(2 * (sigma ** 2)))
    
    @staticmethod
    def gaussian_derivative(u, c=1, sigma=2):
        return -(((math.e ** (-(((u - c) ** 2)/(2 * (sigma ** 2))))) * u - c)/(sigma ** 2))
    
    @staticmethod
    def linear(u):
        return u

    @staticmethod
    def linear_derivative(u):
        return 1
    
    @staticmethod
    def exponential_linear_unit(u, alpha=1):
        if alpha <= 0:
            raise Exception("alpha > 0")
        
        return u if u > 0 else alpha * ((math.e ** u) - 1)
    
    staticmethod
    def exponential_linear_unit_derivative(u, alpha=1):
        return 1 if u > 0 else FullyDiff.exponential_linear_unit(u, alpha) + alpha
    

    

