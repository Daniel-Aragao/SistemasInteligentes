import math

class ActivationFunctions:
# class PartiallyDiff:

    @staticmethod
    def apply_function(func, data, is_derivative=False):
        return [func(i, is_derivative=is_derivative) for i in data]

    @staticmethod
    def step(u, is_derivative=False):
        if not is_derivative:
            return 0 if u < 0 else 1
        else:
            return 0

    @staticmethod
    def signal(u, is_derivative=False):
        if not is_derivative:
            return 0 if u == 0 else (1 if u > 0 else -1)
        else:
            return 0

    @staticmethod
    def symmetrical_ramp(u, a=2, is_derivative=False):
        if not is_derivative:
            return a if u > a else (-a if u < -a else u)
        else:
            return 0 if u > a else (0 if u < -a else 1)

    @staticmethod
    def rectified_linear_unit(u, is_derivative=False):
        if not is_derivative:
            return u if u > 0 else 0
        else:
            return ActivationFunctions.step(u)

    @staticmethod
    def parametric_rectified_linear_unit(u, alpha=0.2, is_derivative=False):
        if alpha <= 0:
            raise Exception("alpha > 0")
        
        if not is_derivative:
            return u if u > 0 else alpha*u
        else:
            return 1 if u > 0 else alpha
    

# class ActivationFunctions.

    @staticmethod
    def logistic(u, beta=0.5, is_derivative=False):
        if beta <= 0:
            raise Exception("beta > 0")

        if not is_derivative:
            return 1/(1 + math.e ** (-beta * u))
        else:
            return ActivationFunctions.logistic(u) * (1 - ActivationFunctions.logistic(u))

    @staticmethod
    def hyperbolic_tangent(u, beta=0.4, is_derivative=False):
        if beta <= 0:
            raise Exception("beta > 0")
        
        if not is_derivative:
            return (1 - math.e ** (-beta * u))/(1 + math.e ** (-beta * u))
        else:
            return 1 - (ActivationFunctions.hyperbolic_tangent(u))**2
    
    @staticmethod
    def gaussian(u, c=1, sigma=2, is_derivative=False):
        if sigma == 0:
            raise Exception("sigma == 0")
        
        if not is_derivative:
            return math.e ** -(((u - c) ** 2)/(2 * (sigma ** 2)))
        else:
            return -(((math.e ** (-(((u - c) ** 2)/(2 * (sigma ** 2))))) * u - c)/(sigma ** 2))
    
    @staticmethod
    def linear(u, is_derivative=False):
        if not is_derivative:
            return u
        else:
            return 1
    
    @staticmethod
    def exponential_linear_unit(u, alpha=1, is_derivative=False):
        if alpha <= 0:
            raise Exception("alpha > 0")
        
        if not is_derivative:
            return u if u > 0 else alpha * ((math.e ** u) - 1)
        else:
            return 1 if u > 0 else ActivationFunctions.exponential_linear_unit(u, alpha) + alpha

# class OthersFunctions:
    @staticmethod
    def sinusoid(u, is_derivative=False):
        if not is_derivative:
            return math.sin(u)
        else:
            return math.cos(u)

    @staticmethod
    def arc_tangente(u, is_derivative=False):
        if not is_derivative:
            return math.atan(u)
        else:
            return 1/((u ** 2) + 1)
    
    @staticmethod
    def soft_plus(u, is_derivative=False):
        if not is_derivative:
            return math.log(1 + (math.e ** u))
        else:
            return 1 / (1 + (math.e ** (-u)))

    @staticmethod
    def soft_sign(u, is_derivative=False):
        if not is_derivative:
            return u/(1 + math.fabs(u))
        else:
            return math.inf if u == 0 else -(u/(math.fabs(u) * ((1 + math.fabs(u)) ** 2)))
            # return 1/((1 + math.fabs(u)) ** 2)
    
    @staticmethod
    def square_nonlinearity(u, b=2, is_derivative=False):
        if not is_derivative:
            return 1 if u > b else (u - (u ** 2)/4 if u >= 0 else ((u + (u ** 2)/4) if u >= -b else -1))
        else:
            return 0 if u > b else (1 - u/2 if u >= 0 else ((1 + (u)/2) if u >= -b else 0))
            # return 1 - math.fabs(u/2)