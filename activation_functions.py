import math

class PartiallyDiff:

    @staticmethod
    def hard_limiter(u):
        return 0 if u < 0 else 1

    @staticmethod
    def symmetric_hard_limiter(u):
        return 0 if u == 0 else (1 if u > 0 else -1)

    @staticmethod
    def symmetrical_ramp(u, a):
        return a if u > a else (-a if u < -a else u)

    @staticmethod
    def rectified_linear_unit(u):
        return u if u > 0 else 0

    @staticmethod
    def parametric_rectified_linear_unit(u, alpha):
        if alpha <= 0:
            raise Exception("alpha > 0")
        
        return u if u > 0 else alpha*u

class FullyDiff:

    @staticmethod
    def logistic(u, beta):
        if beta <= 0:
            raise Exception("beta > 0")

        return 1/(1 + math.e ** (-beta * u))

    @staticmethod
    def Hyperbolic_Tangent(u, beta):
        if beta <= 0:
            raise Exception("beta > 0")
        
        return (1 - math.e ** (-beta * u))/(1 + math.e ** (-beta * u))
    
    @staticmethod
    def gaussian(u, c, sigma):
        if sigma != 0:
            raise Exception("sigma > 0")
            
        return math.e ** -(((u - c) ** 2)/(2 * (sigma ** 2)))
    
    @staticmethod
    def linear(u):
        return u
    
    @staticmethod
    def exponential_linear_unit(u, alpha):
        if alpha <= 0:
            raise Exception("alpha > 0")
        
        return u if u > 0 else alpha * ((math.e ** u) - 1)

