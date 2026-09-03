"""
Attempting to build a Green's function method on python for quantum corrections to the classical field.

Green's equation is as follows:
w''(r) + (w'(r))^2 + g - l * (l+1) = 0 

Inner condition: 
w(x_min) = (l + 1) * ln(x_min)
w'(x_min) = (l + 1) / x_min

Outer condition: 
w(x_max) = - (2^0.5) * m * x_max + ln(x_max)
w'(x_max) = - (2^0.5) * m + 1 / x_max

"""
import numpy as np

def F(x1, x2, l, r, g):

    return x2, l*(l+1) - g - x2**2

h = 0.01
m = 0.5
c = 1
v = np.sqrt(m**2 / c)
p = 2.5
R = 5

