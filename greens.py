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
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('classical_field_profile.csv')
df["r"] = df["r"].round(3)
df["u"] = df["u"].round(6)  
df["residual"] = abs(df["residual"].round(6))
df.to_csv("classical_field_profile.csv", index=False)

CField = np.array(df["u"])
R = 5
h = 0.01
p = 0.1
m = 1
lamb = 1
x = np.arange(0.01, 100 + h, h)

def g_func(r):
    u = np.interp(r, x, CField)
    return m**2 - p*np.heaviside(R-r, 1) - 3*lamb*u**2


def F(w1, w2, l, r):

    g = g_func(r)

    if not np.isfinite(w2):
        print("w2 became non-finite at r =", r)
        raise RuntimeError("w2 exploded")

    if abs(w2) > 1e6:
        print("w2 becoming huge at r =", r, "w2 =", w2)

    f1 = w2
    f2 = l*(l+1)/(r**2) - g - w2 **2 

    return f1, f2 

def rk4(w1, w2, l, h, x):

    k11, k12 = F(w1, w2, l, x)
    k21, k22 = F(w1 + 0.5 * h * k11, w2 + 0.5 * h * k12, l, x + 0.5 * h)
    k31, k32 = F(w1 + 0.5 * h * k21, w2 + 0.5 * h * k22, l, x + 0.5 * h)
    k41, k42 = F(w1 + h * k31, w2 + h * k32, l, x + h)

    w1_next = w1 + (h / 6) * (k11 + 2 * k21 + 2 * k31 + k41)
    w2_next = w2 + (h / 6) * (k12 + 2 * k22 + 2 * k32 + k42)

    return w1_next, w2_next

def ComputeFunc(l, BCs, choice):
    
    n = len(x)
    w1 = np.zeros(n)
    w2 = np.zeros(n)

    if choice == "inner":

        w1[0] = BCs[0]
        w2[0] = BCs[1]

        for i in range(n-1):
            w1[i+1], w2[i+1] = rk4(w1[i], w2[i], l, h, x[i])
        
        return w1, w2
    
    if choice == "outer":

        w1[-1] = BCs[0]
        w2[-1] = BCs[1]

        for i in range(n-1, 0, -1):
            w1[i-1], w2[i-1] = rk4(w1[i], w2[i], l, -h, x[i])
        
        return w1, w2
    
def BoundaryConditions(l, x_min, x_max):

    BCs_inner = [(l + 1) * np.log(x_min), (l + 1) / x_min]
    BCs_outer = [-np.sqrt(2) * m * x_max + np.log(x_max), -np.sqrt(2) * m + 1 / x_max]

    return BCs_inner, BCs_outer

def ComputeGr(l, x_min, x_max):

    BCs_inner, BCs_outer = BoundaryConditions(l, x_min, x_max)

    w1_inner, w2_inner = ComputeFunc(l, BCs_inner, "inner")
    w1_outer, w2_outer = ComputeFunc(l, BCs_outer, "outer")

    n = len(CField)

    f1_inner = np.zeros(n)
    f2_inner = np.zeros(n)
    f1_outer = np.zeros(n)
    f2_outer = np.zeros(n)

    Wronskian = np.zeros(n)

    print(np.max(w1_inner))
    print(np.min(w1_inner))
    print(np.max(w1_outer))
    print(np.min(w1_outer))


    for i in range(n):

        f1_inner[i] = np.exp(w1_inner[i])/x[i]
        f1_outer[i] = np.exp(w1_outer[i])/x[i]

        f2_inner[i] = f1_inner[i] * (w2_inner[i] - 1)/x[i]
        f2_outer[i] = f1_outer[i] * (w2_outer[i] - 1)/x[i]

        Wronskian[i] = (f1_inner[i] * f2_outer[i] - f1_outer[i] * f2_inner[i])/(x[i]**2)

    return f1_inner*f1_outer/Wronskian

Gr0 = ComputeGr(0, 0.01, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, Gr0)
plt.show()

#this method proved unsuccessful









