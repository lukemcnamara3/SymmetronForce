#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:49:06 2025

@author: Luke

Revisited Newton's Method using the result obtained by the Frechet derivative/ pertubation theory 
Code calculates the classical symmetron field profile for the 3D spherical, radially dependent case
Assumes regularity at the origin and that the solution tends to the VEV at infinity
"""

import numpy as np
import matplotlib.pyplot as plt
import math

h1 = 0.01

m = 0.5
c = 1
v = np.sqrt(m**2 / c)
p = 2.5


def correct_Newton(k, mass, coupling_constant, density, h, density_index):
    """
    Solve the nonlinear finite-difference equations using Newton's method.

    Parameters:
        k : int
            Number of Newton iterations.
        mass : float
            Mass parameter.
        coupling_constant : float
            Coupling constant.
        density : float
            Density inside the source region.
        h : float
            Grid spacing.
        density_index : int
            Grid index where the density term switches off.

    Returns:
        u : ndarray
            Numerical solution for the field profile.
    """

    # Compute the vacuum expectation value (VEV)
    v = np.sqrt((mass**2) / coupling_constant)

    # Define the radial computational domain
    x = np.arange(0.01, 100 + h, h)
    n = len(x)

    # Initialise the Newton iteration with the constant solution u = v
    u = np.full(n, v)

    for iteration in range(k):

        # Initialise Jacobian matrix and residual vector
        J = np.zeros((n, n), dtype=float)
        G = np.zeros(n)

        # Residual equation at the left boundary
        G[0] = density*u[0] + coupling_constant*u[0]**3 - mass**2*u[0]

        # Residual equation at the right boundary
        G[n-1] = (coupling_constant*u[n-1]**3
                  - mass**2*u[n-1]
                  - (1/h**2)*(u[n-2] - 2*u[n-1] + v)
                  - (1/(h*x[n-1]))*(v - u[n-2]))

        # Residual equations inside the density region (r < R)
        for i in range(1, math.floor(density_index)):
            G[i] = (density*u[i]
                    + coupling_constant*u[i]**3
                    - mass**2*u[i]
                    - (1/h**2)*(u[i-1] - 2*u[i] + u[i+1])
                    - (1/(h*x[i]))*(u[i+1] - u[i-1]))

        # Residual equations outside the density region (r > R)
        for i in range(math.floor(density_index), n-1):
            G[i] = (coupling_constant*u[i]**3
                    - mass**2*u[i]
                    - (1/h**2)*(u[i-1] - 2*u[i] + u[i+1])
                    - (1/(h*x[i]))*(u[i+1] - u[i-1]))

        # Jacobian entries for the left boundary equation
        J[0, 0] = -3/(h**2) + mass**2 - density - 3*coupling_constant*u[0]**2
        J[0, 1] = 3/(h**2)

        # Jacobian entries for the right boundary equation
        J[n-1, n-1] = -2/(h**2) + mass**2 - 3*coupling_constant*u[n-1]**2
        J[n-1, n-2] = 1/h**2 - 1/(h*x[n-1])

        # Jacobian inside the density region
        for i in range(1, math.floor(density_index)):
            J[i, i] = -2/(h**2) + mass**2 - density - 3*coupling_constant*u[i]**2
            J[i, i+1] = 1/h**2 + 1/(x[i]*h)
            J[i, i-1] = 1/h**2 - 1/(x[i]*h)

        # Jacobian outside the density region
        for i in range(math.floor(density_index), n-1):
            J[i, i] = -2/(h**2) + mass**2 - 3*coupling_constant*u[i]**2
            J[i, i+1] = 1/h**2 + 1/(x[i]*h)
            J[i, i-1] = 1/h**2 - 1/(x[i]*h)

        # Solve the Newton system J * correction = G
        correction = np.linalg.solve(J, G)

        # Update the solution
        u = u + correction

    return u


x1 = np.arange(0.01, 100 + h1, h1)

# Solve using Newton's method
u1 = correct_Newton(5, m, c, p, h1, 5/h1 - 1)


# Compute residuals of the final solution
n = len(u1)
res = np.zeros(n)

# Left boundary residual
res[0] = c*u1[0]**3 + p*u1[0] - m**2*u1[0]

# Right boundary residual
res[n-1] = (c*u1[n-1]**3
            - m**2*u1[n-1]
            - (1/h1**2)*(v - 2*u1[n-1] + u1[n-2])
            - (1/(h1*x1[n-1]))*(v - u1[n-2]))

# Residual inside density region
for i in range(1, math.floor(5/h1 - 1)):
    res[i] = (c*u1[i]**3
              + p*u1[i]
              - m**2*u1[i]
              - (1/h1**2)*(u1[i+1] - 2*u1[i] + u1[i-1])
              - (1/(h1*x1[i]))*(u1[i+1] - u1[i-1]))

# Residual outside density region
for i in range(math.floor(5/h1 - 1), n-1):
    res[i] = (c*u1[i]**3
              - m**2*u1[i]
              - (1/h1**2)*(u1[i+1] - 2*u1[i] + u1[i-1])
              - (1/(h1*x1[i]))*(u1[i+1] - u1[i-1]))