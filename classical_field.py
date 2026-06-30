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
v = np.sqrt(m**2/c)
p= 2.5


def correct_Newton(k, mass, coupling_constant, density, h, density_index):
    
    v = np.sqrt((mass**2)/coupling_constant) #computing the VEV
    x = np.arange(0.01, 100+h, h) #domain chosen in here 
    n = len(x)
    u = np.zeros(n) 
    
    for i in range(n):
        u[i] = v #trial function encoded here 
       
    
    for iter in range(k):
        J = np.zeros((n,n), dtype = float)
        G = np.zeros(n)
        
        G[0] = density*u[0] + (coupling_constant)*(u[0]**3) - (mass**2)*u[0] 
        G[n-1] = (coupling_constant)*(u[n-1]**3) - mass**2*u[n-1] - (1/h**2)*(u[n-2] - 2*u[n-1] + v) - (1/(h*x[n-1]))*(v - u[n-2])
        #The RHS boundary conditions of the nonlinear ODE 
        
        for i in range(1, math.floor(density_index)):
            G[i] = density*u[i] + (coupling_constant)*(u[i]**3) - ((mass**2)*u[i]) - (1/h**2)*(u[i-1] - 2*u[i] + u[i+1]) - (1/(h*x[i]))*(u[i+1]-u[i-1])
            
        for j in range(math.floor(density_index),n-1):
            G[j] = (coupling_constant)*(u[j]**3) - ((mass**2)*u[j]) - (1/h**2)*(u[j-1] - 2*u[j] + u[j+1]) - (1/(h*x[j]))*(u[j+1]-u[j-1])
        #density is encoded here as a switch- on in values of r<R and off otherwise 
        
        J[0,0] = -3/(h**2) + mass**2 - density - (coupling_constant*3)*(u[0]**2) 
        J[0,1] = 3/(h**2)

        
        J[n-1,n-1] = -2/(h**2) + (mass**2) - ((coupling_constant*3)*(u[n-1]**2))
        J[n-1,n-2] = 1/h**2 - 1/(h*x[n-1])
        #boundary values for the Jacobian 
        
        for l in range(1, math.floor(density_index)):
            J[l,l] = -2/(h**2) + (mass**2- density-(coupling_constant*3)*(u[l]**2))
            J[l,l+1] = 1/h**2 +1/(x[l]*h)
            J[l,l-1] = 1/h**2 -1/(x[l]*h)
        
        for k in range(math.floor(density_index), n-1):
            J[k,k] = -2/(h**2) + (mass**2 - (coupling_constant*3)*(u[k]**2))
            J[k,k+1] = 1/h**2 +1/(x[k]*h)
            J[k,k-1] = 1/h**2 -1/(x[k]*h)
        #general values for the Jacobian matrix     
        
        correction = np.linalg.solve(J,G)
        #correction solved using program from numpy
        u = u + correction
        
    return u #corrected solution returned 



x1= np.arange(0.01,100+h1,h1)

u1 = correct_Newton( 5, m, c, p, h1, 5/h1-1)


#calculates residuals of the solution if needed
n = len(u1)
res = np.zeros(n)
res[0] = c*(u1[0]**3) + p*u1[0] - (m**2)*u1[0]
res[n-1] = c*(u1[n-1]**3) - (m**2)*u1[n-1] - (1/h1**2)*(v-2*u1[n-1]+u1[n-2]) - 1/(h1*x1[n-1])*(v-u1[n-2])
for l in range(1, math.floor(5/h1-1)):
    res[l] = c*(u1[l]**3) + p*u1[l] - (m**2)*u1[l] - (1/h1**2)*(u1[l+1]-2*u1[l]+u1[l-1]) - 1/(h1*x1[l])*(u1[l+1]-u1[l-1])
for l in range(math.floor(5/h1-1), n-1):
    res[l] = c*(u1[l]**3) - (m**2)*u1[l] - (1/h1**2)*(u1[l+1]-2*u1[l]+u1[l-1]) - 1/(h1*x1[l])*(u1[l+1]-u1[l-1])













