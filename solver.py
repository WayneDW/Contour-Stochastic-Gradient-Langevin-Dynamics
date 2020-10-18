#!/usr/bin/python
import sys

import autograd.numpy as np
from autograd import grad
from autograd.numpy import sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform

from time import time



from scipy.stats import multivariate_normal

class Sampler:
    def __init__(self, f=None, dim=None, boundary=None, xinit=None, partition=None, lr=0.1, T=1.0, zeta=1, decay_lr=100., parts=100):

        self.f = f

        self.dim = dim
        self.lr = lr
        self.T = T
        self.partition = partition
        self.boundary = boundary
        self.xinit = np.array(xinit)
        self.zeta = zeta
        self.decay_lr = decay_lr
        self.parts = parts
        
        
        # baseline SGLD
        self.sgld_beta = self.xinit
        
        # baseline replica exchange SGLD
        self.resgld_beta_high = self.xinit
        self.resgld_beta_low = self.xinit
        self.swaps = 0
        
        # baseline cyclic SGLD
        self.cycsgld_beta = self.xinit
        self.r_remainder = 0
        
        # initialization for CSGLD
        self.csgld_beta = self.xinit
        self.Gcum = np.array(range(self.parts, 0, -1)) * 1.0 / sum(range(self.parts, 0, -1))
        self.div_f = (self.partition[1] - self.partition[0]) / self.parts
        self.J = self.parts - 1
        self.bouncy_move = 0
        self.grad_mul = 1.

    
        
    def in_domain(self, beta): return sum(map(lambda i: beta[i] < self.boundary[0] or beta[i] > self.boundary[1], range(self.dim))) == 0

    def stochastic_grad(self, beta): 
        return grad(self.f)(beta) + 0.32*normal(size=self.dim)
    

    def stochastic_f(self, beta): return self.f(beta.tolist()) + 0.32*normal(size=1)

    def sgld_step(self):
        proposal = self.sgld_beta - self.lr * self.stochastic_grad(self.sgld_beta) + sqrt(2 * self.lr * self.T) * normal(size=self.dim)
        if self.in_domain(proposal):
            self.sgld_beta = proposal  
    
    
    def cycsgld_step(self, iters=1, cycles=10, total=1e6):
        sub_total = total / cycles
        self.r_remainder = (iters % sub_total) * 1.0 / sub_total
        cyc_lr = self.lr * 5 / 2 * (cos(pi * self.r_remainder) + 1)
        proposal = self.cycsgld_beta - cyc_lr * self.stochastic_grad(self.cycsgld_beta) + sqrt(2 * cyc_lr * self.T) * normal(size=self.dim)
        if self.in_domain(proposal):
            self.cycsgld_beta = proposal 
            
    
    def resgld_step(self, T_multiply=3, var=0.1):
        proposal_low = self.resgld_beta_low - self.lr * self.stochastic_grad(self.resgld_beta_low) + sqrt(2 * self.lr * self.T) * normal(size=self.dim)
        if self.in_domain(proposal_low):
            self.resgld_beta_low = proposal_low 
        
        proposal_high = self.resgld_beta_high - self.lr * self.stochastic_grad(self.resgld_beta_high) + sqrt(2 * self.lr * self.T * T_multiply) * normal(size=self.dim)
        if self.in_domain(proposal_high):
            self.resgld_beta_high = proposal_high
        
        dT = 1 / self.T - 1 / (self.T * T_multiply)
        swap_rate = np.exp(dT * (self.stochastic_f(self.resgld_beta_low) - self.stochastic_f(self.resgld_beta_high)- dT * var))
        intensity_r = 0.1
        if np.random.uniform(0, 1) < intensity_r * swap_rate:
            self.resgld_beta_high, self.resgld_beta_low = self.resgld_beta_low, self.resgld_beta_high
            self.swaps += 1
        
        
    

    def find_idx(self, beta): return(min(max(int((self.stochastic_f(beta) - self.partition[0]) / self.div_f + 1), 1), self.parts - 1))
    def csgld_step(self, iters):        
        self.grad_mul = 1 + self.zeta * self.T * (np.log(self.Gcum[self.J]) - np.log(self.Gcum[self.J-1])) / self.div_f
        proposal = self.csgld_beta - self.lr * self.grad_mul * self.stochastic_grad(self.csgld_beta) + sqrt(2. * self.lr * self.T) * normal(size=self.dim)
        if self.in_domain(proposal):
            self.csgld_beta = proposal

        self.J = self.find_idx(self.csgld_beta)
        
        step_size = min(self.decay_lr, 10./(iters**0.8+100))
        self.Gcum[:self.J] = self.Gcum[:self.J] + step_size * self.Gcum[self.J]**self.zeta * (-self.Gcum[:self.J])
        self.Gcum[self.J] = self.Gcum[self.J] + step_size * self.Gcum[self.J]**self.zeta * (1 - self.Gcum[self.J])
        self.Gcum[(self.J+1):] = self.Gcum[(self.J+1):] + step_size * self.Gcum[self.J]**self.zeta * (-self.Gcum[(self.J+1):])

        if self.grad_mul < 0:
            self.bouncy_move = self.bouncy_move + 1