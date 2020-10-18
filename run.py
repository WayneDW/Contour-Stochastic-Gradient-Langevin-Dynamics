#!/usr/bin/env python
from solver import Sampler
import argparse

import autograd.numpy as np
from autograd import grad
from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-T', default=1, type=float, help='inverse temperature')
parser.add_argument('-zeta', default=0.75, type=float, help='Adaptive hyperparameter')
parser.add_argument('-parts', default=100, type=int, help='Total numer of partitions')
parser.add_argument('-decay_lr', default=3e-3, type=float, help='Decay lr')
parser.add_argument('-seed', default=1, type=int, help='seed')
pars = parser.parse_args()

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.random.seed(pars.seed)

split_, total_ = 20, 2e5


def mixture(x): return ((x[0]**2 + x[1]**2)/10 - (cos(1.2*pi*x[0]) + cos(1.2*pi*x[1]))) / 0.3 + ((x[0]**2 + x[1]**2) > 7) * ((x[0]**2 + x[1]**2) - 7)

def mixture_expand(x, y): return mixture([x, y])
def function_plot(x, y): return np.exp(-mixture([x, y]))

boundary_ = 2.5
axis_x = np.linspace(-boundary_, boundary_, 500)
axis_y = np.linspace(-boundary_, boundary_, 500)
axis_X, axis_Y = np.meshgrid(axis_x, axis_y)

energy_grid = mixture_expand(axis_X, axis_Y)
prob_grid = function_plot(axis_X, axis_Y)
lower_bound, upper_bound = np.min(energy_grid) - 1, np.max(energy_grid) + 1
sampler = Sampler(f=mixture, dim=2, boundary=[-boundary_, boundary_], xinit=[2.,2.], partition=[lower_bound, upper_bound], \
                  lr=pars.lr, T=pars.T, zeta=pars.zeta, decay_lr=pars.decay_lr, parts=pars.parts)


warm_up = 5000
sgld_x = np.array([sampler.sgld_beta])
cycsgld_x = np.array([sampler.cycsgld_beta])
resgld_x = np.array([sampler.resgld_beta_low])
csgld_x = np.array([sampler.csgld_beta])
importance_weights = [0.,]
#https://matplotlib.org/3.1.0/gallery/color/named_colors.html
jump_col = ['blue', 'red']


for iters in range(int(total_)):
    sampler.sgld_step()
    if iters % 2 == 0:
        sampler.resgld_step()
    sampler.cycsgld_step(iters=iters, cycles=20, total=total_)
    sampler.csgld_step(iters)
    if iters > warm_up:
        if iters % split_ == 0:
            sgld_x = np.vstack((sgld_x, sampler.sgld_beta))
            resgld_x = np.vstack((resgld_x, sampler.resgld_beta_low))
            csgld_x = np.vstack((csgld_x, sampler.csgld_beta))
            importance_weights.append(sampler.Gcum[sampler.J]**pars.zeta)
        if sampler.r_remainder > 0.5 and iters % (split_ // 2) == 0:
            cycsgld_x = np.vstack((cycsgld_x, sampler.cycsgld_beta))
        if iters % 1000 == 0:
            fig = plt.figure(figsize=(13, 13))
            
            plt.subplot(2, 2, 1).set_title('(a) SGLD', fontsize=16)
            plt.contour(axis_X, axis_Y, prob_grid)
            plt.yticks([-4, -2, 0, 2, 4]) 
            plt.plot(sgld_x[:,0][:-3], sgld_x[:,1][:-3], linewidth=0.1, marker='.', markersize=2, color='k', label="Iteration="+str(iters))
            plt.plot(sgld_x[:,0][-3:], sgld_x[:,1][-3:], linewidth=0.3, marker='.', markersize=10, color=jump_col[0], alpha=1);
            plt.legend(loc="upper left", prop={'size': 13})
            
            
            plt.subplot(2, 2, 2).set_title('(b) reSGLD', fontsize=16)
            plt.contour(axis_X, axis_Y, prob_grid)
            plt.yticks([-4, -2, 0, 2, 4]) 
            plt.plot(resgld_x[:,0][:-3], resgld_x[:,1][:-3], linewidth=0.1, marker='.', markersize=2, color='k', label="Iteration="+str(iters//2))
            plt.plot(resgld_x[:,0][-3:], resgld_x[:,1][-3:], linewidth=0.3, marker='.', markersize=10, color=jump_col[0], alpha=1, label="No. of swaps =" + str(sampler.swaps));
            plt.legend(loc="upper left", prop={'size': 13})
            
            
            plt.subplot(2, 2, 3).set_title('(c) cycSGLD', fontsize=16)
            plt.contour(axis_X, axis_Y, prob_grid)
            plt.yticks([-4, -2, 0, 2, 4]) 
            plt.plot(cycsgld_x[:,0][:-3], cycsgld_x[:,1][:-3], linewidth=0.1, marker='.', markersize=2, color='k', label="Iteration="+str(iters))
            if sampler.r_remainder < 0.5:
                plt.plot(cycsgld_x[:,0][-3:], cycsgld_x[:,1][-3:], linewidth=0.3, marker='.', markersize=10, color=jump_col[0], alpha=1, label=r'Exploration $\beta$='+str(sampler.r_remainder));
            else:
                plt.plot(cycsgld_x[:,0][-3:], cycsgld_x[:,1][-3:], linewidth=0.3, marker='.', markersize=10, color=jump_col[0], alpha=1, label=r'Sampling $\beta$='+str(sampler.r_remainder));
            plt.legend(loc="upper left", prop={'size': 13})
                
            
            plt.subplot(2, 2, 4).set_title('(d) CSGLD (samples from the flattened density)', fontsize=16)
            plt.contour(axis_X, axis_Y, prob_grid)
            plt.yticks([-4, -2, 0, 2, 4]) 
            plt.plot(csgld_x[:,0][:-5], csgld_x[:,1][:-5], linewidth=0.1, marker='.', markersize=2, color='k', label="Iteration="+str(iters))
            col_type = 0 if sampler.grad_mul > 0 else 1
            plt.plot(csgld_x[:,0][-4], csgld_x[:,1][-4], linewidth=0.15, marker='.', markersize=4, color=jump_col[col_type], alpha=1, label="Bouncy moves="+str(sampler.bouncy_move));
            plt.plot(csgld_x[:,0][-3], csgld_x[:,1][-3], linewidth=0.2, marker='.', markersize=6, color=jump_col[col_type], alpha=1, label="Grad multiplier="+str(sampler.grad_mul)[:4]);
            if sampler.Gcum[sampler.J]**pars.zeta < 1e-4:
                plt.plot(csgld_x[:,0][-2], csgld_x[:,1][-2], linewidth=0.25, marker='.', markersize=8, color=jump_col[col_type], alpha=1, label="Importance weight=0");
            else:
                plt.plot(csgld_x[:,0][-1], csgld_x[:,1][-1], linewidth=0.3, marker='.', markersize=10, color=jump_col[col_type], alpha=1, label="Importance weight="+str(sampler.Gcum[sampler.J]**pars.zeta)[:4]); 
            plt.legend(loc="upper left", prop={'size': 10})
            
            plt.tight_layout()
            plt.show()
            


# resample the parameters by considering the importance weights
scaled_importance_weights = importance_weights / np.mean(importance_weights)

resample_x = np.empty((0,2))
for i in range(len(csgld_x)):
    while scaled_importance_weights[i] > 1:
        tag = np.random.binomial(1, p=min(1, scaled_importance_weights[i]))
        scaled_importance_weights[i] -= 1
        if tag == 1:
            resample_x = np.vstack((resample_x, csgld_x[i,]))


split_ = 1


fig = plt.figure(figsize=(17, 11.6))
plt.subplot(2, 3, 1).set_title('(a) Ground truth')
sns.heatmap(prob_grid, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)


plt.subplot(2, 3, 2).set_title('(b) SGLD')
ax = sns.kdeplot(sgld_x[:,0][::split_], sgld_x[:,1][::split_], bw_adjust=10, cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlim(-boundary_, boundary_)
ax.set_ylim(-boundary_, boundary_)

plt.subplot(2, 3, 3).set_title('(c) cycic SGLD (cycSGLD)')
ax = sns.kdeplot(cycsgld_x[:,0][::split_], cycsgld_x[:,1][::split_], bw_adjust=10, cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlim(-boundary_, boundary_)
ax.set_ylim(-boundary_, boundary_)

plt.subplot(2, 3, 4).set_title('(d) Replica exchange SGLD (reSGLD)')
ax = sns.kdeplot(resgld_x[:,0][::split_], resgld_x[:,1][::split_], bw_adjust=10, cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlim(-boundary_, boundary_)
ax.set_ylim(-boundary_, boundary_)


warm_sample = 50
plt.subplot(2, 3, 5).set_title('(e) CSGLD (before resampling)')
ax = sns.kdeplot(csgld_x[:,0][::split_][warm_sample:], csgld_x[:,1][::split_][warm_sample:], bw_adjust=10, cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlim(-boundary_, boundary_)
ax.set_ylim(-boundary_, boundary_)

plt.subplot(2, 3, 6).set_title('(f) CSGLD (after resampling)')
ax = sns.kdeplot(resample_x[:,0][::split_][warm_sample:], resample_x[:,1][::split_][warm_sample:], bw_adjust=10, cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlim(-boundary_, boundary_)
ax.set_ylim(-boundary_, boundary_)



fig = ax.get_figure()
