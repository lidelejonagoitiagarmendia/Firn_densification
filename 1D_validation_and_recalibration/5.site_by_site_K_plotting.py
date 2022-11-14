# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:02:15 2022

@author: lidel
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

Ks=np.linspace(100,10000,201)
RMSE=np.load('RMSE_site_by_site.npy')
   

plt.figure(figsize=(10,8))

plt.plot(Ks,RMSE[0,:],label='DYE3')
plt.plot(Ks,RMSE[1,:],label='GRIP')
plt.plot(Ks,RMSE[2,:],label='NEEM')
plt.plot(Ks,RMSE[3,:],label='NGRIP')
plt.plot(Ks,RMSE[4,:],label='SITE2')
plt.plot(Ks,RMSE[5,:],label='SITEA')

plt.xlabel('K')
plt.ylabel('RMSE')
plt.legend(loc='upper right')

plt.show() 

fig, ax1 = plt.subplots(figsize=(12,7))

# The data.
ax1.plot(Ks,RMSE[0,:],c='#0072B2',label='DYE3')
ax1.plot(Ks,RMSE[1,:],c='#56B4E9',label='GRIP')
ax1.plot(Ks,RMSE[2,:],c='#009E73',label='NEEM')
ax1.plot(Ks,RMSE[3,:],c='#E69F00',label='NGRIP')
ax1.plot(Ks,RMSE[4,:],c='#D55E00',label='SITE2')
ax1.plot(Ks,RMSE[5,:],c='#CC79A7',label='SITEA')

ax1.set_xlabel(r'$K$')
ax1.set_ylabel(r'RMSE  $(kg\ m^{-3})$')
ax1.legend(loc='lower right',ncol=3)
ax1.set_ylim([0,90])

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.4,0.67,0.4,0.3])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
pp,p1,p2 =mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
pp.set_fill(True)
linecolor='#c88242'
pp.set_facecolor('#fff1cd')
pp.set_edgecolor('k')
p1.set_edgecolor(linecolor)
p2.set_edgecolor(linecolor)

# The data: only display for low temperature in the inset figure.
Tmax = -31.8
ax2.plot(Ks[Ks<1000], RMSE[0,Ks<1000], c='#0072B2')#, mew=2, alpha=0.8,label='Experiment')
ax2.plot(Ks[Ks<1000], RMSE[1,Ks<1000], c='#56B4E9')#,'x', c='purple', mew=2, alpha=0.8,label='Experiment')
ax2.plot(Ks[Ks<1000], RMSE[2,Ks<1000], c='#009E73')#,'x', c='purple', mew=2, alpha=0.8,label='Experiment')
ax2.plot(Ks[Ks<1000], RMSE[3,Ks<1000], c='#E69F00')#,'x', c='purple', mew=2, alpha=0.8,label='Experiment')
ax2.plot(Ks[Ks<1000], RMSE[4,Ks<1000], c='#D55E00')#,'x', c='purple', mew=2, alpha=0.8,label='Experiment')
ax2.plot(Ks[Ks<1000], RMSE[5,Ks<1000], c='#CC79A7')#,'x', c='purple', mew=2, alpha=0.8,label='Experiment')

# "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"
# Some ad hoc tweaks.
ax2.set_xlim([0,1000])
ax2.set_ylim([5,70])
ax2.set_facecolor('#fff1cd')
# ax2.set_yticks(np.arange(0,2,0.4))
ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
ax2.tick_params(axis='x', which='major', pad=8)
plt.savefig('RMSE_site_by_site.png',dpi=150,format='png')
plt.show()
