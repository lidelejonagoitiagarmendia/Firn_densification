# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:28:55 2022

@author: lidel
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib import colors
import statsmodels.api as sm
from scipy.stats import linregress
from math import pi

deg2rad= (2*pi)/360

sites=['dye3','grip','neem','ngrip','site_2','siteA_crete']
Ks=np.linspace(100,10000,201)

#Load and re-structure Aglen data
Agrip=np.load('Aglens_fit_grip_(100-10000).npy')
Angrip=np.load('Aglens_fit_ngrip_(100-10000).npy')
Aneem=np.load('Aglens_fit_neem_(100-10000).npy')
Adye3=np.load('Aglens_fit_dye3_(100-10000).npy')
Asite2=np.load('Aglens_fit_site_2_(100-10000).npy')
AsiteA=np.load('Aglens_fit_siteA_crete_(100-10000).npy')

As=np.zeros((np.shape(Agrip)[1],6))
As[:,0]=Agrip[0,:]
As[:,1]=Angrip[0,:]
As[:,2]=Aneem[0,:]
As[:,3]=Adye3[0,:]
As[:,4]=Asite2[0,:]
As[:,5]=AsiteA[0,:]

Tmeas=np.array([-21,-31.7,-28.8,-31.5,-25,-29.5])

Tobs=np.zeros((np.shape(Agrip)[1],6))
Tobs[:,0]+=Tmeas[1]
Tobs[:,1]+=Tmeas[3]
Tobs[:,2]+=Tmeas[2]
Tobs[:,3]+=Tmeas[0]
Tobs[:,4]+=Tmeas[4]
Tobs[:,5]+=Tmeas[5]

A0,Q,R = 3.985e-13, 60e3, 8.314 # Canonical values, same as Zwinger et al. (2007)
T0 = 273.15

Tinv=np.zeros((np.shape(Agrip)[1],6))
Tinv=-Q/(R*np.log(As/A0)) -T0 # temperature from flow-rate factor


#Scatter plot of inferred and observed temperatures with colomap behind

fig,ax=plt.subplots(figsize=(8,8))
ax.set_aspect('equal')

#Plot colormap of the difference between the inferred and observed temperatures
x = y = np.linspace(-37, -20, 200)
X, Y = np.meshgrid(x, y)
z = np.array([abs(i-j) for j in y for i in x])
Z = z.reshape(200, 200)
levels=[0,2,4,6,8,10,12,14,16,18]
cp = plt.contourf(X, Y, Z,levels=levels,cmap='binary')
cbar=plt.colorbar(cp,orientation='horizontal')
cbar.set_label(r'$|T_{\rm{inferred}}-T_{\rm{observed}}|$ $(^\circ C)$',fontsize=12.5)

for i in range(len(Ks)):
    
    if i%2==0:

        k=Ks[i]
        ax.scatter(Tinv[i,:],Tobs[i,:],label='k='+str(k),c=cm.rainbow(k/Ks.max()))

norm=colors.Normalize(vmin=0, vmax=Ks.max())
fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'),label='K')
plt.xlim([-37,-20])
plt.ylim([-37,-20])
ax.set_xlabel(r'$T_{\rm{inferred}}$ $(^\circ C)$')
ax.set_ylabel(r'$T_{\rm{observed}}$ $(^\circ C)$')
plt.savefig('temperatures_scatter.png',dpi=150,format='png')
plt.show()

#Perform linear regression for each set of six points for each K

slope=np.zeros(len(Tinv))
std_slope=np.zeros(len(Tinv))
slope_deg=np.zeros(len(Tinv))
std_slope_deg=np.zeros(len(Tinv))
intercept=np.zeros(len(Tinv))
std_intercept=np.zeros(len(Tinv))
r2=np.zeros(len(Tinv))

for i in range(len(Ks)):
    
    x=Tinv[i,:]
    y=Tobs[i,:]
    
    res = linregress(x, y)
    
    slope[i]=res.slope
    slope_deg[i]=np.arctan(res.slope) /deg2rad #Get slope in degrees
    intercept[i]=res.intercept
    
    std_slope[i]=res.stderr
    std_slope_deg[i]=np.arctan(res.stderr) /deg2rad
    std_intercept[i]=res.intercept_stderr
    
    r2[i]=res.rvalue**2


plt.figure(figsize=(12,3))
plt.axhline(45,lw=0.5,c='k')
plt.errorbar(Ks[::2], slope_deg[::2], yerr = std_slope_deg[::2],fmt='o',ms=2.5,mec='blue',ecolor = 'mediumslateblue',color='black')
plt.xlabel(r'$K$')
plt.ylabel(r'Slope $(^\circ)$')
plt.ylim([20,70])
plt.tight_layout()
plt.savefig('slopes_regressions.png',dpi=150,format='png')
plt.show()

plt.figure(figsize=(12,3))
plt.axhline(0,lw=0.5,c='k')
plt.errorbar(Ks[::2], intercept[::2], yerr = std_intercept[::2], fmt='o',ms=2.5,mec='red',ecolor = 'tomato',color='black')
plt.xlabel(r'$K$')
plt.ylabel(r'Intercept $(^\circ C)$')
plt.ylim([-15,15])
plt.tight_layout()
plt.savefig('intercepts_linear_regressions.png',dpi=150,format='png')
plt.show()

plt.figure(figsize=(12,3))
plt.scatter(Ks,r2,s=4,c='limegreen')
plt.xlabel(r'$K$')
plt.ylabel(r'$R^2$')
plt.ylim([0.75,1])
plt.tight_layout()
plt.savefig('r2_linear_regressions.png',dpi=150,format='png')
plt.show()


fig,ax=plt.subplots(figsize=(8,6))
ax.set_aspect('equal')

xplot=np.linspace(-50,-10,100)

ax.plot(np.linspace(-50,-10,100),np.linspace(-50,-10,100),'k--',linewidth=1)

for i in range(len(Ks)):
    
    if i%2==0:

        k=Ks[i]
        m=slope[i]
        y0=intercept[i]
        
        yplot= y0 + m*xplot
        
        ax.plot(xplot,yplot,label='k='+str(k),c=cm.rainbow(k/Ks.max()))
        
ax.set_xlim([-37,-20])
ax.set_ylim([-37,-20])
ax.set_xlabel(r'$T_{\rm{inferred}}$ $(^\circ C)$')
ax.set_ylabel(r'$T_{\rm{observed}}$ $(^\circ C)$')

norm=colors.Normalize(vmin=0, vmax=Ks.max())
fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'),label='K')
 
plt.savefig('linear_regressions.png',dpi=150,format='png')
plt.show()