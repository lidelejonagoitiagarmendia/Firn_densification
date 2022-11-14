import numpy as np
import matplotlib.pyplot as plt

#Define values of the parameters and variables
sites=['dye3','grip','neem','ngrip','site_2','siteA_crete']
Tmeas=np.array([-21,-31.7,-28.8,-31.5,-25,-29.5])
ks=[10,100,1000,10000]

#Calculate the inverted temperatures from the best-fit Aglen
allAs=np.load('Aglens_inv_raw.npy')
A0,Q,R = 3.985e-13, 60e3, 8.314 # Canonical values, same as Zwinger et al. (2007)
T0 = 273.15
Tinv=-Q/(R*np.log(allAs/A0)) -T0 # temperature from flow-rate factor

#Plot data
fig,ax=plt.subplots(figsize=(8,6))

#Plot colormap of the difference between the inferred and observed temperatures
x = y = np.linspace(-37, -20, 200)
X, Y = np.meshgrid(x, y)
z = np.array([abs(i-j) for j in y for i in x])
Z = z.reshape(200, 200)
levels=[0,2,4,6,8,10,12,14,16,18]
cp = plt.contourf(X, Y, Z,levels=levels,cmap='binary')
cbar=plt.colorbar(cp)
cbar.set_label(r'$|T_{\rm{inferred}}-T_{\rm{observed}}|$ $(^\circ C)$',fontsize=12.5)

#Scatter plot of the temperatures
ax.scatter(Tinv[0,0],Tmeas[0],marker='o',color='mediumblue',s=50)#,label=r'DYE-3, k=$10^1$')
ax.scatter(Tinv[0,1],Tmeas[0],marker='o',color='forestgreen',s=50)#,label=r'DYE-3, k=$10^2$')
ax.scatter(Tinv[0,2],Tmeas[0],marker='o',color='red',s=50)#,label=r'DYE-3, k=$10^3$')
ax.scatter(Tinv[0,3],Tmeas[0],marker='o',color='orange',s=50)#,label=r'DYE-3, k=$10^4$')

ax.scatter(Tinv[1,0],Tmeas[1],marker='s',s=50,color='mediumblue')#,label=r'GRIP, k=$10^1$')
ax.scatter(Tinv[1,1],Tmeas[1],marker='s',s=50,color='forestgreen')#,label=r'GRIP, k=$10^2$')
ax.scatter(Tinv[1,2],Tmeas[1],marker='s',s=50,color='red')#,label=r'GRIP, k=$10^3$')
ax.scatter(Tinv[1,3],Tmeas[1],marker='s',s=50,color='orange')#,label=r'GRIP, k=$10^4$')

ax.scatter(Tinv[2,0],Tmeas[2],marker='^',s=90,color='mediumblue')#,label=r'NEEM, k=$10^1$')
ax.scatter(Tinv[2,1],Tmeas[2],marker='^',s=90,color='forestgreen')#,label=r'NEEM, k=$10^2$')
ax.scatter(Tinv[2,2],Tmeas[2],marker='^',s=90,color='red')#,label=r'NEEM, k=$10^3$')
ax.scatter(Tinv[2,3],Tmeas[2],marker='^',s=90,color='orange')#,label=r'NEEM, k=$10^4$')

ax.scatter(Tinv[3,0],Tmeas[3],marker='X',s=90,color='mediumblue')#,label=r'NGRIP, k=$10^1$')
ax.scatter(Tinv[3,1],Tmeas[3],marker='X',s=90,color='forestgreen')#,label=r'NGRIP, k=$10^2$')
ax.scatter(Tinv[3,2],Tmeas[3],marker='X',s=90,color='red')#,label=r'NGRIP, k=$10^3$')
ax.scatter(Tinv[3,3],Tmeas[3],marker='X',s=90,color='orange')#,label=r'NGRIP, k=$10^4$')

ax.scatter(Tinv[4,0],Tmeas[4],marker='d',s=90,color='mediumblue')#,label=r'SITE-2, k=$10^1$')
ax.scatter(Tinv[4,1],Tmeas[4],marker='d',s=90,color='forestgreen')#,label=r'SITE-2, k=$10^2$')
ax.scatter(Tinv[4,2],Tmeas[4],marker='d',s=90,color='red')#,label=r'SITE-2, k=$10^3$')
ax.scatter(Tinv[4,3],Tmeas[4],marker='d',s=90,color='orange')#,label=r'SITE-2, k=$10^4$')

ax.scatter(Tinv[5,0],Tmeas[5],marker='*',s=110,color='mediumblue')#,label=r'SITE-A, k=$10^1$')
ax.scatter(Tinv[5,1],Tmeas[5],marker='*',s=110,color='forestgreen')#,label=r'SITE-A, k=$10^2$')
ax.scatter(Tinv[5,2],Tmeas[5],marker='*',s=110,color='red')#,label=r'SITE-A, k=$10^3$')
ax.scatter(Tinv[5,3],Tmeas[5],marker='*',s=110,color='orange')#,label=r'SITE-A, k=$10^4$')

#These are dummy variables for the legend
ax.scatter(10,10,marker='d',color='k',alpha=0.7,s=90,label='SITE-2')
ax.scatter(10,10,marker='*',color='k',alpha=0.7,s=90,label='SITE-A')
ax.scatter(10,10,marker='o',color='k',alpha=0.7,s=50,label='DYE-3')
ax.scatter(10,10,marker='s',color='k',alpha=0.7,s=50,label='GRIP')
ax.scatter(10,10,marker='X',color='k',alpha=0.7,s=90,label='NGRIP')
ax.scatter(10,10,marker='^',color='k',alpha=0.7,s=90,label='NEEM')

ax.plot(10,10,color='mediumblue',label=r'$K=10^1$')
ax.plot(10,10,color='forestgreen',label=r'$K=10^2$')
ax.plot(10,10,color='red',label=r'$K=10^3$')
ax.plot(10,10,color='orange',label=r'$K=10^4$')
ax.plot(10,10, color='w',  label=' ')
ax.plot(10,10, color='w', label=' ')

#------------------- Change order of items in legend
handles, labels = ax.get_legend_handles_labels() #get handles and labels
order = [6,7,8,9,10,11,4,5,0,1,2,3] #specify order of items in legend
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=1,bbox_to_anchor=(1.5, 0.8)) # add legend to plot

ax.set_xlim([-37,-20])
ax.set_ylim([-37,-20])
ax.set_xlabel(r'$\bar{T}_{\rm{inferred}}$ $(^\circ C)$',fontsize=12.5)
ax.set_ylabel(r'$\bar{T}_{\rm{observed}}$ $(^\circ C)$',fontsize=12.5)
 
fig.savefig('invT_vs_measT_raw.png',bbox_inches='tight',dpi=150,format='png')