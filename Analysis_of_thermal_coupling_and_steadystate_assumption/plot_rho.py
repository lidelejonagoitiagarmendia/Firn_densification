# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:04:42 2022

@author: lidel
"""

import numpy as np
import matplotlib.pyplot as plt 

def get_rho_rawdata(H,site):

    obs= np.loadtxt(r'icecores/density-raw/dens_%s.txt'%(site))#, header=0, dtype={'a': np.float32, 'b': np.float32}, sep='\s+')
    # df = pd.read_csv(r'Nicholas fit/icecores/density-fits/dens_%s_fit.txt'%(site), header=0, dtype={'a': np.float32, 'b': np.float32}, sep='\s+')
    # obs = df.to_numpy()
    z_obs = H - (obs[:,0] - 0*obs[0,0])
    I_obs = np.argsort(z_obs) # must be sorted for interpolation below
    z_obs = z_obs[I_obs]
    rho_obs = obs[I_obs,1] # to kg/m^3

    I_rm = np.argwhere(z_obs < 0)
    rho_obs = rho_obs[z_obs >= 0]
    z_obs   = z_obs[z_obs >= 0]
    
    return rho_obs, z_obs

H0=200
Hs={'dye3':H0, 'grip':H0, 'neem':H0-15, 'ngrip':H0-10, 'site_2':H0, 'siteA_crete':H0}
sites=['dye3','grip','neem','ngrip','site_2','siteA_crete']
sites=['grip']#['dye3','grip','neem','ngrip','site2','siteA_crete']
Ks=[500,1000]
Ks=[500]
rho_ice=910

for site in sites:
    
    H=Hs[site]
    
    rho_raw, z_rho_raw=get_rho_rawdata(H,site)
    
    for K in Ks:
        
        zSS=np.load('./SSresults/z_1D_'+str(site)+'_withoutT_K='+str(K)+'.npy')
        rhoSS=np.load('./SSresults/rho_1D_'+str(site)+'_withoutT_K='+str(K)+'.npy')
        zSST=np.load('./SSTresults/z_1D_'+str(site)+'_withT_K='+str(K)+'.npy')
        rhoSST=np.load('./SSTresults/rho_1D_'+str(site)+'_withT_K='+str(K)+'.npy')
        zT=np.load('./Tresults/z_1D_'+str(site)+'_withT_K='+str(K)+'.npy')
        rhoT=np.load('./Tresults/rho_1D_'+str(site)+'_withT_K='+str(K)+'.npy')
        z=np.load('./noTresults/z_1D_'+str(site)+'_withoutT_K='+str(K)+'.npy')
        rho=np.load('./noTresults/rho_1D_'+str(site)+'_withoutT_K='+str(K)+'.npy')
        
        plt.figure(figsize=(5,7))
        plt.scatter(rho_raw/rho_ice,-(max(z_rho_raw)-z_rho_raw),s=25,color='grey',label='Measured density at GRIP',facecolor='white',edgecolor='grey',alpha=0.8)
        plt.plot(rhoSS/rho_ice,(zSS-max(zSS)),color='darkcyan',label='Steady state (const. T)')
        plt.plot(rho/rho_ice,(z-max(z)),color='darkorange',label='Non steady state (const. T)')
        plt.plot(rhoT/rho_ice,(zT-max(zT)),'--',color='crimson',label='Non steady state (coupled T)')
        # plt.plot(rhoSST,-(zSST-max(zSST)),label='Steady state (coupled T)')        
        # plt.xlim([0.35,1])
        # plt.ylim([0,200])
        # plt.gca().invert_yaxis()
        # plt.title(str(site)+', K='+str(K))
        plt.xlabel(r'$\hat\rho$')
        plt.ylabel('z  (m)')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('./Comparison_figures/1D_comparison_'+str(site)+'_K='+str(K)+'_1.png',format='png',dpi=150)
        #1D_comparison_grip_K=500
        plt.show()


# for site in sites:
    
#     H=Hs[site]
    
#     rho_raw, z_rho_raw=get_rho_rawdata(H,site)
        
#     zSS=np.load('./SSresults/z_1D_'+str(site)+'_withoutT_K=500.npy')
#     rhoSS=np.load('./SSresults/rho_1D_'+str(site)+'_withoutT_K=500.npy')
#     zT=np.load('./Tresults/z_1D_'+str(site)+'_withT_K=500.npy')
#     rhoT=np.load('./Tresults/rho_1D_'+str(site)+'_withT_K=500.npy')
#     z=np.load('./noTresults/z_1D_'+str(site)+'_withoutT_K=500.npy')
#     rho=np.load('./noTresults/rho_1D_'+str(site)+'_withoutT_K=500.npy')
    
#     zSS1=np.load('./SSresults/z_1D_'+str(site)+'_withoutT_K=1000.npy')
#     rhoSS1=np.load('./SSresults/rho_1D_'+str(site)+'_withoutT_K=1000.npy')
#     zT1=np.load('./Tresults/z_1D_'+str(site)+'_withT_K=1000.npy')
#     rhoT1=np.load('./Tresults/rho_1D_'+str(site)+'_withT_K=1000.npy')
#     z1=np.load('./noTresults/z_1D_'+str(site)+'_withoutT_K=1000.npy')
#     rho1=np.load('./noTresults/rho_1D_'+str(site)+'_withoutT_K=1000.npy')
    
#     plt.figure(figsize=(5,7))
#     plt.scatter(rho_raw,-(max(z_rho_raw)-z_rho_raw),s=1,color='grey')
#     plt.plot(rho,z-max(z),label='Constant T')
#     plt.plot(rhoT,zT-max(zT),label='Varying T')
#     plt.plot(rhoSS,zSS-max(zSS),label='SS')
#     plt.plot(rho1,z1-max(z1),'--',label='Constant T')
#     plt.plot(rhoT1,zT1-max(zT1),'--',label='Varying T')
#     plt.plot(rhoSS1,zSS1-max(zSS1),'--',label='SS')
#     # plt.plot(rho1D,(max(z1D)-z1D),label='Constant T')
#     # plt.plot(rho1DT,(max(z1DT)-z1DT),label='Varying T')
#     # plt.plot(rho1DSS,(max(z1DSS)-z1DSS),label='SS')
# #     # plt.plot(rho,-(max(z)-z),label='Constant T')
# #     # plt.plot(rhoT,(max(zT)-zT), label='Uncoupled, T=-28.8ºC')
# #     # # plt.plot(rhoT1,(max(zT1)-zT1), label='Coupled, HF=0.026 W/m$^2$')
# #     # plt.plot(rhoT2,(max(zT2)-zT2), label='Coupled, HF=0.079 W/m$^2$')
# #     # plt.plot(rhoT3,(max(zT3)-zT3), label='Coupled, HF=0.132 W/m$^2$')
# #     # plt.gca().invert_yaxis()
#     plt.title(str(site))
#     plt.xlabel('Density (kg/m$^3$)')
#     plt.ylabel('Depth (m)')
#     plt.legend()
#     plt.show()
