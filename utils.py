# coding: utf-8
from scipy.stats import mannwhitneyu
import numpy as np 


def mann_witt_matrix(mat,y):
    lcol = mat.shape[1]
    pvalue = np.zeros((lcol,lcol))
    statU = np.zeros((lcol,lcol))
    probav = np.zeros((2,lcol,lcol))
    unik = (np.unique(y))
    valstd=  unik[0]
    valdev= unik[1]
    for i in range(lcol):
        for j in range(lcol):
            #curp_S=mat[:,i,j,0]
            #curp_D=mat[:,i,j,1]
            curp_S_S=mat[(y==valstd),i,j,0]
            curp_S_D=mat[(y==valdev),i,j,0]
            
            stat,pval = mannwhitneyu(curp_S_S,curp_S_D,alternative='two-sided')
            statU[i,j]=stat
            pvalue[i,j]=pval
            
            probav[0,i,j]= np.mean(curp_S_S,axis=0)
            probav[1,i,j]= np.mean(curp_S_D,axis=0)
    
    return statU,pvalue,probav
            
def mann_witt_all(bigmat,y):
    resU = []
    resP = []
    resprobav = []
    for mat in bigmat:
        U,pval,probav = mann_witt_matrix(mat,y)
        
        resU.append(U)
        resP.append(pval)
        resprobav.append(probav)
        
    return np.stack(resU),np.stack(resP),np.stack(resprobav)

import matplotlib.pyplot as plt 

def plot_ROC_allcond(allscores,timepoints,figtitle='Default Title'):

    fig, (ax) = plt.subplots(ncols=2,nrows=2,figsize=(10,10))

    titles = ['Train Regular Test Regular',
              'Train Regular Test Irregular',
              'Train Irregular Test Regular',
              'Train Irregular Test Irregular']

    for i in range(4):
        curax = ax.ravel()[i]

        im = curax.matshow(allscores.mean(axis=1)[i], cmap='RdBu_r', vmin=0.2,vmax=0.8,origin='lower',
                            extent=timepoints)
        curax.axhline(0., color='k')
        curax.axvline(0., color='k')
        curax.xaxis.set_ticks_position('bottom')
        curax.set_xlabel('Testing Time (s)')
        curax.set_ylabel('Training Time (s)')
        curax.set_title(titles[i])
        plt.colorbar(im, ax=curax)
    fig.suptitle(figtitle)
    return fig