import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import Rbf

# Load data

path_lc = input('Data File Name:')
df = pd.read_csv(path_lc)[['Julian Date', 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)', 'Photon Flux Error(photons cm-2 s-1)']]

# Remove spaces and operation symbols in data

df = df[df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].str.strip() != '']
df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'] = df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].str.replace('>', '').str.replace('<', '')
df.to_csv('test.csv', index=False)


def remove_below(threshold, data):
    thre = threshold*max(data[1])
    cons = (data[1] > thre)
    new = data[cons]
    return new

def PSD(delt, time, flux):

    import numpy as np
    
    fj = []
    for i in range(len(flux)):
        fj.append((i)/(len(flux)*delt))
    xbar = flux - np.mean(flux)
    time = time - time.values[0]
    # dft_list = []
    # for i in range(len(flux)):
    #     dft = (np.sum(xbar*np.cos(2*np.pi*fj[i]*time)))**2 + \
    #         (np.sum(xbar*np.sin(2*np.pi*fj[i]*time)))**2
    #     dft_list.append(dft)
    dft = (np.sum(xbar*np.cos(2*np.pi*fj*time)))**2 + (np.sum(xbar*np.sin(2*np.pi*fj*time)))**2
    return fj[1:], dft_list[1:]

def interpolation(x, y, Type):
    import numpy as np
    from scipy.interpolate import Rbf, CubicSpline
  
    if (Type == 'linear'):
        xnew = np.linspace(min(x), max(x), 1000)
        ynew = np.interp(xnew, x, y)
        return xnew, ynew
    if (Type == 'spline'):
        cs = CubicSpline(x, y)
        xs = np.linspace(min(x), max(x), 1000)
        ys = cs(xs)
        return xs, ys
    if (Type == 'gauss'):
        rbfi = Rbf(x, y, function='gaussian')
        xnew = np.linspace(min(x), max(x), 1000)
        ynew = rbfi(xnew)
        return xnew, ynew   


def simulate_LC():

    from DELCgen3 import *
    import scipy.stats as st
    import numpy as np
    from astropy.modeling import models, fitting
    
    #------- Input parameters -------
    
    # File Route
    
    route = "/Users/shinnakim/fermi/fermipy-tutorial/data/blazar/mysource_all/lightcurve/"
    datfile = "t_coltrim.dat"
    
    org_time, org_flux = np.genfromtxt(route+datfile, usecols=(0,1), unpack=True)
    
    
    # Bending power law params
    A,v_bend,a_low,a_high,c = 0.03, 2.3e-4, 1.1, 2.2, 0.009
    
    # Probability density function params
    kappa,theta,lnmu,lnsig,weight = 5.67, 5.96, 2.14, 0.31,0.82
    # Simulation params
    RedNoiseL,aliasTbin, tbin = 100,1,100
    
    #--------- Commands ---------------
    
    # load data lightcurve
    datalc = Load_Lightcurve(route+datfile,tbin)
    
    def Fix_BL(v,A,v_bend,a_high,c):
        p = BendingPL(v,A,v_bend,1.1,a_high,c)
        return p
    
    datalc.Fit_PSD(initial_params=[1,0.001,2.5,0],model=Fix_BL)
    
    # create mixture distribution to fit to PDF
    mix_model = Mixture_Dist([st.gamma,st.lognorm],[3,3],[[[2],[0]],[[2],[0],]])
    
    
    # estimate underlying variance of data light curve
    datalc.STD_Estimate()
    
    # simulate artificial light curve with Timmer & Koenig method
    tklc = Simulate_TK_Lightcurve(BendingPL, (A,v_bend,a_low,a_high,c),lightcurve=datalc,
                                    RedNoiseL=RedNoiseL,aliasTbin=aliasTbin)
    
    # simulate artificial light curve with Emmanoulopoulos method, scipy distribution
    delc_mod = Simulate_DE_Lightcurve(BendingPL, (A,v_bend,a_low,a_high,c),
                                   mix_model, (kappa, theta, lnsig, np.exp(lnmu),
                                                                  weight,1-weight),lightcurve=datalc)
    
    # simulate artificial light curve with Emmanoulopoulos method, using the PSD
    # and PDF of the data light curve, with default parameters (bending power law
    # for PSD and mixture distribution of gamma and lognormal distribution for PDF)
    delc = datalc.Simulate_DE_Lightcurve()
    
    # save the simulated light curve as a txt file
    delc.Save_Lightcurve('mylightcurve.dat')





