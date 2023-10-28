from DELCgen import *

def simLC(path_datafile, path_output, mode='original'):

    # python2 version of simulating LC
    # in case Jeff's workaround for py3 doesn't work

    # mode = original, simplePL, brokenPL, curvedPL
    
    import scipy.stats as st
    import numpy as np
    from astropy.modeling import models, fitting
    import os 
    import sys
    
    
    #------- Input parameters -------
    
    # File Route


    
    
    # org_time, org_flux = np.genfromtxt(path_datafile, usecols=(0,1), unpack=True)
    
    
    
    # Bending power law params
    A,v_bend,a_low,a_high,c = 0.03, 2.3e-4, 1.1, 2.2, 0.009
    
    # Probability density function params
    kappa,theta,lnmu,lnsig,weight = 5.67, 5.96, 2.14, 0.31,0.82
    # Simulation pa
    RedNoiseL,aliasTbin, tbin = 100,1,100
    
    #--------- Commands ---------------
    
    # load data lightcurve
    datalc = Load_Lightcurve(path_datafile,tbin)
    
    
    
    
    
    # create mixture distribution to fit to PDF
    mix_model = Mixture_Dist([st.gamma,st.lognorm],[3,3],[[[2],[0]],[[2],[0],]])
    
    
    # estimate underlying variance of data light curve
    
    
    
    if(mode=='original'):
        delc = datalc.Simulate_DE_Lightcurve()
    
    else:
        
        if(mode=='simplePL'):
            initials = [1,1]
            def model(v, alpha1, b1):
                return 10**(np.log10(v)*alpha1 + np.log10(b1))
                
        if(mode=='brokenPL'):
            initials = [1,1,1,1]
            def model(v,A,v_bend,a_high,c):
                p = BendingPL(v,A,v_bend,1.1,a_high,c)
                return p
            
        if(mode=='curvedPL'):
            initials = [1,1,1]
            def model(v, alpha1, alpha2, b1):
                return 10**(alpha1*np.log10(v)**2 + alpha2*np.log10(v) + np.log10(b1))
        
        datalc.Fit_PSD(initial_params=initials,model=model)
        datalc.STD_Estimate()
        # delc = Simulate_DE_Lightcurve(model,lightcurve=datalc)
        
        delc = datalc.Simulate_DE_Lightcurve()
    
    # save the simulated light curve as a txt file
    delc.Save_Lightcurve(path_output)



import sys

path_datafile = sys.argv[1]
path_output   = sys.argv[2]
mode = sys.argv[3]

print(path_datafile)

simLC(path_datafile, path_output, mode)
