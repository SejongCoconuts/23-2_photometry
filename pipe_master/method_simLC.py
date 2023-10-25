
def simLC():

    # python2 version of simulating LC
    # in case Jeff's workaround for py3 doesn't work

  
    from DELCgen import *
    import scipy.stats as st
    import numpy as np
    from astropy.modeling import models, fitting
    import os 
    import sys
    
    #------- Input parameters -------
    
    # File Route

    path_datafile = sys.argv[1]
    path_output   = sys.argv[2]
    
    
    org_time, org_flux = np.genfromtxt(path_datafile, usecols=(0,1), unpack=True)
    
    
    # Bending power law params
    A,v_bend,a_low,a_high,c = 0.03, 2.3e-4, 1.1, 2.2, 0.009
    
    # Probability density function params
    kappa,theta,lnmu,lnsig,weight = 5.67, 5.96, 2.14, 0.31,0.82
    # Simulation pa
    RedNoiseL,aliasTbin, tbin = 100,1,100
    
    #--------- Commands ---------------
    
    # load data lightcurve
    datalc = Load_Lightcurve(path_datafile,tbin)
    
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
    delc.Save_Lightcurve(path_output)
