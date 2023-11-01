from DELCgen import *
import matplotlib.pyplot as plt


def simLC(path_datafile, path_output, mode='original',paramsSPL=[1,-1],paramsBPL=[1,1,1,1],paramsCPL=[1,1,1],fit_original=False,create_arbitrary=True):

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
    kappa,theta,lnmu,lnsig,weight = 1.67, 1.96, 1.14, 0.31,0.12
    # Simulation pa
    RedNoiseL,aliasTbin, RandomSeed, tbin = 100,1,6501,7

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

        # datalc.Fit_PSD(initial_params=initials,model=model)
        datalc.Fit_PDF(initial_params=[kappa,theta,lnmu,lnsig,weight],model=mix_model )
        # datalc.STD_Estimate()
        # delc = Simulate_DE_Lightcurve(model,lightcurve=datalc)
        if fit_original == True:
            delc = datalc.Simulate_DE_Lightcurve()
        elif create_arbitrary == True:
            if(mode=='simplePL'):
                # print(datalc.pdfFit[s'x'][0])
                #delc = datalc.Simulate_DE_Lightcurve(model,(paramsSPL[0],paramsSPL[1]))
                print(paramsSPL)
                print(model)
                surrogate, PSDlast, shortLC, periodogram, fft = EmmanLC(datalc.time, RedNoiseL,aliasTbin,RandomSeed, tbin, PSDmodel=model, PSDparams=paramsSPL,
                                                PDFmodel=mix_model, PDFparams=[kappa,theta,lnmu,lnsig,weight])
                ## surrogate[0] = simulated time, surrogate[1] = simulated flux
                ## PSDLast[0] = simulated freq, PSDLast[1] = simulated power
                ax1 = plt.subplot(121)
                ax2 = plt.subplot(122)
                ax1.plot(surrogate[0],surrogate[1],label="Simulated LC params:{0}".format(paramsSPL))
                ax2.plot(np.log10(PSDlast[0]),np.log10(PSDlast[1]),label="Simulated PSD")
                ax1.legend()


    # save the simulated light curve as a txt file
    # delc.Save_Lightcurve(path_output)



import sys

path_datafile = "./filtered.csv"
path_output   = "./test1.lc"
mode = "simplePL"

print(path_datafile)


simLC(path_datafile, path_output, mode, paramsSPL = [1,1])
simLC(path_datafile, path_output, mode, paramsSPL = [0,1])
simLC(path_datafile, path_output, mode, paramsSPL = [-1,1])



plt.show()
