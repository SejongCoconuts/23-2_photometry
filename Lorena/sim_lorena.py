import scipy.stats as st
import numpy as np
import astropy.modeling
import os
import sys
from DELCgen import *
import matplotlib.pyplot as plt


### Info about the params structure of de PSD: [amplitude, x_break, alpha1, alpha2]; params structure of the PDF: [kappa,theta,lnmu,lnsig,weight]

paramspsd = [5e-10, 0.05492, 0.88712, 20.03170] # Arbitrary fitting values, in this case I am using BPL as an example
paramspdf = [6, 1, 0.7157433993358401, 1.1964475745604903, 0.82] # kappa, theta, lnmu, lnsig, weight
n = 100 # Numbers of simulating LCs that you want

###### path of your data:

path_datafile = "/home/loramaya/filtered.csv"

#### Definition for the fitting model that it is the best for the CTA102 source:

def Bpower_law(fmean, amplitude, x_break, alpha1, alpha2): # Define the broken power-law model function
    information = astropy.modeling.powerlaws.BrokenPowerLaw1D(amplitude = abs(amplitude), x_break=x_break, alpha_1=alpha1, alpha_2=alpha2)
    return information.evaluate(fmean, information.amplitude[0], information.x_break[0], information.alpha_1[0], information.alpha_2[0])
    
def SPL(v, alpha1, b1):
    return 10**(np.log10(v)*alpha1 + np.log10(b1))
    

RedNoiseL, aliasTbin, tbin = 100, 1, 14

# Load data lightcurve
datalc = Load_Lightcurve(path_datafile, tbin)


ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(datalc.time, datalc.flux, 'b-o', label = 'Original LC', zorder = 2)
ax1.legend(loc = 'best')

#### Loop def:

def lcs_sim(paramspsd, paramspdf, nn):

  
     # create mixture distribution to fit to PDF
    #mix_model = Mixture_Dist([st.gamma, st.lognorm],[3,3],[[[2],[0]],[[2],[0],]])
    
    #datalc.Fit_PDF(initial_params=paramspdf,model= mix_model)
    
    #Simulate the lightcurve using the EmmanLC function:
    
   # surrogate, PSDlast, shortLC, periodogram, fft = EmmanLC(time = datalc.time, RedNoiseL =  RedNoiseL, aliasTbin = aliasTbin, tbin = tbin, PSDmodel=Bpower_law, PSDparams=paramspsd,
                                                #PDFmodel=mix_model, PDFparams=(datalc.pdfFit['x'][0], datalc.pdfFit['x'][1], datalc.pdfFit['x'][3], np.exp(datalc.pdfFit['x'][2]), datalc.pdfFit['x'][4]))   
                
    delc_mod = Simulate_DE_Lightcurve(Bpower_law, paramspsd, st.lognorm, paramspdf, lightcurve=datalc)

    
    ax1.plot(delc_mod.time, delc_mod.flux, 'k-o', zorder = 1)
    ax2.plot(np.log10(delc_mod.periodogram[0]), np.log10(delc_mod.periodogram[1]), 'k-o', zorder = 1)
    
    ###### Save the files:
    
    delc_mod.Save_Lightcurve('lightcurveCTA102_sim_{}.dat'.format(nn))

##### Calling the loop
for nn in range(n):
    lcs_sim(paramspsd, paramspdf, nn)    
plt.show()
