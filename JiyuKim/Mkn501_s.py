from DELCgen import *
import scipy.stats as st

#------- Input parameters -------

# File Route
datfile = "Mkn501.txt"

# Simulation params
RedNoiseL,aliasTbin, tbin = 100,1,100 

# load data lightcurve
datalc = Load_Lightcurve(datfile,tbin)

datalc.Fit_PSD()

datalc.Fit_PDF()

# Bending power law params
A,v_bend,a_low,a_high,c = 0.0138811083141,0.00571000795953,0.932238448798,1.24342229457,8.15457058163
                          
# Probability density function params
kappa,theta,lnmu,lnsig,weight = 6.0,6.0,0.3,7.4,0.8



delc = datalc.Simulate_DE_Lightcurve()

delc.Save_Lightcurve('Mkn501_s.dat')
