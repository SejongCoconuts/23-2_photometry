from DELCgen import *
import scipy.stats as st

#------- Input parameters -------

# File Route
datfile = "Mkn501_r.txt"

# Simulation params
RedNoiseL,aliasTbin, tbin = 100,1,100 

# load data lightcurve
datalc = Load_Lightcurve(datfile,tbin)

datalc.Fit_PSD()

datalc.Fit_PDF()

# Bending power law params
A,v_bend,a_low,a_high,c =0.00809431301277, 0.00145176982764,0.996666659769,23.7459334225,15.9486529797
                          
# Probability density function params
kappa,theta,lnmu,lnsig,weight = 6.0,6.0,0.3,7.4,0.8



delc = datalc.Simulate_DE_Lightcurve()

delc.Save_Lightcurve('Mkn501_rs.dat')