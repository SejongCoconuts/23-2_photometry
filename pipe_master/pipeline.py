
import os 
import numpy as np
import pandas as pd
# EDIT HERE ==========================================================
wdir = ''
======================================================================



f0 = pd.read_csv(wdir+'lightcurve.dat', delim_whitespace=True, header=None)
f1 = pd.read_csv(wdir+'lightcurve_power.dat', delim_whitespace=True, header=None)
f2 = pd.read_csv(wdir+'lightcurve_broken_power.dat', delim_whitespace=True, header=None)
f3 = pd.read_csv(wdir+'lightcurve_smoothed_power.dat', delim_whitespace=True, header=None)

