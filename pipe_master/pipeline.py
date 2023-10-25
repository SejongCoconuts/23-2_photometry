
import os 
import numpy as np
import pandas as pd

from tasks import * 

# EDIT HERE ==========================================================
wdir = ''

delt = 14
# ====================================================================

path_lc_ori = wdir + 'lightcurve.dat'
path_lc_pow = wdir + 'lightcurve_power.dat'
path_lc_bpo = wdir + 'lightcurve_broken_power.dat'
path_lc_spo = wdir + 'lightcurve_smoothed_power.dat'


os.system('python2 method_simLC.py {} {} {}'.format(path_csv, path_lc_ori, args_bpo))
os.system('python2 method_simLC.py {} {} {}'.format(path_csv, path_lc_pow, args_bpo))
os.system('python2 method_simLC.py {} {} {}'.format(path_csv, path_lc_bpo, args_bpo))
os.system('python2 method_simLC.py {} {} {}'.format(path_csv, path_lc_spo, args_bpo))


f0 = pd.read_csv(wdir+'lightcurve.dat',                delim_whitespace=True, header=None)
f1 = pd.read_csv(wdir+'lightcurve_power.dat',          delim_whitespace=True, header=None)
f2 = pd.read_csv(wdir+'lightcurve_broken_power.dat',   delim_whitespace=True, header=None)
f3 = pd.read_csv(wdir+'lightcurve_smoothed_power.dat', delim_whitespace=True, header=None)


#PSD, original data ========================
fj0, dft0 = PSD(delt, f0[0], f0[1])
fj1, dft1 = PSD(delt, f1[0], f1[1])
fj2, dft2 = PSD(delt, f2[0], f2[1])
fj3, dft3 = PSD(delt, f3[0], f3[1])
#===========================================


#DATA REMOVAL===============================
rdata0 = remove_below(thre, f0)
rdata1 = remove_below(thre, f1)
rdata2 = remove_below(thre, f2)
rdata3 = remove_below(thre, f3)
#===========================================


#PSD, removed data==========================
rfj0, rdft0 = PSD(delt, rdata0[0], rdata0[1])
rfj1, rdft1 = PSD(delt, rdata1[0], rdata1[1])
rfj2, rdft2 = PSD(delt, rdata2[0], rdata2[1])
rfj3, rdft3 = PSD(delt, rdata3[0], rdata3[1])
#==========================================


#INTERPOLATION, original data=======================================
l_inter_time0, l_inter_flux0 = interpolation(f0[0], f0[1], 'linear')
l_inter_time1, l_inter_flux1 = interpolation(f1[0], f1[1], 'linear')
l_inter_time2, l_inter_flux2 = interpolation(f2[0], f2[1], 'linear')
l_inter_time3, l_inter_flux3 = interpolation(f3[0], f3[1], 'linear')

s_inter_time0, s_inter_flux0 = interpolation(f0[0], f0[1], 'spline')
s_inter_time1, s_inter_flux1 = interpolation(f1[0], f1[1], 'spline')
s_inter_time2, s_inter_flux2 = interpolation(f2[0], f2[1], 'spline')
s_inter_time3, s_inter_flux3 = interpolation(f3[0], f3[1], 'spline')

g_inter_time0, g_inter_flux0 = interpolation(f0[0], f0[1], 'gauss')
g_inter_time1, g_inter_flux1 = interpolation(f1[0], f1[1], 'gauss')
g_inter_time2, g_inter_flux2 = interpolation(f2[0], f2[1], 'gauss')
g_inter_time3, g_inter_flux3 = interpolation(f3[0], f3[1], 'gauss') 
#===================================================================


#INTERPOLATION, removeds========================================================
r_l_inter_time0, r_l_inter_flux0 = interpolation(rdata0[0], rdata0[1], 'linear')
r_l_inter_time1, r_l_inter_flux1 = interpolation(rdata1[0], rdata1[1], 'linear')
r_l_inter_time2, r_l_inter_flux2 = interpolation(rdata2[0], rdata2[1], 'linear')
r_l_inter_time3, r_l_inter_flux3 = interpolation(rdata3[0], rdata3[1], 'linear')

r_s_inter_time0, r_s_inter_flux0 = interpolation(rdata0[0], rdata0[1], 'spline')
r_s_inter_time1, r_s_inter_flux1 = interpolation(rdata1[0], rdata1[1], 'spline')
r_s_inter_time2, r_s_inter_flux2 = interpolation(rdata2[0], rdata2[1], 'spline')
r_s_inter_time3, r_s_inter_flux3 = interpolation(rdata3[0], rdata3[1], 'spline')

r_g_inter_time0, r_g_inter_flux0 = interpolation(rdata0[0], rdata0[1], 'gauss')
r_g_inter_time1, r_g_inter_flux1 = interpolation(rdata1[0], rdata1[1], 'gauss')
r_g_inter_time2, r_g_inter_flux2 = interpolation(rdata2[0], rdata2[1], 'gauss')
r_g_inter_time3, r_g_inter_flux3 = interpolation(rdata3[0], rdata3[1], 'gauss') 
#===============================================================================


#PSD, interpolated=============================================================================
l_inter_fj0, l_inter_dft0 = PSD(delt, pd.DataFrame(l_inter_time0), pd.DataFrame(l_inter_flux0))
l_inter_fj1, l_inter_dft1 = PSD(delt, pd.DataFrame(l_inter_time1), pd.DataFrame(l_inter_flux1))
l_inter_fj2, l_inter_dft2 = PSD(delt, pd.DataFrame(l_inter_time2), pd.DataFrame(l_inter_flux2))
l_inter_fj3, l_inter_dft3 = PSD(delt, pd.DataFrame(l_inter_time3), pd.DataFrame(l_inter_flux3))

s_inter_fj0, s_inter_dft0 = PSD(delt, pd.DataFrame(s_inter_time0), pd.DataFrame(s_inter_flux0))
s_inter_fj1, s_inter_dft1 = PSD(delt, pd.DataFrame(s_inter_time1), pd.DataFrame(s_inter_flux1))
s_inter_fj2, s_inter_dft2 = PSD(delt, pd.DataFrame(s_inter_time2), pd.DataFrame(s_inter_flux2))
s_inter_fj3, s_inter_dft3 = PSD(delt, pd.DataFrame(s_inter_time3), pd.DataFrame(s_inter_flux3))

g_inter_fj0, g_inter_dft0 = PSD(delt, pd.DataFrame(g_inter_time0), pd.DataFrame(g_inter_flux0))
g_inter_fj1, g_inter_dft1 = PSD(delt, pd.DataFrame(g_inter_time1), pd.DataFrame(g_inter_flux1))
g_inter_fj2, g_inter_dft2 = PSD(delt, pd.DataFrame(g_inter_time2), pd.DataFrame(g_inter_flux2))
g_inter_fj3, g_inter_dft3 = PSD(delt, pd.DataFrame(g_inter_time3), pd.DataFrame(g_inter_flux3))
#==============================================================================================


#PSD, interpolated, removed ===========================================================================
r_l_inter_fj0, r_l_inter_dft0 = PSD(delt, pd.DataFrame(r_l_inter_time0), pd.DataFrame(r_l_inter_flux0))
r_l_inter_fj1, r_l_inter_dft1 = PSD(delt, pd.DataFrame(r_l_inter_time1), pd.DataFrame(r_l_inter_flux1))
r_l_inter_fj2, r_l_inter_dft2 = PSD(delt, pd.DataFrame(r_l_inter_time2), pd.DataFrame(r_l_inter_flux2))
r_l_inter_fj3, r_l_inter_dft3 = PSD(delt, pd.DataFrame(r_l_inter_time3), pd.DataFrame(r_l_inter_flux3))

r_s_inter_fj0, r_s_inter_dft0 = PSD(delt, pd.DataFrame(r_s_inter_time0), pd.DataFrame(r_s_inter_flux0))
r_s_inter_fj1, r_s_inter_dft1 = PSD(delt, pd.DataFrame(r_s_inter_time1), pd.DataFrame(r_s_inter_flux1))
r_s_inter_fj2, r_s_inter_dft2 = PSD(delt, pd.DataFrame(r_s_inter_time2), pd.DataFrame(r_s_inter_flux2))
r_s_inter_fj3, r_s_inter_dft3 = PSD(delt, pd.DataFrame(r_s_inter_time3), pd.DataFrame(r_s_inter_flux3))

r_g_inter_fj0, r_g_inter_dft0 = PSD(delt, pd.DataFrame(r_g_inter_time0), pd.DataFrame(r_g_inter_flux0))
r_g_inter_fj1, r_g_inter_dft1 = PSD(delt, pd.DataFrame(r_g_inter_time1), pd.DataFrame(r_g_inter_flux1))
r_g_inter_fj2, r_g_inter_dft2 = PSD(delt, pd.DataFrame(r_g_inter_time2), pd.DataFrame(r_g_inter_flux2))
r_g_inter_fj3, r_g_inter_dft3 = PSD(delt, pd.DataFrame(r_g_inter_time3), pd.DataFrame(r_g_inter_flux3))
#======================================================================================================
