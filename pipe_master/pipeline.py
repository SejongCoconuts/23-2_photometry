import os 
import numpy as np
import pandas as pd

from tasks import * 
import matplotlib.pyplot as plt
import readline

# EDIT HERE ==========================================================
delt = 14
thre = 0.5
# ====================================================================


# Read line 

readline.parse_and_bind("tab: complete")
def path_completer(text, state):
    return [x for x in os.listdir('.') if x.startswith(text)][state]
readline.set_completer(path_completer)

# Load data

path_csv = input('Data File Name:')

if not path_csv.startswith('./'):
    path_csv = './' + path_csv

wdir = os.path.dirname(path_csv)+'/'
os.chdir(wdir)

df = pd.read_csv(path_csv)[['Julian Date', 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)', 'Photon Flux Error(photons cm-2 s-1)']]

# Remove spaces and operation symbols in data

df = df[df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].str.strip() != '']
df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'] = df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].str.replace('>', '').str.replace('<', '')

df.to_csv('filtered.csv', index=False)
df.to_string(wdir+'filtered.csv', index=False)
path_csv = wdir+'filtered.csv'

path_lc_ori = wdir + 'lightcurve.dat'
path_lc_pow = wdir + 'lightcurve_power.dat'
path_lc_bpo = wdir + 'lightcurve_broken_power.dat'
path_lc_spo = wdir + 'lightcurve_smoothed_power.dat'


os.system('python2 method_simLC.py {} {} original'.format(path_csv, path_lc_ori))
os.system('python2 method_simLC.py {} {} simplePL'.format(path_csv, path_lc_pow))
os.system('python2 method_simLC.py {} {} brokenPL'.format(path_csv, path_lc_bpo))
os.system('python2 method_simLC.py {} {} curvedPL'.format(path_csv, path_lc_spo))


f0 = pd.read_csv(wdir+'lightcurve.dat',                delim_whitespace=True, header=1, names=[0,1])
f1 = pd.read_csv(wdir+'lightcurve_power.dat',          delim_whitespace=True, header=1, names=[0,1])
f2 = pd.read_csv(wdir+'lightcurve_broken_power.dat',   delim_whitespace=True, header=1, names=[0,1])
f3 = pd.read_csv(wdir+'lightcurve_smoothed_power.dat', delim_whitespace=True, header=1, names=[0,1])



#PSD, original data ========================
fj0, dft0 = PSD(delt, f0[0], f0[1])
fj1, dft1 = PSD(delt, f1[0], f1[1])
fj2, dft2 = PSD(delt, f2[0], f2[1])
fj3, dft3 = PSD(delt, f3[0], f3[1])
#===========================================


#DATA REMOVAL===============================
rdata0 = f0.loc[f0[1]>f0[1].max()*thre]
rdata1 = f1.loc[f1[1]>f1[1].max()*thre]
rdata2 = f2.loc[f2[1]>f2[1].max()*thre]
rdata3 = f3.loc[f3[1]>f3[1].max()*thre]
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


origin_image_plot = True
removed_image_plot = True

psd_image_plot = True
removed_psd_plot = True

pdf_image_plot = True
removed_pdf_plot = True

linear_interp_plot = True
linear_inter_psd = True
removed_linear_psd = True
linear_inter_pdf = True
removed_linear_pdf = True

spline_interp_plot = True
spline_inter_psd = True
removed_spline_psd = True
spline_inter_pdf = True
removed_spline_pdf = True

gauss_inter_plot = True
gauss_inter_psd = True
removed_gauss_psd = True
gauss_inter_pdf = True
removed_gauss_pdf = True

remove_linear_interp_plot = True
remove_spline_interp_plot = True
remove_gauss_interp_plot = True

imagesave = True


if (removed_gauss_pdf == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(np.array(r_g_inter_dft0))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(np.array(r_g_inter_dft1))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(np.array(r_g_inter_dft2))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(np.array(r_g_inter_dft3))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_gauss_PDFplots.png', dpi=300) 
        

if (removed_spline_pdf == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(np.array(r_s_inter_dft0))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(np.array(r_s_inter_dft1))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(np.array(r_s_inter_dft2))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(np.array(r_s_inter_dft3))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_spline_PDFplots.png', dpi=300) 
        
if (removed_linear_pdf == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(np.array(r_l_inter_dft0))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(np.array(r_l_inter_dft1))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(np.array(r_l_inter_dft2))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(np.array(r_l_inter_dft3))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_linear_PDFplots.png', dpi=300) 

if (gauss_inter_pdf == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(np.array(g_inter_dft0))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(np.array(g_inter_dft1))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(np.array(g_inter_dft2))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(np.array(g_inter_dft3))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('gauss_inter_PDFplots.png', dpi=300) 

if (spline_inter_pdf == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(np.array(s_inter_dft0))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(np.array(s_inter_dft1))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(np.array(s_inter_dft2))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(np.array(s_inter_dft3))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('spline_inter_PDFplots.png', dpi=300) 
        
if (linear_inter_pdf == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(np.array(l_inter_dft0))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(np.array(l_inter_dft1))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(np.array(l_inter_dft2))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(np.array(l_inter_dft3))
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('linear_inter_PDFplots.png', dpi=300) 
        
        
if (removed_gauss_psd == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(r_g_inter_fj0), np.log10(r_g_inter_dft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(r_g_inter_fj1), np.log10(r_g_inter_dft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(r_g_inter_fj2), np.log10(r_g_inter_dft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(r_g_inter_fj3), np.log10(r_g_inter_dft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_gauss_inter_PSDplots_thre%s.png' %(thre), dpi=300) 
        
        
if (removed_spline_psd == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(r_s_inter_fj0), np.log10(r_s_inter_dft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(r_s_inter_fj1), np.log10(r_s_inter_dft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(r_s_inter_fj2), np.log10(r_s_inter_dft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(r_s_inter_fj3), np.log10(r_s_inter_dft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_spline_inter_PSDplots_thre%s.png' %(thre), dpi=300) 

if (removed_linear_psd == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(r_l_inter_fj0), np.log10(r_l_inter_dft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(r_l_inter_fj1), np.log10(r_l_inter_dft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(r_l_inter_fj2), np.log10(r_l_inter_dft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(r_l_inter_fj3), np.log10(r_l_inter_dft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_linear_inter_PSDplots_thre%s.png' %(thre), dpi=300) 



if (gauss_inter_psd == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(g_inter_fj0), np.log10(g_inter_dft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(g_inter_fj1), np.log10(g_inter_dft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(g_inter_fj2), np.log10(g_inter_dft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(g_inter_fj3), np.log10(g_inter_dft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('gauss_interpolated_PSDplots.png', dpi=300) 


if (spline_inter_psd == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(s_inter_fj0), np.log10(s_inter_dft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(s_inter_fj1), np.log10(s_inter_dft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(s_inter_fj2), np.log10(s_inter_dft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(s_inter_fj3), np.log10(s_inter_dft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('spline_interpolated_PSDplots.png', dpi=300) 

if (linear_inter_psd == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(l_inter_fj0), np.log10(l_inter_dft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(l_inter_fj1), np.log10(l_inter_dft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(l_inter_fj2), np.log10(l_inter_dft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(l_inter_fj3), np.log10(l_inter_dft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('linear_interpolated_PSDplots.png', dpi=300) 
    

if (remove_gauss_interp_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.plot(r_g_inter_time0, r_g_inter_flux0, c='blue', zorder=10)
    plt.scatter(rdata0[0], rdata0[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.plot(r_g_inter_time1, r_g_inter_flux1, c='blue', zorder=10)
    plt.scatter(rdata1[0], rdata1[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.plot(r_g_inter_time2, r_g_inter_flux2, c='blue', zorder=10)
    plt.scatter(rdata2[0], rdata2[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.plot(r_g_inter_time3, r_g_inter_flux3, c='blue', zorder=10)
    plt.scatter(rdata3[0], rdata3[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_gauss_interp_LCplots.png', dpi=300)
        
if (remove_spline_interp_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.plot(r_s_inter_time0, r_s_inter_flux0, c='blue', zorder=10)
    plt.scatter(rdata0[0], rdata0[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.plot(r_s_inter_time1, r_s_inter_flux1, c='blue', zorder=10)
    plt.scatter(rdata1[0], rdata1[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.plot(r_s_inter_time2, r_s_inter_flux2, c='blue', zorder=10)
    plt.scatter(rdata2[0], rdata2[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.plot(r_s_inter_time3, r_s_inter_flux3, c='blue', zorder=10)
    plt.scatter(rdata3[0], rdata3[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_spline_interp_LCplots.png', dpi=300)



if (remove_linear_interp_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.plot(r_l_inter_time0, r_l_inter_flux0, c='blue', zorder=10)
    plt.scatter(rdata0[0], rdata0[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.plot(r_l_inter_time1, r_l_inter_flux1, c='blue', zorder=10)
    plt.scatter(rdata1[0], rdata1[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.plot(r_l_inter_time2, r_l_inter_flux2, c='blue', zorder=10)
    plt.scatter(rdata2[0], rdata2[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.plot(r_l_inter_time3, r_l_inter_flux3, c='blue', zorder=10)
    plt.scatter(rdata3[0], rdata3[1], c='black', s=5, zorder=15)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_linear_interp_LCplots.png', dpi=300)
        

if (gauss_inter_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.plot(g_inter_time0, g_inter_flux0, c='blue')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.plot(g_inter_time1, g_inter_flux1, c='blue')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.plot(g_inter_time2, g_inter_flux2, c='blue')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.plot(g_inter_time3, g_inter_flux3, c='blue')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('gauss_interp_LCplots.png', dpi=300)
        

if (spline_interp_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.plot(s_inter_time0, s_inter_flux0, c='orange')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.plot(s_inter_time1, s_inter_flux1, c='orange')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.plot(s_inter_time2, s_inter_flux2, c='orange')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.plot(s_inter_time3, s_inter_flux3, c='orange')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('spline_interp_LCplots.png', dpi=300)
        

if (linear_interp_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.plot(l_inter_time0, l_inter_flux0, c='green')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.plot(l_inter_time1, l_inter_flux1, c='green')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.plot(l_inter_time2, l_inter_flux2, c='green')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.plot(l_inter_time3, l_inter_flux3, c='green')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('linear_interp_LCplots.png', dpi=300)
    
    

if (removed_pdf_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(rdft0, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(rdft1, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(rdft2, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(rdft3, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_PDFplots_thre%s.png' %(thre), dpi=300) 


if (removed_psd_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(rfj0), np.log10(rdft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(rfj1), np.log10(rdft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(rfj2), np.log10(rdft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(rfj3), np.log10(rdft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_PSDplots_thre%s.png' %(thre), dpi=300) 


if (pdf_image_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.hist(dft0, bins=50, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PDF', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.hist(dft1, bins=50, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.hist(dft2, bins=50, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PDF', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.hist(dft3, bins=50, color='black')    
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PDF', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('PDFplots.png', dpi=300) 

if (psd_image_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.7, hspace=1)
    plt.subplot(2,2,1)
    plt.scatter(np.log10(fj0), np.log10(dft0), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original PSD', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(np.log10(fj1), np.log10(dft1), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(np.log10(fj2), np.log10(dft2), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law PSD', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(np.log10(fj3), np.log10(dft3), c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law PSD', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('PSDplots.png', dpi=300) 
        
    
if (removed_image_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.scatter(rdata0[0], rdata0[1], c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.scatter(rdata1[0], rdata1[1], c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.scatter(rdata2[0], rdata2[1], c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.scatter(rdata3[0], rdata3[1], c='black', s=3)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)

    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('removed_LCplots_thre%s.png' %(thre), dpi=300)
        

if (origin_image_plot == True):
    
    plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.5, hspace=1.2)
    plt.subplot(2,2,1)
    plt.plot(f0[0], f0[1], c='black')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    #plt.xticks(fontsize=5)
    plt.title('Original LC', fontsize=7)
    
    plt.subplot(2,2,2)
    plt.plot(f1[0], f1[1], c='black')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,3)
    plt.plot(f2[0], f2[1], c='black')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Broken-Power-Law LC', fontsize=7)
    
    plt.subplot(2,2,4)
    plt.plot(f3[0], f3[1], c='black')
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Flux', fontsize=7)
    plt.title('Smoothed-Power-Law LC', fontsize=7)
    
    if (imagesave == True):
        os.chdir(wdir)
        plt.savefig('LCplots.png', dpi=300)
