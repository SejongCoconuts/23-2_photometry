#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:14:48 2023

@author: jaebeom
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scienceplots
from scipy.interpolate import CubicSpline
from scipy.interpolate import Rbf

path = '/home/jaebeom/desktop/lecture/gamma/Emmanoulopoulos/DELightcurveSimulation'
savepath = '/home/jaebeom/desktop/lecture/gamma/3c279/psd/midterm'
#=======================================================parameter setting
origin_image_plot = False
removed_image_plot = False

psd_image_plot = False
removed_psd_plot = False

pdf_image_plot = False
removed_pdf_plot = False

linear_interp_plot = False
linear_inter_psd = False
removed_linear_psd = True
linear_inter_pdf = False
removed_linear_pdf = True

spline_interp_plot = False
spline_inter_psd = False
removed_spline_psd = True
spline_inter_pdf = False
removed_spline_pdf = True

gauss_inter_plot = False
gauss_inter_psd = False
removed_gauss_psd = True
gauss_inter_pdf = False
removed_gauss_pdf = True

remove_linear_interp_plot = False
remove_spline_interp_plot = False
remove_gauss_interp_plot = False

imagesave = False

thre = 0.5
#=======================================================parameter setting

os.chdir(path)

f0 = pd.read_csv('lightcurve.dat', delim_whitespace=True, header=None)
f1 = pd.read_csv('lightcurve_power_7.dat', delim_whitespace=True, header=None)
f2 = pd.read_csv('lightcurve_broken_power_7.dat', delim_whitespace=True, header=None)
f3 = pd.read_csv('lightcurve_smoothed_power_7.dat', delim_whitespace=True, header=None)

def remove_below(threshold, data):
    thre = threshold*max(data[1])
    cons = (data[1] > thre)
    new = data[cons]
    return new

def PSD(delt, time, flux):
    fj = []
    for i in range(len(flux)):
        fj.append((i)/(len(flux)*delt))
    xbar = flux - np.mean(flux)
    time = time - time.values[0]
    dft_list = []
    for i in range(len(flux)):
        dft = (np.sum(xbar*np.cos(2*np.pi*fj[i]*time)))**2 + \
            (np.sum(xbar*np.sin(2*np.pi*fj[i]*time)))**2
        dft_list.append(dft)
    return fj[1:], dft_list[1:]

def interpolation(x, y, Type):
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

#==========================================original data PSD
fj0, dft0 = PSD(14, f0[0], f0[1])
fj1, dft1 = PSD(14, f1[0], f1[1])
fj2, dft2 = PSD(14, f2[0], f2[1])
fj3, dft3 = PSD(14, f3[0], f3[1])
#==========================================original data PSD
#==========================================data removing from original data

rdata0 = remove_below(thre, f0)
rdata1 = remove_below(thre, f1)
rdata2 = remove_below(thre, f2)
rdata3 = remove_below(thre, f3)
#==========================================data removing from original data
#==========================================removed data PSD
rfj0, rdft0 = PSD(14, rdata0[0], rdata0[1])
rfj1, rdft1 = PSD(14, rdata1[0], rdata1[1])
rfj2, rdft2 = PSD(14, rdata2[0], rdata2[1])
rfj3, rdft3 = PSD(14, rdata3[0], rdata3[1])
#==========================================removed data PSD
#==========================================original data interpolation
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
#==========================================original data interpolation
#==========================================below removed data interpolation
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
#==========================================below removed data interpolation
#==========================================interpolated data psd
l_inter_fj0, l_inter_dft0 = PSD(14, pd.DataFrame(l_inter_time0), pd.DataFrame(l_inter_flux0))
l_inter_fj1, l_inter_dft1 = PSD(14, pd.DataFrame(l_inter_time1), pd.DataFrame(l_inter_flux1))
l_inter_fj2, l_inter_dft2 = PSD(14, pd.DataFrame(l_inter_time2), pd.DataFrame(l_inter_flux2))
l_inter_fj3, l_inter_dft3 = PSD(14, pd.DataFrame(l_inter_time3), pd.DataFrame(l_inter_flux3))

s_inter_fj0, s_inter_dft0 = PSD(14, pd.DataFrame(s_inter_time0), pd.DataFrame(s_inter_flux0))
s_inter_fj1, s_inter_dft1 = PSD(14, pd.DataFrame(s_inter_time1), pd.DataFrame(s_inter_flux1))
s_inter_fj2, s_inter_dft2 = PSD(14, pd.DataFrame(s_inter_time2), pd.DataFrame(s_inter_flux2))
s_inter_fj3, s_inter_dft3 = PSD(14, pd.DataFrame(s_inter_time3), pd.DataFrame(s_inter_flux3))

g_inter_fj0, g_inter_dft0 = PSD(14, pd.DataFrame(g_inter_time0), pd.DataFrame(g_inter_flux0))
g_inter_fj1, g_inter_dft1 = PSD(14, pd.DataFrame(g_inter_time1), pd.DataFrame(g_inter_flux1))
g_inter_fj2, g_inter_dft2 = PSD(14, pd.DataFrame(g_inter_time2), pd.DataFrame(g_inter_flux2))
g_inter_fj3, g_inter_dft3 = PSD(14, pd.DataFrame(g_inter_time3), pd.DataFrame(g_inter_flux3))
#==========================================interpolated data psd
#==========================================interpolated removed data psd
r_l_inter_fj0, r_l_inter_dft0 = PSD(14, pd.DataFrame(r_l_inter_time0), pd.DataFrame(r_l_inter_flux0))
r_l_inter_fj1, r_l_inter_dft1 = PSD(14, pd.DataFrame(r_l_inter_time1), pd.DataFrame(r_l_inter_flux1))
r_l_inter_fj2, r_l_inter_dft2 = PSD(14, pd.DataFrame(r_l_inter_time2), pd.DataFrame(r_l_inter_flux2))
r_l_inter_fj3, r_l_inter_dft3 = PSD(14, pd.DataFrame(r_l_inter_time3), pd.DataFrame(r_l_inter_flux3))

r_s_inter_fj0, r_s_inter_dft0 = PSD(14, pd.DataFrame(r_s_inter_time0), pd.DataFrame(r_s_inter_flux0))
r_s_inter_fj1, r_s_inter_dft1 = PSD(14, pd.DataFrame(r_s_inter_time1), pd.DataFrame(r_s_inter_flux1))
r_s_inter_fj2, r_s_inter_dft2 = PSD(14, pd.DataFrame(r_s_inter_time2), pd.DataFrame(r_s_inter_flux2))
r_s_inter_fj3, r_s_inter_dft3 = PSD(14, pd.DataFrame(r_s_inter_time3), pd.DataFrame(r_s_inter_flux3))

r_g_inter_fj0, r_g_inter_dft0 = PSD(14, pd.DataFrame(r_g_inter_time0), pd.DataFrame(r_g_inter_flux0))
r_g_inter_fj1, r_g_inter_dft1 = PSD(14, pd.DataFrame(r_g_inter_time1), pd.DataFrame(r_g_inter_flux1))
r_g_inter_fj2, r_g_inter_dft2 = PSD(14, pd.DataFrame(r_g_inter_time2), pd.DataFrame(r_g_inter_flux2))
r_g_inter_fj3, r_g_inter_dft3 = PSD(14, pd.DataFrame(r_g_inter_time3), pd.DataFrame(r_g_inter_flux3))

#==========================================interpolated removed data psd
if (removed_gauss_pdf == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_gauss_PDFplots.png', dpi=300) 
        

if (removed_spline_pdf == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_spline_PDFplots.png', dpi=300) 
        
if (removed_linear_pdf == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_linear_PDFplots.png', dpi=300) 

if (gauss_inter_pdf == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('gauss_inter_PDFplots.png', dpi=300) 

if (spline_inter_pdf == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('spline_inter_PDFplots.png', dpi=300) 
        
if (linear_inter_pdf == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('linear_inter_PDFplots.png', dpi=300) 
        
        
if (removed_gauss_psd == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_gauss_inter_PSDplots_thre%s.png' %(thre), dpi=300) 
        
        
if (removed_spline_psd == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_spline_inter_PSDplots_thre%s.png' %(thre), dpi=300) 

if (removed_linear_psd == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_linear_inter_PSDplots_thre%s.png' %(thre), dpi=300) 



if (gauss_inter_psd == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('gauss_interpolated_PSDplots.png', dpi=300) 


if (spline_inter_psd == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('spline_interpolated_PSDplots.png', dpi=300) 

if (linear_inter_psd == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('linear_interpolated_PSDplots.png', dpi=300) 
    

if (remove_gauss_interp_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_gauss_interp_LCplots.png', dpi=300)
        
if (remove_spline_interp_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_spline_interp_LCplots.png', dpi=300)



if (remove_linear_interp_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_linear_interp_LCplots.png', dpi=300)
        

if (gauss_inter_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('gauss_interp_LCplots.png', dpi=300)
        

if (spline_interp_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('spline_interp_LCplots.png', dpi=300)
        

if (linear_interp_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('linear_interp_LCplots.png', dpi=300)
    
    

if (removed_pdf_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_PDFplots_thre%s.png' %(thre), dpi=300) 


if (removed_psd_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_PSDplots_thre%s.png' %(thre), dpi=300) 


if (pdf_image_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('PDFplots.png', dpi=300) 

if (psd_image_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('PSDplots.png', dpi=300) 
        
    
if (removed_image_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('removed_LCplots_thre%s.png' %(thre), dpi=300)
        

if (origin_image_plot == True):
    plt.style.use('science')
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
        os.chdir(savepath)
        plt.savefig('LCplots.png', dpi=300)
