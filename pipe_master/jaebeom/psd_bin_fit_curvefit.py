import numpy as np
import pandas as pd
import os
import astropy.modeling
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scienceplots

f = pd.read_csv('7143_lightcurve.dat', delim_whitespace=True, header=None)

time = f[0].values
flux = f[1].values
error = f[2].values


def PSD(time, flux, error, delt):
    dt = 14
    fj = []
    for i in range(len(time)):
        fj.append((i)/(len(time)*dt))

    xbar = flux - np.mean(flux)
    err = error - np.mean(error)
    time = time - time[0]
    dft_list, dfterr_list = [], []
    
    for i in range(len(time)):
        dft = (np.sum(xbar*np.cos(2*np.pi*fj[i]*time)))**2 + (np.sum(xbar*np.sin(2*np.pi*fj[i]*time)))**2
        dfterr = (np.sum(err*np.cos(2*np.pi*fj[i]*time)))**2 + (np.sum(err*np.sin(2*np.pi*fj[i]*time)))**2
        
        dft_list.append(dft)
        dfterr_list.append(dfterr)
        
    return fj, dft_list, dfterr_list


def bin_fit_PSD(freq, power, power_error, binning, model):
    
    def make_mean_bin_psd(binsize, freq, power, power_error):
        binned_freq, binned_power, binned_error = [],[],[]
        for i in range(0, len(f), binsize):
            j = i + binsize - 1
            logdft = np.log10(power)
            logdft_err = np.log10(power_error)
            
            binned_freq.append(np.nanmean(freq[i:j]))
            binned_power.append(np.nanmean(logdft[i:j])+0.25068)
            binned_error.append(np.nanmean(logdft_err[i:j])+0.25068)
            
        binned_freq = np.array(binned_freq)
        binned_power = 10**np.array(binned_power)
        binned_error = 10**np.array(binned_error)
        
        return binned_freq, binned_power, binned_error
    
    binned_freq, binned_power, binned_error = make_mean_bin_psd(binning, freq, power, power_error)

    if (model == 'simplePL'):
        def spl(x, a, b):
            return a*x**b
        
        fit_params, fit_params_err = curve_fit(spl, binned_freq, binned_power)
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = spl(fit_modelx, fit_params[0], fit_params[1])
            
        chi = abs(chisquare(f_obs=np.log10(binned_power), f_exp=np.log10(spl(binned_freq, fit_params[0], fit_params[1])))[0])
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi
    
    if (model == 'brokenPL'):
        def bpl(x, amp, x_break, a1, a2):
            bp = astropy.modeling.powerlaws.BrokenPowerLaw1D(amplitude=amp, \
                                x_break=x_break, alpha_1=a1, alpha_2=a2)
            model = bp.evaluate(x, bp.amplitude[0], bp.x_break[0], bp.alpha_1[0], bp.alpha_2[0])
            
            return model
        
        fit_params, fit_params_err = curve_fit(bpl, binned_freq, binned_power, \
                                     bounds=((-np.inf, min(binned_freq), -np.inf, -np.inf),\
                                             (np.inf, max(binned_freq), np.inf, np.inf)))
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = bpl(fit_modelx, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
        
        chi = abs(chisquare(f_obs=np.log10(binned_power), \
                f_exp=np.log10(bpl(binned_freq, fit_params[0], fit_params[1], fit_params[2], fit_params[3])))[0])
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi
    
    if (model == 'curvedPL'):
        def spl(x, amp, x_break, a1, a2, delta):
            bp = astropy.modeling.powerlaws.SmoothlyBrokenPowerLaw1D(amplitude=abs(amp), \
                                x_break=x_break, alpha_1=a1, alpha_2=a2, delta=delta)
            model = bp.evaluate(x, bp.amplitude[0], bp.x_break[0], bp.alpha_1[0], bp.alpha_2[0], bp.delta[0])
            return model
        
        fit_params, fit_params_err = curve_fit(spl, binned_freq, binned_power, \
                                     bounds=((-np.inf, min(binned_freq), -np.inf, -np.inf, -np.inf),\
                                             (np.inf, max(binned_freq), np.inf, np.inf, np.inf)))
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = spl(fit_modelx, fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4])
        chi = abs(chisquare(f_obs=np.log10(binned_power), \
                f_exp=np.log10(spl(binned_freq, fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4])))[0])
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi
    
binsize = 20
    
freq, power, power_error = PSD(time, flux, error, 14)

binned_freq_s, binned_power_s, fit_params_s, fit_params_err_s, fit_modelx_s, fit_modely_s, chi_s\
    = bin_fit_PSD(freq, power, power_error, binsize, 'simplePL')
    
binned_freq_b, binned_power_b, fit_params_b, fit_params_err_b, fit_modelx_b, fit_modely_b, chi_b\
    = bin_fit_PSD(freq, power, power_error, binsize, 'brokenPL')
    
binned_freq_c, binned_power_c, fit_params_c, fit_params_err_c, fit_modelx_c, fit_modely_c, chi_c\
    = bin_fit_PSD(freq, power, power_error, binsize, 'curvedPL')
    
print (fit_params_s)

plt.style.use('science')
plt.figure(figsize=(15,5), dpi=300)

plt.subplot(131)
plt.scatter(np.log10(freq), np.log10(power), c='black')
plt.scatter(np.log10(binned_freq_s), np.log10(binned_power_s), c='green')
plt.plot(np.log10(fit_modelx_s), np.log10(fit_modely_s), c='red', label='$\chi^2$ = %.4lf' %(chi_s))
plt.xlabel('Frequency', fontsize=15)
plt.ylabel('Power', fontsize=15)
plt.title('Simple PL', fontsize=15)
plt.legend(fontsize=15)

plt.subplot(132)
plt.scatter(np.log10(freq), np.log10(power), c='black')
plt.scatter(np.log10(binned_freq_b), np.log10(binned_power_b), c='green')
plt.plot(np.log10(fit_modelx_b), np.log10(fit_modely_b), c='red', label='$\chi^2$ = %.4lf' %(chi_b))
plt.xlabel('Frequency', fontsize=15)
plt.ylabel('Power', fontsize=15)
plt.title('Broken PL', fontsize=15)
plt.legend(fontsize=15)


plt.subplot(133)
plt.scatter(np.log10(freq), np.log10(power), c='black')
plt.scatter(np.log10(binned_freq_c), np.log10(binned_power_c), c='green')
plt.plot(np.log10(fit_modelx_c), np.log10(fit_modely_c), c='red', label='$\chi^2$ = %.4lf' %(chi_c))
plt.xlabel('Frequency', fontsize=15)
plt.ylabel('Power', fontsize=15)
plt.title('Curved PL', fontsize=15)
plt.legend(fontsize=15)

#plt.savefig('curfit_results.png', dpi=300)
