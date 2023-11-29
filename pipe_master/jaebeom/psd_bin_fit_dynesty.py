import numpy as np
import dynesty
import matplotlib.pyplot as plt
import pandas as pd
import os
from dynesty import plotting as dyplot
from scipy.stats import chisquare
import astropy.modeling
from scipy import stats



def PSD(time, flux, error, delt):
    dt = delt
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
        for i in range(0, len(freq), binsize):
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
#============================================================================== Simple PL     
    if (model == 'simplePL'):
        def spl(thata, x):   
            return thata[0]*x**thata[1]
        
        def log_likelihood_spl(theta):
            model = spl(theta, binned_freq)
            obs = binned_power            
            yerr = binned_error
            inv_sigma2 = 1.0 / (yerr**2 + model**2)
            return -0.5 * (np.sum((obs-model)**2 * inv_sigma2 - np.log(inv_sigma2)))
  
        def prior_transform_spl(utheta):
            uamp, ualpha = utheta
            amp = uamp*1e-7
            alpha = -ualpha
            return amp, alpha
        
        dsampler = dynesty.DynamicNestedSampler(log_likelihood_spl, prior_transform_spl, ndim=2,\
                                                bound='multi', sample='rwalk')
        dsampler.run_nested()
        dres = dsampler.results
        
        fit_params = np.array([np.median(dres['samples'][:, 0:1]),  np.median(dres['samples'][:, 1:2])])
        #estimated = dres['samples'][dres['logl'].argmax()]
        #fit_params = np.array([estimated[0], estimated[1]])
        
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = spl(fit_params, fit_modelx)
        
        chi = abs(chisquare(f_obs=np.log10(binned_power), f_exp=np.log10(spl(fit_params, binned_freq)))[0])
        fit_params_err = np.array([np.std(dres['samples'][:, 0:1]), np.std(dres['samples'][:, 1:2])])
        
        labels = ['amp', 'alpha']
        fig, axes = dyplot.cornerplot(dres, truths=fit_params, show_titles=True, \
                              title_kwargs={'y': 1.04}, labels=labels,\
                              fig=plt.subplots(2, 2, figsize=(10, 10)))
        plt.show()
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi
#============================================================================== Simple PL     
#============================================================================== Broken PL         
    if (model == 'brokenPL'):
        def bpl(theta, x):
            amp, x_break, a1, a2 = theta[0], theta[1], theta[2], theta[3]
            bp = astropy.modeling.powerlaws.BrokenPowerLaw1D(amplitude=amp, \
                                x_break=x_break, alpha_1=a1, alpha_2=a2)
            model = bp.evaluate(x, bp.amplitude[0], bp.x_break[0], bp.alpha_1[0], bp.alpha_2[0])
            return model
        
        def log_likelihood_bpl(theta):
            model = bpl(theta, binned_freq)
            obs = binned_power            
            yerr = binned_error
            inv_sigma2 = 1.0 / (yerr**2 + model**2)
            return -0.5 * (np.sum((obs-model)**2 * inv_sigma2 - np.log(inv_sigma2)))
  
        def prior_transform_bpl(utheta):
            uamp, uxbreak, ualpha1, ualpha2 = utheta
            amp = uamp*1e-6
            xbreak = np.random.uniform(0.001,0.01,1)[0]
            alpha1 = -ualpha1
            alpha2 = ualpha2
                    
            return amp, xbreak, alpha1, alpha2                    
            
 
        dsampler = dynesty.DynamicNestedSampler(log_likelihood_bpl, prior_transform_bpl, ndim=4,\
                                                bound='multi', sample='rwalk')
        dsampler.run_nested()
        dres = dsampler.results
        
        fit_params = np.array([np.median(dres['samples'][:, 0:1]),  np.median(dres['samples'][:, 1:2]),\
                               np.median(dres['samples'][:, 2:3]),  np.median(dres['samples'][:, 3:4])])
        #estimated = dres['samples'][dres['logl'].argmax()]
        #fit_params = np.array([estimated[0], estimated[1], estimated[2], estimated[3]])
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = bpl(fit_params, fit_modelx)
        
        chi = abs(chisquare(f_obs=np.log10(binned_power), f_exp=np.log10(bpl(fit_params, binned_freq)))[0])
        fit_params_err = np.array([np.std(dres['samples'][:, 0:1]), np.std(dres['samples'][:, 1:2]),\
                                  np.std(dres['samples'][:, 2:3]), np.std(dres['samples'][:, 3:4])])
        
        labels = ['amp', 'xbreak', 'alpha1', 'alpha2']
        fig, axes = dyplot.cornerplot(dres, truths=fit_params, show_titles=True, \
                              title_kwargs={'y': 1.04}, labels=labels,\
                              fig=plt.subplots(4, 4, figsize=(20, 20)))
        plt.show()
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi
#============================================================================== Broken PL         
#============================================================================== Curved PL         
    if (model == 'curvedPL'):
        def cpl(theta, x):
            amp, x_break, a1, a2, delta = theta[0], theta[1], theta[2], theta[3], theta[4]
            if (delta > 0.01):
                cp = astropy.modeling.powerlaws.SmoothlyBrokenPowerLaw1D(amplitude=abs(amp), \
                                x_break=x_break, alpha_1=a1, alpha_2=a2, delta=delta)
            if (delta < 0.01):
                cp = astropy.modeling.powerlaws.SmoothlyBrokenPowerLaw1D(amplitude=abs(amp), \
                                x_break=x_break, alpha_1=a1, alpha_2=a2, delta=np.random.uniform(0,5,1)[0])
            model = cp.evaluate(x, cp.amplitude[0], cp.x_break[0], cp.alpha_1[0], cp.alpha_2[0], cp.delta[0])
            return model
        
        def log_likelihood_cpl(theta):
            model = cpl(theta, binned_freq)
            obs = binned_power            
            yerr = binned_error
            inv_sigma2 = 1.0 / (yerr**2 + model**2)
            return -0.5 * (np.sum((obs-model)**2 * inv_sigma2 - np.log(inv_sigma2)))
  
        def prior_transform_cpl(utheta):
            uamp, uxbreak, ualpha1, ualpha2, udelta = utheta
            amp = uamp*1e-6
            xbreak = np.random.uniform(0.001,0.01,1)[0]
            alpha1 = -ualpha1
            alpha2 = ualpha2
            delta = udelta
            
            return amp, xbreak, alpha1, alpha2, delta             
            
 
        dsampler = dynesty.DynamicNestedSampler(log_likelihood_cpl, prior_transform_cpl, ndim=5,\
                                                bound='multi', sample='rwalk')
        dsampler.run_nested()
        dres = dsampler.results
        
        fit_params = np.array([np.median(dres['samples'][:, 0:1]),  np.median(dres['samples'][:, 1:2]),\
                               np.median(dres['samples'][:, 2:3]),  np.median(dres['samples'][:, 3:4]),\
                               np.median(dres['samples'][:, 4:5])])
        #estimated = dres['samples'][dres['logl'].argmax()]
        #fit_params = np.array([estimated[0], estimated[1], estimated[2], estimated[3], estimated[4]])
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = cpl(fit_params, fit_modelx)
        
        chi = abs(chisquare(f_obs=np.log10(binned_power), f_exp=np.log10(cpl(fit_params, binned_freq)))[0])
        fit_params_err = np.array([np.std(dres['samples'][:, 0:1]), np.std(dres['samples'][:, 1:2]),\
                                  np.std(dres['samples'][:, 2:3]), np.std(dres['samples'][:, 3:4]),\
                                      np.std(dres['samples'][:, 4:5])])
        
        labels = ['amp', 'xbreak', 'alpha1', 'alpha2', 'delta']
        fig, axes = dyplot.cornerplot(dres, truths=fit_params, show_titles=True, \
                              title_kwargs={'y': 1.04}, labels=labels,\
                              fig=plt.subplots(5, 5, figsize=(20, 20)))
        plt.show()
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi
#============================================================================== Curved PL        
        
    
f = pd.read_csv('7143_lightcurve.dat', delim_whitespace=True, header=None)

time = f[0].values
flux = f[1].values
error = f[2].values

binsize = 20
freq, power, power_error = PSD(time, flux, error, 14)   

binned_freq_s, binned_power_s, fit_params_s, fit_params_err_s, fit_modelx_s, fit_modely_s, chi_s\
    = bin_fit_PSD(freq, power, power_error, binsize, 'simplePL')
    
binned_freq_b, binned_power_b, fit_params_b, fit_params_err_b, fit_modelx_b, fit_modely_b, chi_b\
    = bin_fit_PSD(freq, power, power_error, binsize, 'brokenPL')

binned_freq_c, binned_power_c, fit_params_c, fit_params_err_c, fit_modelx_c, fit_modely_c, chi_c\
    = bin_fit_PSD(freq, power, power_error, binsize, 'curvedPL')

print ('\n')
print (fit_params_s)
print (fit_params_b)
print (fit_params_c)
print ('\n')

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
plt.title('curved PL', fontsize=15)
plt.legend(fontsize=15)
