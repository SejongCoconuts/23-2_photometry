import numpy as np
import emcee
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare
import astropy.modeling
import corner



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

def bin_fit_PSD(freq, power, power_error, binning, model, burnin):
    
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
        
        def log_likelihood_spl(theta, x, y, yerr):
            model = spl(theta, x)
            return -0.5 * np.sum(((y - model) / yerr)**2)

        def run_mcmc(x, y, yerr, nwalkers=100, ndim=2, nsteps=3000):
            initial_theta = np.array([1e-12, -1e-01])
            initial_positions = [initial_theta + 1e-12 * np.random.randn(ndim) for _ in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_spl, args=(x, y, yerr))
            sampler.run_mcmc(initial_positions, nsteps, progress=True)
            return sampler
        
        sampler = run_mcmc(binned_freq, binned_power, binned_error)
        flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
        
        fit_params = np.array([np.mean(flat_samples[:,0]), np.mean(flat_samples[:,1])])
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = spl(fit_params, fit_modelx)
            
        chi = abs(chisquare(f_obs=np.log10(binned_power), f_exp=np.log10(spl(fit_params, binned_freq)))[0])
        fit_params_err = np.array([np.std(flat_samples[:,0]), np.std(np.mean(flat_samples[:,1]))])
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi, flat_samples
#============================================================================== Simple PL
#============================================================================== Broken PL
    if (model == 'brokenPL'):
        def bpl(theta, x):
            amp, x_break, a1, a2 = theta[0], theta[1], theta[2], theta[3]
            bp = astropy.modeling.powerlaws.BrokenPowerLaw1D(amplitude=amp, \
                                x_break=x_break, alpha_1=a1, alpha_2=a2)
            model = bp.evaluate(x, bp.amplitude[0], bp.x_break[0], bp.alpha_1[0], bp.alpha_2[0])
            return model
      
        def log_likelihood_bpl(theta, x, y, yerr):
            model = bpl(theta, x)
            if any(np.isnan(model)) or any(model <= 0):
                return -np.inf
            return -0.5 * np.sum(((y - model) / yerr)**2)

        def log_prior(theta):
            amp, x_break, a1, a2 = np.log(theta[0]), np.log(theta[1]), np.log(theta[2]), np.log(theta[3])
            if (amp == 'nan') or (x_break == 'nan') or (a1 == 'nan') or (a2 == 'nan'):
                return -np.inf    
            #if ((1e-13 < amp < 1e-12) and (-5 < x_break < -1) and (0.05 < a1 < 5) and (0.05 < a2 < 5)):
            return 0

        def log_posterior(theta, x, y, yerr):
            prior = log_prior(theta)
            if not np.isfinite(prior):
                return -np.inf
            return prior + log_likelihood_bpl(theta, x, y, yerr)

        def run_mcmc(x, y, yerr, nwalkers=100, ndim=4, nsteps=3000):
            initial_theta = np.array([1e-12, 1e-2, 1e-01, 2])
            initial_positions = [initial_theta + 1e-12 * np.random.randn(ndim) for _ in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_bpl, args=(x, y, yerr))
            sampler.run_mcmc(initial_positions, nsteps, progress=True)
            return sampler
        
        sampler = run_mcmc(binned_freq, binned_power, binned_error)
        flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
        
        fit_params = np.array([np.mean(flat_samples[:,0]), np.mean(flat_samples[:,1]), \
                               np.mean(flat_samples[:,2]), np.mean(flat_samples[:,3])])
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = bpl(fit_params, fit_modelx)
            
        chi = abs(chisquare(f_obs=np.log10(binned_power), f_exp=np.log10(bpl(fit_params, binned_freq)))[0])
        fit_params_err = np.array([np.std(flat_samples[:,0]), np.std(np.mean(flat_samples[:,1]))])
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi, flat_samples
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
      
        def log_likelihood_cpl(theta, x, y, yerr):
            model = cpl(theta, x)
            if any(np.isnan(model)) or any(model <= 0):
                return -np.inf
            return -0.5 * np.sum(((y - model) / yerr)**2)

        def log_prior(theta):
            amp, x_break, a1, a2, delta = theta[0], theta[1], theta[2], theta[3], theta[4]
            delta = np.random.uniform(0,5,1)[0]
            if (delta > 0.01):
            #if ((1e-15 < amp < 1e-12) and (-5 < x_break < -1) and (0.05 < a1 < 5) and (0.05 < a2 < 5) and (delta > 0.01)):
                return 0
            return -np.inf

        def log_posterior(theta, x, y, yerr):
            prior = log_prior(theta)
            if not np.isfinite(prior):
                return -np.inf
            return prior + log_likelihood_bpl(theta, x, y, yerr)

        def run_mcmc(x, y, yerr, nwalkers=100, ndim=5, nsteps=3000):
            initial_theta = np.array([1e-12, 1e-2, 1e-01, 2, 5])
            initial_positions = [initial_theta + 1e-12 * np.random.randn(ndim) for _ in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_cpl, args=(x, y, yerr))
            sampler.run_mcmc(initial_positions, nsteps, progress=True)
            return sampler
        
        sampler = run_mcmc(binned_freq, binned_power, binned_error)
        flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
        
        fit_params = np.array([np.mean(flat_samples[:,0]), np.mean(flat_samples[:,1]), \
                               np.mean(flat_samples[:,2]), np.mean(flat_samples[:,3]), np.mean(flat_samples[:,4])])
        
        fit_modelx = np.linspace(min(binned_freq), max(binned_freq), 100)
        fit_modely = cpl(fit_params, fit_modelx)
            
        chi = abs(chisquare(f_obs=np.log10(binned_power), f_exp=np.log10(cpl(fit_params, binned_freq)))[0])
        fit_params_err = np.array([np.std(flat_samples[:,0]), np.std(np.mean(flat_samples[:,1]))])
        
        return binned_freq, binned_power, fit_params, fit_params_err, fit_modelx, fit_modely, chi, flat_samples
#============================================================================== Curved PL     

f = pd.read_csv('7143_lightcurve.dat', delim_whitespace=True, header=None)

time = f[0].values
flux = f[1].values
error = f[2].values

binsize = 20
burnin = 1000
freq, power, power_error = PSD(time, flux, error, 14)

binned_freq_s, binned_power_s, fit_params_s, fit_params_err, fit_modelx_s, fit_modely_s, chi_s, flat_samples_s\
    = bin_fit_PSD(freq, power, power_error, binsize, 'simplePL', burnin)
    
binned_freq_b, binned_power_b, fit_params_b, fit_params_err_b, fit_modelx_b, fit_modely_b, chi_b, flat_samples_b\
    = bin_fit_PSD(freq, power, power_error, binsize, 'brokenPL', burnin)

binned_freq_c, binned_power_c, fit_params_c, fit_params_err_c, fit_modelx_c, fit_modely_c, chi_c, flat_samples_c\
    = bin_fit_PSD(freq, power, power_error, binsize, 'curvedPL', burnin)

print ('\n')    
print (fit_params_s)
print (fit_params_b)
print (fit_params_c)

#plt.style.use('science')
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

plt.savefig('bin_fit_psd_emcee.png', dpi=300)
plt.show()

plt.figure(figsize=(15,8), dpi=300)
plt.subplot(211)
xx = np.linspace(0, len(flat_samples_s), len(flat_samples_s))
plt.plot(xx, flat_samples_s[:,0], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_s[:,0]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('amp', fontsize=15)

plt.subplot(212)
plt.plot(xx, flat_samples_s[:,1], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_s[:,1]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('alpha', fontsize=15)

plt.savefig('psd_fit_simple_emcee_chain.png', dpi=300)
plt.show()

plt.figure(figsize=(10,8), dpi=300)
plt.subplot(411)
xx = np.linspace(0, len(flat_samples_b), len(flat_samples_b))
plt.plot(xx, flat_samples_b[:,0], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_b[:,0]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('amp', fontsize=15)

plt.subplot(412)
plt.plot(xx, flat_samples_b[:,1], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_b[:,1]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('x_break', fontsize=15)

plt.subplot(413)
plt.plot(xx, flat_samples_b[:,2], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_b[:,2]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('alpha1', fontsize=15)

plt.subplot(414)
plt.plot(xx, flat_samples_b[:,3], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_b[:,3]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('alpha2', fontsize=15)

plt.savefig('psd_fit_broken_emcee_chain.png', dpi=300)
plt.show()

plt.figure(figsize=(10,8), dpi=300)
plt.subplot(511)
xx = np.linspace(0, len(flat_samples_c), len(flat_samples_c))
plt.plot(xx, flat_samples_c[:,0], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_b[:,0]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('amp', fontsize=15)

plt.subplot(512)
plt.plot(xx, flat_samples_c[:,1], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_c[:,1]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('x_break', fontsize=15)

plt.subplot(513)
plt.plot(xx, flat_samples_c[:,2], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_c[:,2]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('alpha1', fontsize=15)

plt.subplot(514)
plt.plot(xx, flat_samples_c[:,3], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_c[:,3]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('alpha2', fontsize=15)

plt.subplot(515)
plt.plot(xx, flat_samples_c[:,3], c='black', alpha=0.8)
plt.axhline(np.mean(flat_samples_c[:,4]), c='blue')
plt.xlabel('Step Number', fontsize=15)
plt.ylabel('delta', fontsize=15)

plt.savefig('psd_fit_curved_emcee_chain.png', dpi=300)
plt.show()
