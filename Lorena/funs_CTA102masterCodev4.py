#### All the functions used in CTA102_mastercodev4
### by: Lorena Amaya Ruiz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import linregress
import astropy.modeling
from scipy.stats import gamma, lognorm
from scipy import interpolate
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

############# PSD Calculation #############

def psd_cal(JULDATE, FLUX, FLUXERROR):
    
    t_data = JULDATE
    fl_data = FLUX
    fl_data_err = FLUXERROR
    del_t = 14
    xbar = fl_data - np.mean(fl_data)
    xbar_err = fl_data_err - np.mean(fl_data_err)
    n_data = len(FLUX)
    # Calculate the power spectrum density
    frqj = []
    psd_ = []

    for ind in range(n_data - 1):
        fqj = ind / (n_data * del_t)
        frqj.append(fqj)
        cos = np.cos((2*np.pi*fqj*t_data))
        sin = np.sin((2*np.pi*fqj*t_data))
        dft = (np.sum(xbar * cos)) **2 + (np.sum(xbar * sin)) **2
        psd_.append(dft)

    freq = np.array(np.log10(frqj))
    psd = np.array(np.log10(psd_))

    freq = freq[1:]
    psd = psd[1:]

    # BINNED
    binsize = 5
    fmean, rpsd, err_psd = [], [], []
    for i in range(0, n_data, binsize):
        j = i + binsize - 1
        logdft = psd

        fmean.append(np.mean(10**freq[i:j]))
        rpsd.append(np.mean(logdft[i:j])+0.25068)
    #     rpsd.append(np.mean(logdft[i:j]))
        err_psd.append(np.std(logdft[i:j]))

    fmean = np.array(fmean)
    rpsd = np.array(rpsd)
    err_psd = np.array(err_psd)
    fmean = fmean[:-1]
    rpsd = rpsd[:-1]
    err_psd = err_psd[:-1]
   
    return psd, freq, fmean, rpsd, err_psd  
    
    
    
############# Fitting Calculation #############

def models_fit(fmean, rpsd, err_psd, initials_PL, initials_BPL, initials_CPL):
    
    # Different models fitting:

    # Linear regression:

    slope, intercept, r_value, p_value, std_err = linregress(np.log10(fmean), rpsd) # Fit a linear regression model
    fitted_line = slope * np.log10(fmean) + intercept # Calculate the fitted line
    residuals = rpsd - fitted_line # Calculate residuals
    chi_squared_linear = np.sum((residuals)**2 / abs(fitted_line)) # Calculate chi-squared
    print('LINEAR REGRESSION:')
    print('------------------------')
    print("Fitted Linear Regression Parameters:")
    print("Slope (a) =", slope)
    print("Intercept (b) =", intercept)
    print(f'Chi-squared: {chi_squared_linear:.10f}')
    # print('Residuals', residuals)


    # Power Law:


    def power_law(fmean, alpha, b): # Define the broken power-law model function
        information = astropy.modeling.powerlaws.PowerLaw1D(amplitude = abs(alpha), x_0=1, alpha=b)
        return information.evaluate(fmean, information.amplitude[0], information.x_0[0], information.alpha[0])

    params, covariance = curve_fit(power_law, fmean, 10**rpsd, p0 = initials_PL) # Fit the power-law model to the binned data
    a_fit, b_fit = params # Extract the fitted parameters
    fitted_curve = power_law(fmean, a_fit, b_fit)
    residuals = rpsd - np.log10(fitted_curve)
    chi_squared_powerlaw = np.sum((residuals)**2 / abs(np.log10(fitted_curve))) # Calculate the chi-square value
    print ('\n')
    print('POWER LAW:')
    print('------------------------')
    print(f'Best-fit parameters: a = {a_fit}, b = {b_fit:.2f}')
    print(f'Chi-squared: {chi_squared_powerlaw:.10f}')


    # Broken Power Law:

    def Bpower_law(fmean, amplitude, x_break, alpha1, alpha2): # Define the broken power-law model function
        information = astropy.modeling.powerlaws.BrokenPowerLaw1D(amplitude = abs(amplitude), x_break=x_break, alpha_1=alpha1, alpha_2=alpha2)
        return information.evaluate(fmean, information.amplitude[0], information.x_break[0], information.alpha_1[0], information.alpha_2[0])

    params, covariance = curve_fit(Bpower_law, fmean, 10**rpsd, p0 = initials_BPL) # Fit the broken power-law model to the binned data
    amplitude = params[0]
    x_break = params[1]
    alpha1 = params[2]
    alpha2 = params[3]
    fitted_bpowerlaw = Bpower_law(fmean, amplitude, x_break, alpha1, alpha2)
    residuals = rpsd - np.log10(fitted_bpowerlaw)
    chi_squared_bpowerlaw = np.sum((residuals**2 / abs(np.log10(fitted_bpowerlaw)))) # Calculate the chi-square value
    print ('\n')
    print('BROKEN POWER LAW:')
    print('------------------------')
    print(f'Best-fit parameters: alpha1 = {alpha1:.5f}, alpha2 = {alpha2:.5f}, Amplitude = {amplitude: .3e}, x_break = {x_break:.5f}')
    print(f'Chi-squared: {chi_squared_bpowerlaw:.10f}')


    # Curved Broken Power Law:

#     def CBpower_law(fmean, amplitude, alpha, xcutoff): # Define the curved power-law model function
#         information = astropy.modeling.powerlaws.ExponentialCutoffPowerLaw1D(amplitude = amplitude, x_0=1, alpha=alpha, x_cutoff= xcutoff)
#         return information.evaluate(fmean, information.amplitude[0], information.x_0[0], information.alpha[0], information.x_cutoff[0])

    # def CBpower_law(fmean, amplitude, alpha, beta, xo): # Define the broken power-law model function
    #     information = astropy.modeling.powerlaws.LogParabola1D(amplitude = amplitude, x_0=xo, alpha=alpha, beta = beta)
    #     return information.evaluate(fmean, information.amplitude[0], information.x_0[0], information.alpha[0], information.beta[0])

    def CBpower_law(fmean, alpha1, alpha2, b1):
                return 10**(alpha1*np.log10(fmean)**2 + alpha2*np.log10(fmean) + np.log10(b1))
        
    params, covariance = curve_fit(CBpower_law, fmean, 10**rpsd, p0 = initials_CPL) # Fit the broken power-law model to the binned data
    # p0 =[amplitude, x_break, alpha1, alpha2, 1.5], p0 =[0.03, 0, 1, 1, 1]
#     C_amplitude = params[0]
    # Cx_0 = params[1]
#     C_alpha = params[1]
    # C_beta = params[3]
#     C_xcutoff = params[2]
    # C_delta = params[4]
    C_alpha1 = params[0]
    C_alpha2 = params[1]
    C_b1 = params[2]
#     fitted_Cbpowerlaw = CBpower_law(fmean, C_amplitude, C_alpha, C_xcutoff)
    fitted_Cbpowerlaw = CBpower_law(fmean, C_alpha1, C_alpha2, C_b1)
    residuals = rpsd - np.log10(fitted_Cbpowerlaw)
    chi_squared_Cbpowerlaw = np.sum((residuals)**2 / abs(np.log10(fitted_Cbpowerlaw))) # Calculate the chi-square value
    print ('\n')
    print('CURVED BROKEN POWER LAW:')
    print('------------------------')
#     print(f'Best-fit parameters: alpha = {C_alpha}, xcutoff = {C_xcutoff}, Amplitude = {C_amplitude}')
    print(f'Best-fit parameters: alpha1 = {C_alpha1}, Alpha2 = {C_alpha2}, Beta = {C_b1}')

    print(f'Chi-squared: {chi_squared_Cbpowerlaw:.5f}')

    fig, g = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('white')
    # g.plot(freq, psd,'b-o',linewidth=1, markersize = 3, label = 'PSD', alpha = 0.2)
    g.errorbar(np.log10(fmean), rpsd, yerr=err_psd / 5, fmt='o', color='black', markersize=3, label = 'PSD binned')
    g.plot(np.log10(fmean), fitted_line, 'm-', label=f'Fitted Linear Regression: Slope = {a_fit}, Intercept = {b_fit:.2f}, Chi-squared: {chi_squared_linear:.10f}')
    g.plot(np.log10(fmean), np.log10(fitted_curve), 'g-', label=f'Power Law Fit: alpha={a_fit: .3e}, beta={b_fit:.3e}, Chi-squared: {chi_squared_powerlaw:.10f}')
    g.plot(np.log10(fmean), np.log10(fitted_bpowerlaw), 'r-', label=f'Broken Power Law: alpha1={alpha1:.2f}, alpha2={alpha2:.2f}, Amplitude={amplitude: .3e}, x_break = {x_break:.5f}, Chi-squared: {chi_squared_bpowerlaw:.10f}')
#     g.plot(np.log10(fmean), np.log10(fitted_Cbpowerlaw), '-', c = 'orange', label=f'Curved Broken Power Law: alpha = {C_alpha: .3f}, xcutoff = {C_xcutoff: .3f}, Amplitude = {C_amplitude: .3e}, Chi-squared: {chi_squared_Cbpowerlaw:.5f}')
    g.plot(np.log10(fmean), np.log10(fitted_Cbpowerlaw), '-', c = 'orange', label=f'Curved Broken Power Law: alpha1 = {C_alpha1: .3f}, alpha2 = {C_alpha2: .3f}, beta = {C_b1: .3e}, Chi-squared: {chi_squared_Cbpowerlaw:.5f}')

    g.set_ylabel('log(PSD)',fontsize=15)
    g.set_xlabel('log(frequency)',fontsize=15)
    plt.title('Power Spectrum Density of CTA 102',fontsize=15)
    g.legend(loc ='lower center', bbox_to_anchor=(0.5, -.4), fontsize = 15)
    for spine in ['top', 'right','bottom','left']:
        g.spines[spine].set_linewidth(2)
    g.tick_params(labelsize=10,length=3,width=2)
    
    return
    
############# PDF Calculation ############# 

def pdf_cal(FLUX):
    data = FLUX
    # Fit a gamma distribution to the data and estimate parameters
    kappa, loc, theta = gamma.fit(data, fscale=1)
    print('Kappa, loc, theta', kappa, loc, theta)
    # Parameters
#     print("kappa:", kappa) # shape of the gamma distribution's probability density function
#     print("theta:", theta) # influences the scale or spread of the gamma distribution.
    # Generate PDF
    data_pdf = np.linspace(0, data.max(), 1000)
    pdf_light = gamma.pdf(data_pdf, kappa, loc, theta)   
    # Fit a log-normal distribution to the data and estimate parameters
    lnsig, loc, scale = lognorm.fit(data, floc=0)
    # Calculate the mean (lnmu)
    lnmu = loc - (lnsig**2) / 2
    # Parameters
#     print("lnmu:", lnmu) # corresponds to the mean of the natural logarithm of the data
#     print("lnsig:", lnsig) # represents the standard deviation of the natural logarithm of the data
    # Generate PDF
    data_pdf2 = np.linspace(0, data.max(), 1000)
    pdf_light2 = lognorm.pdf(data_pdf2, lnsig, loc, scale)
    return kappa, theta, lnmu, lnsig, data_pdf, data_pdf2, pdf_light, pdf_light2
    
############# Function to remove data #############

def rem_cal(data, CTAdata):
    fig, g9 = plt.subplots(figsize=(10, 5))
    fig.set_facecolor('white')
    g9.plot(data['Time'], data['Flux'],'b-o',linewidth=1, markersize = 3, label = 'Simulated', alpha = 0.3)
    limits = np.linspace (0.5, 0.05, 3)
    vkr = []
    vtr = []
    vlnr = []
    vlsr = []
    vdpfl = []
    vdpfl2 = []
    vpdfl1 = []
    vpdfl2 = []
    limit_v = []
    data_r = []
    dftr_v = []
    fjr_v = []
    fmeanr_v = []
    rpsdr_v = []
    errpsdr_v = []

    for ll in range(len(limits)):

        limit1 = limits[ll]*np.max(CTAdata['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'])
        filt1_ = data[data['Flux'] > limit1]
        data_not_limit1 = data[data['Flux'] <= limit1]
        num_random_points = 10
        random_points = data_not_limit1.sample(n=num_random_points, random_state=42)
        if len(filt1_) >= 3:
            num_random_points2 = 2
            random_points2 = filt1_.sample(n=num_random_points2, random_state=42)
            filt1_mod = filt1_.drop(random_points2.index)
            filt1 = pd.concat([filt1_mod, random_points])
        else:

            filt1 = pd.concat([filt1_, random_points])
        
        filt1 = filt1.sort_values(by='Time', ascending=True)
        g9.plot(filt1['Time'], filt1['Flux'],'o',linewidth=1, markersize = 3, label = 'Missing data limit = {:.3e}'.format(limit1))

        g9.axhline(y = limit1, linestyle='--', label = 'Limit = {:.3e}'.format(limit1))
        g9.text(max(data['Time']) + 300, limit1, f'{limit1:.3e}', ha='left', va='center', color='blue', fontsize=12)
        

        dftr, fjr, fmeanr, rpsdr, err_psdr = psd_cal(filt1['Time'], filt1['Flux'], CTAdata['Photon Flux Error(photons cm-2 s-1)'])
        add_datapsd(len(filt1['Time']), 'Sim:{:.3e}'.format(limit1), dftr, fjr, rpsdr, fmeanr, err_psdr)
    
        kappa, theta, lnmu, lnsig, data_pdfl, data_pdf2l, pdf_lightl, pdf_light2l = pdf_cal(filt1['Flux'])
        add_datapdfs(len(filt1['Time']), 'Sim:{:.3e}'.format(limit1), kappa, theta, lnmu, lnsig, data_pdfl, data_pdf2l, pdf_lightl, pdf_light2l)
        vkr.append(kappa)
        vtr.append(theta)
        vlnr.append(lnmu)
        vlsr.append(lnsig)
        vdpfl.append(data_pdfl)
        vdpfl2.append(data_pdf2l)
        vpdfl1.append(pdf_lightl)
        vpdfl2.append(pdf_light2l)
        limit_v.append(limit1)
        data_r.append(filt1)
        dftr_v.append(dftr)
        fjr_v.append(fjr)
        fmeanr_v.append(fmeanr)
        rpsdr_v.append(rpsdr)
        errpsdr_v.append(err_psdr)
    kappa, theta, lnmu, lnsig, data_pdfl, data_pdf2l, pdf_lightl, pdf_light2l = pdf_cal(data['Flux'])
    add_datapdfs(len(data['Time']), 'Sim without lim', kappa, theta, lnmu, lnsig, data_pdfl, data_pdf2l, pdf_lightl, pdf_light2l)
    dftr, fjr, fmeanr, rpsdr, err_psdr = psd_cal(data['Time'], data['Flux'], CTAdata['Photon Flux Error(photons cm-2 s-1)'])
    add_datapsd(len(data['Time']), 'Sim without lim', dftr, fjr, rpsdr, fmeanr, err_psdr)
    vkr.append(kappa)
    vtr.append(theta)
    vlnr.append(lnmu)
    vlsr.append(lnsig)
    vdpfl.append(data_pdfl)
    vdpfl2.append(data_pdf2l)
    vpdfl1.append(pdf_lightl)
    vpdfl2.append(pdf_light2l)
    limit_v.append(limit1)
    data_r.append(data)
    dftr_v.append(dftr)
    fjr_v.append(fjr)
    fmeanr_v.append(fmeanr)
    rpsdr_v.append(rpsdr)
    errpsdr_v.append(err_psdr)
    g9.set_xlabel('Julian Date',fontsize=15)
    g9.set_ylabel('Photon Flux',fontsize=15)
    g9.legend(loc ='center left', bbox_to_anchor=(-0.6,0.5))
    g9.set_title('Light Curve of CTA 102',fontsize=15)
    for spine in ['top', 'right','bottom','left']:
        g9.spines[spine].set_linewidth(2)
    g9.tick_params(labelsize=10,length=3,width=2)
    return vkr, vtr, vlnr, vlsr,vdpfl, vdpfl2, vpdfl1, vpdfl2, limit_v, data_r, dftr_v, fjr_v, fmeanr_v, rpsdr_v, errpsdr_v

############# Function to interpolate in the tree different ways #############

def inter(timex,fluxy):
    
    time_interp = np.arange(min(timex), max(timex), 1)

    inter_func = interpolate.interp1d(timex, fluxy, kind='linear')
    flux_interp = inter_func(time_interp)

    s_inter = CubicSpline(timex, fluxy)
    flux_interp2 = s_inter(time_interp)


    # Create a Gaussian Process Regressor with an RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # Fit the Gaussian process to the data
    gp.fit(timex[:, np.newaxis], fluxy)
    # Perform the Gaussian process interpolation

    flux_mean, flux_std = gp.predict(time_interp[:, np.newaxis], return_std=True)
    return time_interp, flux_interp, flux_interp2, flux_mean, flux_std


############# Plotting ############# 

def plot_psd(psd, freq, fmean, rpsd, err_psd, title2, g):
    g.plot(freq, psd,'b-o',linewidth=1, markersize = 3, label = 'PSD', alpha = 0.2)
    plt.errorbar(np.log10(fmean[1:]), rpsd[1:], yerr=err_psd[1:], fmt='o', color='black', markersize=3, label = 'PSD binned')
    g.set_ylabel('log(PSD)',fontsize=15)
    g.set_xlabel('log(frequency)',fontsize=15)
    g.set_title(title2,fontsize=15)
    g.legend(loc ='best')
    for spine in ['top', 'right','bottom','left']:
        g.spines[spine].set_linewidth(2)
    g.tick_params(labelsize=10,length=3,width=2)
    return
    
def plot_pdf(data_pdf, data_pdf2, pdf_light, pdf_light2, title, data, g):
    g.plot(data_pdf, pdf_light, 'k-', lw=2, label='gamma PDF')
    g.plot(data_pdf2, pdf_light2, 'r-', lw=2, label='log-normal distribution PDF')
    g.plot(data_pdf, pdf_light2 + pdf_light, 'g-', lw=2, label='Mix model PDF')
    g.hist(data, alpha = 0.5, bins=30, density=True, color = 'b', label=r'Photon Flux', edgecolor = 'black')
    g.set_ylabel('Probability Density Fuction',fontsize=15)
    plt.title(title,fontsize=15)
    g.legend(loc ='best')
    for spine in ['top', 'right','bottom','left']:
        g.spines[spine].set_linewidth(2)
    g.tick_params(labelsize=10,length=3,width=2)
    return    
    
    
def plotthing(timex, fluxy, time_interp, flux_interp, title, title2, title3, g):
    
    g.plot(timex, fluxy,'k-o',linewidth=1, markersize = 3, label = title2)
    g.plot(time_interp, flux_interp,'b-o',linewidth=1, markersize = 3, label = title)
    g.legend(loc = 'best')
    g.set_ylabel('Flux',fontsize=15)
    g.set_xlabel('Julian Date',fontsize=15)
    g.set_title(title3,fontsize=15)
   
    g.legend(fontsize = 10, loc = 'upper center')
    for spine in ['top', 'right','bottom','left']:
        g.spines[spine].set_linewidth(2)
    g.tick_params(labelsize=10,length=3,width=2)
    
    return
    
############# Tables creation with all the data ############# 

def add_datapdfs(n, LCp, Kappa, theta, lnmu, lnsig, data_pdf, data_pdf2, pdf_light, pdf_light2):
    global data_pdfs0

    data_pdfs0 = data_pdfs0.append({'Number Points':n,'LC/LC section': LCp, 'Kappa': Kappa, 'Theta': theta, 'lnmu':lnmu, 'lnsig':lnsig, 'dPDF':data_pdf, 'dPDFlog':data_pdf2, 'PDF':pdf_light, 'PDFlog':pdf_light2}, ignore_index = True)
    return data_pdfs0
    
def add_datapsd(n, LCp, PSD, Freq, PSDb, Freqmean, errpsd):
    global data_psd0
    data_psd0 = data_psd0.append({'Number Points':n,'LC/LC section': LCp, 'PSD': PSD, 'Freq': Freq, 'PSDb':PSDb, 'Freqmean':Freqmean, 'err': errpsd}, ignore_index = True)
    return data_psd0
    
data_pdfs0 = pd.DataFrame(columns = ['Number Points','LC/LC section', 'Kappa', 'Theta', 'lnmu', 'lnsig', 'dPDF', 'dPDFlog', 'PDF', 'PDFlog'])
data_psd0 = pd.DataFrame(columns = ['Number Points','LC/LC section', 'PSD', 'Freq', 'PSDb', 'Freqmean', 'err'])



