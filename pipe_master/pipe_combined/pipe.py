import readline
import numpy as np
import pandas as pd
import os
from scipy.odr import ODR, Model, RealData
import scipy.stats as st
import numpy.fft as ft
import numpy.random as rnd
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import Rbf

class Analysis_LC:
    def __init__(self,path_csv):

        readline.parse_and_bind("tab: complete")
        def path_completer(text, state):
            return [x for x in os.listdir('.') if x.startswith(text)][state]
        readline.set_completer(path_completer)

        df = pd.read_csv(path_csv)[['Julian Date', 'Photon Flux [0.1-100 GeV](photons cm-2 s-1)', 'Photon Flux Error(photons cm-2 s-1)']]

        # Remove spaces and operation symbols in data

        df = df[df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].str.strip() != '']
        mask = df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].str.contains('<|>', regex=True)
        df = df[~mask]

        self.time=np.array(df['Julian Date'])
        self.flux=np.array(df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)']).astype(float)
        self.error=np.array(df['Photon Flux Error(photons cm-2 s-1)'])

        self.all_flux=np.array([self.flux])
        self.all_Time=np.array(self.time)

        time1=self.time-self.time[0]

        self.tbin=np.min(time1[1:])

        df.to_csv('filtered.csv',sep = '\t',index = False)


    def fit_PDF(self,flux=None,origin_data=True):
        if origin_data:
            flux=self.flux
        
        # Fit a log-normal distribution to the data and estimate parameters
        lnsig, loc, scale = st.lognorm.fit(flux, floc=0)
        
        # Calculate the mean (lnmu)
        lnmu = loc - (lnsig**2) / 2

        if origin_data:
            self.lnsig=lnsig
            self.loc=loc
            self.scale=scale

        return lnsig, loc, scale

    def plot_PDF(self,flux=None,lnsig=None, loc=None, scale=None,origin_data=True):
        if origin_data:
            flux=self.flux
            lnsig=self.lnsig
            loc=self.loc
            scale=self.scale

        # Generate PDF
        data_pdf = np.linspace(0, max(flux), 1000)
        pdf_light = st.lognorm.pdf(data_pdf, lnsig, loc, scale)
        
        fig, g = plt.subplots(figsize=(10, 5))
        fig.set_facecolor('white')
        g.plot(data_pdf, pdf_light, 'r-', lw=2, label='log-normal distribution PDF')
        g.hist(flux, alpha = 0.5, bins=30, density=True, color = 'b', label=r'Photon Flux', edgecolor = 'black')
        g.set_ylabel('Probability Density Fuction',fontsize=15)
        g.legend(loc ='best')
        for spine in ['top', 'right','bottom','left']:
            g.spines[spine].set_linewidth(2)
        g.tick_params(labelsize=10,length=3,width=2)
        plt.show()

        return data_pdf, pdf_light


    def calc_PSD(self,delt,time=None,flux=None,origin_data=True):
        if origin_data:
            time=self.time
            flux=self.flux
        
        self.delt=delt
        n=len(flux)
        flux_mean=np.mean(flux)

        DFJ=np.zeros(n)
        f=np.zeros(n)
        time1=time-time[0]

        self.tbin=np.min(time1[1:])

        for i in range(n):
            f[i]=float(i)/n/delt
            DFJ[i]=(np.sum((flux-flux_mean)*np.cos((2*np.pi*f[i]*time1))))**2+(np.sum((flux-flux_mean)*np.sin((2*np.pi*f[i]*time1)))**2)

        if origin_data:
            self.f=f[1:]
            self.DFJ=DFJ[1:]

        return f[1:], DFJ[1:]


    def fit_PSD(self,delt,binning,time=None,flux=None,model='SPL',paramsSPL=[-1,1],paramsBPL=[-1,-1,1,1],paramsCPL=[-1,1,1],origin_data=True):
        if origin_data:
            self.calc_PSD(delt)
            f=self.f
            PSD=self.DFJ
        
        else:
            f,PSD=self.calc_PSD(delt,time=time,flux=flux,origin_data=False)
        
        hist,bin_edges=np.histogram((f), bins=binning, range=None, density=None, weights=None)
        bin_edges=(bin_edges[1:]+(bin_edges[0]-bin_edges[1])/2)

        start_num=1
        end_num=1

        PSDmean=np.zeros(binning)
        PSDerror=np.zeros(binning)
        fmean=np.zeros(binning)
        ferror=np.zeros(binning)

        for i in range(binning):
            end_num += hist[i]

            PSDmean[i]=(np.mean((PSD[start_num:end_num])))
            PSDerror[i]=np.std(((PSD[start_num:end_num])))

            fmean[i]=(np.mean((f[start_num:end_num])))
            ferror[i]=np.std(((f[start_num:end_num])))

            start_num += hist[i]

        self.PSDmean=PSDmean
        self.fmean=fmean

        if model=='SPL':
            self.model=SPL
            odr_model = Model(SPL)
            mydata = RealData(fmean, PSDmean, sx=PSDerror, sy=ferror)
            odr = ODR(mydata, odr_model, beta0=paramsSPL)
            output = odr.run()

            fit_params=output.beta
            fit_param_errors=output.cov_beta

            chi_sq=np.sum((PSDmean-SPL(fit_params,fmean))**2)

        elif model=='CPL':
            self.model=CPL
            odr_model = Model(CPL)
            mydata = RealData(fmean, PSDmean, sx=PSDerror, sy=ferror)
            odr = ODR(mydata, odr_model, beta0=paramsCPL)
            output = odr.run()

            fit_params=output.beta
            fit_param_errors=output.cov_beta

            chi_sq=np.sum((PSDmean-CPL(fit_params,fmean))**2)

        elif model=='BPL':
            self.model=BPL
            odr_model = Model(BPL)
            mydata = RealData(fmean, PSDmean, sx=PSDerror, sy=ferror)
            odr = ODR(mydata, odr_model, beta0=paramsBPL)
            output = odr.run()

            fit_params=output.beta
            fit_param_errors=output.cov_beta

            chi_sq=np.sum((PSDmean-BPL(fit_params,fmean))**2)

        if origin_data:
            self.fit_params=fit_params
            self.fit_param_errors=fit_param_errors
            self.chi_sq=chi_sq
            
        return fit_params,fit_param_errors,chi_sq
    
    def plot_PSD(self):
        plt.plot(self.f,self.DFJ, zorder=1)
        y=self.model(self.fit_params,self.f)
        plt.scatter(self.fmean,self.PSDmean,c='C1' ,zorder=2)
        plt.plot(self.f,y,c='r',zorder=3)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()


    def sim_LC(self,time=None,flux=None, PSD_params=None,PSD_model='SPL',PDFparams=None,PDFmodel='lognorm',best_fit=True):
        time=self.time
        flux=self.flux
        tbin=self.tbin
        if best_fit:
            PSD_model=self.model
            PSD_params=self.fit_params
            PDFparams=[self.lnsig, self.loc, self.scale]
            PDFparams=[self.lnsig, self.loc, self.scale]
        else:
            if PSD_model=='SPL':
                PSD_model=SPL
            elif PSD_model=='CPL':
                PSD_model=SPL
            elif PSD_model=='BPL':
                PSD_model=BPL
            
            
            time1=time-time[0]

            tbin=np.min(time1[1:])

        
        surrogate, PSDlast, shortLC, periodogram, ffti=EmmanLC(time,flux,tbin=tbin,PSDmodel=PSD_model,PSDparams=PSD_params,PDFmodel=PDFmodel,PDFparams=PDFparams
                                                               ,RedNoiseL=100,aliasTbin=1,RandomSeed=None,maxIterations=1000,histSample=True)
        
        simulation_time=surrogate[0]
        simulation_flux=surrogate[1]

        simulation_freq,simulation_PSD=self.calc_PSD(delt=self.delt,time=simulation_time,flux=simulation_flux,origin_data=False)


        return simulation_time,simulation_flux,simulation_freq,simulation_PSD

    def significance_calc(self,niters,PSD_params=None,PSD_model='SPL',PDFparams=None,PDFmodel='lognorm',best_fit=True):
        lengthT=len(self.flux)
        lengthf=len(self.f)
        self.all_flux=np.zeros((lengthT,niters))
        self.all_PSD=np.zeros((lengthf,niters))
        if best_fit:
            for i in range(niters):
                simulation_time,simulation_fliux,simulation_freq,simulation_PSD= self.sim_LC()
                self.all_flux[:,i]=simulation_fliux
                self.all_PSD[:,i]=simulation_PSD
            return simulation_time,self.all_flux,simulation_freq,self.all_PSD
        else:
            for i in range(niters):
                simulation_time,simulation_fliux,simulation_freq,simulation_PSD= self.sim_LC(PSD_params=PSD_params,PSD_model=PSD_model,PDFparams=PDFparams,PDFmodel=PDFmodel,best_fit=best_fit)
                self.all_flux[:,i]=simulation_fliux
                self.all_PSD[:,i]=simulation_PSD
            return simulation_time,self.all_flux,simulation_freq,self.all_PSD


    def calc_fit_sim(self,data='all_data'):
        if data=='all_data':
            lengthT,niters=np.shape(self.all_flux)
            lengthf,niters=np.shape(self.all_PSD)
            

            PSD_sigma1=np.zeros(lengthf)
            PSD_sigma2=np.zeros(lengthf)
            PSD_sigma3=np.zeros(lengthf)

            sigma1_pos=int(0.68*niters)
            sigma2_pos=int(0.95*niters)
            sigma3_pos=int(0.99*niters)

            PSD_sigma1=np.sort(self.all_PSD, axis=1)[:,sigma1_pos]
            PSD_sigma2=np.sort(self.all_PSD, axis=1)[:,sigma2_pos]
            PSD_sigma3=np.sort(self.all_PSD, axis=1)[:,sigma3_pos]


            re_flux=self.all_flux.reshape(lengthT*niters)
            sorted_list = sorted(re_flux)

            sigma1_pos=int(0.68*len(sorted_list))
            sigma2_pos=int(0.95*len(sorted_list))
            sigma3_pos=int(0.99*len(sorted_list))
            

            sigma1=sorted_list[sigma1_pos]
            sigma2=sorted_list[sigma2_pos]
            sigma3=sorted_list[sigma3_pos]

            return sigma1,sigma2,sigma3, PSD_sigma1,PSD_sigma2,PSD_sigma3
        elif data=='rm_data':
            new_all_flux=self.re_all_flux
            new_all_time=self.re_all_time
            all_re_freq=[]
            all_re_PSD=[]

            PSD_sigma1=np.zeros(len(self.f))
            PSD_sigma2=np.zeros(len(self.f))
            PSD_sigma3=np.zeros(len(self.f))

            for i in range(self.number):
                re_freq,re_PSD=self.calc_PSD(delt=self.delt,time=new_all_time[i],flux=new_all_flux[i],origin_data=False)
                all_re_freq.append(re_freq)
                all_re_PSD.append(re_PSD)
            
            for i in range(len(self.f)):

                indices = np.array([all_re_PSD[j][k] for j, row in enumerate(all_re_freq) for k, value in enumerate(row) if value == self.f[i]])

                
                sigma1_pos=int(0.68*len(indices))
                sigma2_pos=int(0.95*len(indices))
                sigma3_pos=int(0.99*len(indices))
                if sigma1_pos !=0:

                    PSD_sigma1[i]=np.sort(indices)[sigma1_pos]
                    PSD_sigma2[i]=np.sort(indices)[sigma2_pos]
                    PSD_sigma3[i]=np.sort(indices)[sigma3_pos]


            new_all_flux=self.re_all_flux
            re_flux = [item for sublist in new_all_flux for item in sublist]

            sorted_list = sorted(re_flux)

            sigma1_pos=int(0.68*len(sorted_list))
            sigma2_pos=int(0.95*len(sorted_list))
            sigma3_pos=int(0.99*len(sorted_list))
            

            sigma1=sorted_list[sigma1_pos]
            sigma2=sorted_list[sigma2_pos]
            sigma3=sorted_list[sigma3_pos]

            return sigma1,sigma2,sigma3, PSD_sigma1,PSD_sigma2,PSD_sigma3
        
        elif data=='interpolate_data':

            lengthT,niters=np.shape(self.interpol_flux)
            lengthf,niters=np.shape(self.interpol_PSD)
            

            PSD_sigma1=np.zeros(lengthf)
            PSD_sigma2=np.zeros(lengthf)
            PSD_sigma3=np.zeros(lengthf)

            sigma1_pos=int(0.68*niters)
            sigma2_pos=int(0.95*niters)
            sigma3_pos=int(0.99*niters)

            PSD_sigma1=np.sort(self.interpol_PSD, axis=1)[:,sigma1_pos]
            PSD_sigma2=np.sort(self.interpol_PSD, axis=1)[:,sigma2_pos]
            PSD_sigma3=np.sort(self.interpol_PSD, axis=1)[:,sigma3_pos]


            re_flux=self.interpol_flux.reshape(lengthT*niters)
            sorted_list = sorted(re_flux)

            sigma1_pos=int(0.68*len(sorted_list))
            sigma2_pos=int(0.95*len(sorted_list))
            sigma3_pos=int(0.99*len(sorted_list))
            

            sigma1=sorted_list[sigma1_pos]
            sigma2=sorted_list[sigma2_pos]
            sigma3=sorted_list[sigma3_pos]

            return sigma1,sigma2,sigma3, PSD_sigma1,PSD_sigma2,PSD_sigma3
            return

    
    def remove_below(self,threshold,probability=1):

        flux=self.all_flux
        time=self.all_Time
        thre = threshold*np.max(flux)
        lengthT,niters=np.shape(flux)
        new_all_time=[]
        new_all_flux=[]

        self.number=niters

        for j in range(niters):
            new_flux=[]
            new_time=[]
            for i in range(lengthT):
                if flux[i,j] > thre and rnd.rand()<probability:
                    new_time.append(time[i])
                    new_flux.append(flux[i,j])
                elif rnd.rand()>probability:
                    new_time.append(time[i])
                    new_flux.append(flux[i,j])
            new_all_time.append(new_time)
            new_all_flux.append(new_flux)

        self.re_all_flux=new_all_flux
        self.re_all_time=new_all_time
        return new_all_time,new_all_flux

    def interpolate(self,Type='linear'):
        xnew = np.arange(min(self.time), max(self.time), self.tbin)

        niters=self.number

        interpol_flux=np.zeros((niters,len(xnew)))
        interpol_PSD=np.zeros((niters,int(len(xnew)-1)))

        time=self.re_all_time
        flux=self.re_all_flux

        for i in range(niters):
            interpol_time, interpol_flux[i]=interpolate_method(xnew, time[i], flux[i],Type)
            interpol_f, interpol_PSD[i]=self.calc_PSD(self.delt,time=interpol_time,flux=interpol_flux[i],origin_data=False)

        
        self.interpol_flux=interpol_flux
        self.interpol_time=interpol_time
        self.interpol_f=interpol_f
        self.interpol_PSD=interpol_PSD
        return interpol_time,interpol_flux,interpol_f,interpol_PSD
            



def interpolate_method(xnew, time, flux,Type):
    if (Type == 'linear'):
        ynew = np.interp(xnew, time, flux)
        return xnew, ynew
    if (Type == 'spline'):
        cs = CubicSpline(time, flux)
        ys = cs(xnew)
        return xnew, ys
    if (Type == 'gauss'):
        rbfi = Rbf(time, flux, function='gaussian')
        ynew = rbfi(xnew)
        return xnew, ynew   


def SPL(theta,v):
  alpha1, b1=theta
  return 10**(np.log10(v)*alpha1 + b1)

def CPL(theta,v):
  alpha1, alpha2, b1=theta
  return 10**(alpha1*np.log10(v)**2 + alpha2*np.log10(v) +(b1))

def BPL(theta,v):
    alpha1,alpha2,break1,b1=theta

    result=np.zeros(len(v))
    for i in range(len(v)):
        if v[i]<break1:
            result[i]= 10**(alpha1*np.log10(v[i]/break1)+b1)
        else:
            result[i]= 10**(alpha2*np.log10(v[i]/break1)+b1)
    return result
          
    

def PDF_Sample(flux):
    '''
    Generate random sample the flux histogram of a lightcurve by sampling the 
    piecewise distribution consisting of the box functions forming the 
    histogram of the lightcurve's flux.
    
    inputs:
        lc (Lightcurve)
            - Lightcurve whose histogram will be sample
    outputs:
        sample (array, float)
            - Array of data sampled from the lightcurve's flux histogram
    '''
    
    bins = OptBins(flux)
    
    pdf = np.histogram(flux,bins=bins)
    chances = pdf[0]/float(sum(pdf[0]))
    nNumbers = len(flux)
    
    sample = np.random.choice(len(chances), nNumbers, p=chances)
    
    sample = np.random.uniform(pdf[1][:-1][sample],pdf[1][1:][sample])

    return sample


def OptBins(data,maxM=100):
    '''
     Python version of the 'optBINS' algorithm by Knuth et al. (2006) - finds 
     the optimal number of bins for a one-dimensional data set using the 
     posterior probability for the number of bins. WARNING sometimes doesn't
     seem to produce a high enough number by some way...
    
     inputs:
         data (array)           - The data set to be binned
         maxM (int, optional)   - The maximum number of bins to consider
         
     outputs:
        maximum (int)           - The optimum number of bins
    
     Ref: K.H. Knuth. 2012. Optimal data-based binning for histograms
     and histogram-based probability density models, Entropy.
    '''
    
    N = len(data)
    
    # loop through the different numbers of bins
    # and compute the posterior probability for each.
    
    logp = np.zeros(maxM)
    
    for M in range(1,maxM+1):
        n = np.histogram(data,bins=M)[0] # Bin the data (equal width bins)
        
        # calculate posterior probability
        part1 = N * np.log(M) + sp.gammaln(M/2.0)
        part2 = - M * sp.gammaln(0.5)  - sp.gammaln(N + M/2.0)
        part3 = np.sum(sp.gammaln(n+0.5))
        logp[M-1] = part1 + part2 + part3 # add to array of posteriors

    maximum = np.argmax(logp) + 1 # find bin number of maximum probability
    return maximum + 10 

    
def TimmerKoenig(RedNoiseL, aliasTbin, randomSeed, tbin, LClength,\
                    PSDmodel, PSDparams,std=1.0, mean=0.0):    
    '''
    Generates an artificial lightcurve with the a given power spectral 
    density in frequency space, using the method from Timmer & Koenig, 1995,
    Astronomy & Astrophysics, 300, 707.

    inputs:
        RedNoiseL (int)        - multiple by which simulated LC is lengthened 
                                 compared to data LC to avoid red noise leakage
        aliasTbin (int)        - divisor to avoid aliasing
        randomSeed (int)       - Random number seed
        tbin (int)           - Sample rate of output lightcurve        
        LClength  (int)        - Length of simulated LC
        std (float)            - standard deviation of lightcurve to generate
        mean (float)           - mean amplitude of lightcurve to generate
        PSDmodel (function)    - Function for model used to fit PSD
        PSDparams (various) - Arguments/parameters of best-fitting PSD model
   
    outputs:
        lightcurve (array)     - array of amplitude values (cnts/flux) with the 
                                 same timing properties as entered, length 1024
                                 seconds, sampled once per second.  
        fft (array)            - Fourier transform of the output lightcurve
        shortPeriodogram (array, 2 columns) - periodogram of the output 
                                              lightcurve [freq, power]
        '''                    
    # --- create freq array up to the Nyquist freq & equivalent PSD ------------
    frequency = np.arange(1.0, (RedNoiseL*LClength)/2 +1)/ \
                                            (RedNoiseL*LClength*tbin*aliasTbin)
    powerlaw = PSDmodel(PSDparams,frequency)

    # -------- Add complex Gaussian noise to PL --------------------------------
    rnd.seed(randomSeed)
    real = (np.sqrt(powerlaw*0.5))*rnd.normal(0,1,int((RedNoiseL*LClength)/2))
    imag = (np.sqrt(powerlaw*0.5))*rnd.normal(0,1,int((RedNoiseL*LClength)/2))
    positive = np.vectorize(complex)(real,imag) # array of +ve, complex nos
    noisypowerlaw = np.append(positive,positive.conjugate()[::-1])
    znoisypowerlaw = np.insert(noisypowerlaw,0,complex(0.0,0.0)) # add 0

    # --------- Fourier transform the noisy power law --------------------------
    inversefourier = np.fft.ifft(znoisypowerlaw)  # should be ONLY  real numbers
    longlightcurve = inversefourier.real       # take real part of the transform
 
    # extract random cut and normalise output lightcurve, 
    # produce fft & periodogram
    if RedNoiseL == 1:
        lightcurve = longlightcurve
    else:
        extract = rnd.randint(LClength-1,RedNoiseL*LClength - LClength)
        lightcurve = np.take(longlightcurve,range(extract,extract + LClength))

    if mean: 
        lightcurve = lightcurve-np.mean(lightcurve)
    if std:
        lightcurve = (lightcurve/np.std(lightcurve))*std
    if mean:
        lightcurve += mean

    fft = ft.fft(lightcurve)

    periodogram = np.absolute(fft)**2.0 * ((2.0*tbin*aliasTbin*RedNoiseL)/\
                   (LClength*(np.mean(lightcurve)**2)))   
    shortPeriodogram = np.take(periodogram,range(1,int(LClength/2 +1)))
    #shortFreq = np.take(frequency,range(1,LClength/2 +1))
    shortFreq = np.arange(1.0, (LClength)/2 +1)/ (LClength*tbin)
    shortPeriodogram = [shortFreq,shortPeriodogram]

    return lightcurve, fft, shortPeriodogram

def EmmanLC(time,flux,tbin,PSDmodel, PSDparams, PDFmodel='lognorm', PDFparams=None,RedNoiseL=100,aliasTbin=1,RandomSeed=None,
            maxIterations=1000,histSample=True):
    '''
    Produces a simulated lightcurve with the same power spectral density, mean,
    standard deviation and probability density function as those supplied.
    Uses the method from Emmanoulopoulos et al., 2013, Monthly Notices of the
    Royal Astronomical Society, 433, 907. Starts from a lightcurve using the
    Timmer & Koenig (1995, Astronomy & Astrophysics, 300, 707) method, then
    adjusts a random set of values ordered according to this lightcurve, 
    such that it has the correct PDF and PSD. Using a scipy.stats distribution
    recommended for speed.
    
    inputs:
        time (array)    
            - Times from data lightcurve
        flux (array)    
            - Fluxes from data lightcurve       
        RedNoiseL (int) 
            - multiple by which simulated LC is lengthened compared to the data 
              LC to avoid red noise leakage
        aliasTbin (int) 
            - divisor to avoid aliasing
        RandomSeed (int)
            - random number generation seed, for repeatability
        tbin (int)      
            - lightcurve bin size
        PSDmodel (fn)   
            - Function for model used to fit PSD
        PSDparams (tuple,var) 
                        
            - parameters of best-fitting PSD model
        PDFmodel (fn,optional) 
            - Function for model used to fit PDF if not scipy
        PDFparams (tuple,var) 
            - Distributions/params of best-fit PDF model(s). If a scipy random 
              variate is used, this must be in the form: 
                  ([distributions],[[shape,loc,scale]],[weights])
        maxIterations (int,optional) 
            - The maximum number of iterations before the routine gives up 
              (default = 1000)
        verbose (bool, optional) 
            - If true, will give you some idea what it's doing, by telling you 
              (default = False)
        LClength  (int) 
            - Length of simulated LC        
        histSample (Lightcurve, optional)
            - If 
 
    outputs: 
        surrogate (array, 2 column)     
                        - simulated lightcurve [time,flux]
        PSDlast (array, 2 column)       
                        - simulated lighturve PSD [freq,power]
        shortLC (array, 2 column)       
                        - T&K lightcurve [time,flux]
        periodogram (array, 2 column)   
                        - T&K lighturve PSD [freq,power]
        ffti (array)                    
                        - Fourier transform of surrogate LC
        LClength (int)              
                        - length of resultant LC if not same as input
    '''

    length = len(time)

    maxFlux = max(flux) 

    ampAdj = None
        
    tries = 0      
    success = False
    
    if histSample:
        mean = np.mean(flux)
    else:
        mean = 1.0
    
    while success == False and tries < 5:
        try:
            shortLC, fft, periodogram =TimmerKoenig(RedNoiseL,aliasTbin,RandomSeed,tbin,len(time),PSDmodel,PSDparams,mean=1.0)
            success = True
        # This has been fixed and should never happen now in theory...
        except IndexError:
            tries += 1
            print ("Simulation failed for some reason (IndexError) - restarting...")

    shortLC = [np.arange(len(shortLC))*tbin, shortLC]
    
    # Produce random distrubtion from PDF, up to max flux of data LC
    # use inverse transform sampling if a scipy dist
    '''
    if histSample:
        #dist = PDF_Sample(histSample)
    else:
        mix = False
        scipy = False
        try:
            if PDFmodel.__name__ == "Mixture_Dist":
                mix = True
        except AttributeError:
            mix = False
        try:
            if PDFmodel.__module__ == 'scipy.stats.distributions' or \
                PDFmodel.__module__ == 'scipy.stats._continuous_distns' or \
                    force_scipy == True:
                scipy = True
        except AttributeError:
            scipy = False
    
        if mix:     
            if verbose: 
                print "Inverse tranform sampling..."
            dist = PDFmodel.Sample(PDFparams,length)
        elif scipy:
            if verbose: 
                print "Inverse tranform sampling..."
            dist = PDFmodel.rvs(*PDFparams,size=length)
            
        else: # else use rejection
            if verbose: 
                print "Rejection sampling... (slow!)"
            if maxFlux == None:
                maxFlux = 1
            dist = RandAnyDist(PDFmodel,PDFparams,0,max(maxFlux)*1.2,length)
            dist = np.array(dist)
    '''
    #data_pdf = np.linspace(0, maxFlux, 1000)
    dist = PDF_Sample(flux)

    sortdist = dist[np.argsort(dist)] # sort!
    
    i = 0
    oldSurrogate = np.array([-1])
    surrogate = np.array([1])
    
    while i < maxIterations and np.array_equal(surrogate,oldSurrogate) == False:
    
        oldSurrogate = surrogate
    
        if i == 0:
            surrogate = [time, dist] # start with random distribution from PDF
        else:
            surrogate = [time,ampAdj]#
            
        ffti = ft.fft(surrogate[1])
        
        PSDlast = ((2.0*tbin)/(length*(mean**2))) *np.absolute(ffti)**2
        PSDlast = [periodogram[0],np.take(PSDlast,range(1,int(length/2 +1)))]
        
        fftAdj = np.absolute(fft)*(np.cos(np.angle(ffti)) \
                                    + 1j*np.sin(np.angle(ffti)))  #adjust fft
        LCadj = ft.ifft(fftAdj)
        LCadj = [time/tbin,LCadj]

        PSDLCAdj = ((2.0*tbin)/(length*np.mean(LCadj)**2.0)) \
                                                 * np.absolute(ft.fft(LCadj))**2
        PSDLCAdj = [periodogram[0],np.take(PSDLCAdj, range(1,int(length/2 +1)))]
        sortIndices = np.argsort(LCadj[1])
        sortPos = np.argsort(sortIndices)
        ampAdj = sortdist[sortPos]
        
        i += 1

    
    return surrogate, PSDlast, shortLC, periodogram, ffti
