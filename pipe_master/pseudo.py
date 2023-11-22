
def read_original_csv(file):
  pd.read_csv()
  return time,flux,error

def fit_PDF(flux):
 
  # Fit a log-normal distribution to the data and estimate parameters
  lnsig, loc, scale = lognorm.fit(flux, floc=0)
  
  # Calculate the mean (lnmu)
  lnmu = loc - (lnsig**2) / 2

  # Generate PDF
  data_pdf = np.linspace(0, data.max(), 1000)
  pdf_light = lognorm.pdf(data_pdf, lnsig, loc, scale)
  
  fig, g = plt.subplots(figsize=(10, 5))
  fig.set_facecolor('white')
  g.plot(data_pdf, pdf_light, 'r-', lw=2, label='log-normal distribution PDF')
  g.hist(flux, alpha = 0.5, bins=30, density=True, color = 'b', label=r'Photon Flux', edgecolor = 'black')
  g.set_ylabel('Probability Density Fuction',fontsize=15)
  g.legend(loc ='best')
  for spine in ['top', 'right','bottom','left']:
      g.spines[spine].set_linewidth(2)
  g.tick_params(labelsize=10,length=3,width=2)
 
  return lnmu, lnsig, data_pdf, pdf_light



def calc_PSD(delt, time, flux):
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


def bin_PSD(time,flux,error,tbin,binning,model,init_params):
  # SPL
  # BPL
  # CPL
  return PSD,freq,fit_params,fit_param_errors,chi_squ,red_chi_squ

def sim_LC(PSD_model,PSD_params,PDF_model,PDF_params):
  return time,flux,PSD,freq

def significance_calc(niters,PSD_model,PSD_params,PDF_model,PDF_params):
  for i in range(niters):
    time,flux,PSD,freq = sim_LC(PSD_model,PSD_params,PDF_model,PDF_params)
    return time_i,flux_i,PSD_i,freq_i
  return flux99,PSD99

def remove_data(rules):
  return time,flux

def calc_fit_sim(time,flux):
  PSD, freq = calc_PSD(time,flux)
  params, param_err = bin_PSD(PSD,freq)
  return params,param_errs

def interpolate(type):
  return interp_time, interp_flux

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



