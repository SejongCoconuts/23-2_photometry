
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

def calc_PSD():
  return PSD,freq

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





