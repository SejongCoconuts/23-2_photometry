
def read_original_csv(file):
  pd.read_csv()
  return time,flux,error

def fit_PDF():
  return PDF

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





