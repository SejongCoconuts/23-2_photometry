from pipe import *

def analysis_LC(path_orig,model="SPL",niters=10,binning=10,interpolate=False,interpolate_method="Linear",removal_threshold=0,plot=False):
  
  LC = Analysis_LC(path_orig)
  LC.fit_PDF()
  if plot == True:         
    LC.plot_PDF()
  fit_param,fit_param_errors,chi_square = LC.fit_PSD(delt=14,binning=binning,model=model)
  print("Fit on original LC: {0} pm {1}, chisqu:{2}",fit_param,fit_param_errors,chi_square)

  # Check that params match the model

  time_sim,flux_sim_array,freq_sim, PSD_sim_array = LC.significance_calc(niters=1,PSD_params=[-1,1],PSD_model=model,best_fit=False)

  # Re-bin and fit the simulated LC (to confirm that we get back what we put in)
  sim_fit_param, sim_fit_param_errors, sim_chi_square = LC.fit_PSD(origin_data=False,delt=14,binning=binning,time=time_sim,flux=flux_sim_array,model=model)

  print("Input params for {0}: {1}".format(model,PSD_params))
  print("Fit on simulated LC: {0} pm {1}, chisqu:{2}",sim_fit_param,sim_fit_param_errors,sim_chi_square)

  # Remove data

  removed_time, removed_flux = LC.remove_below(threshold=removal_threshold)

  # Re-bin and fit the LC with data removed
  removed_fit_param, removed_fit_param_errors, removed_chi_square = LC.fit_PSD(origin_data=False,delt=14,binning=binning,time=removed_time,flux=removed_flux,model=model)

  print("Input params for {0}: {1}".format(model,PSD_params))
  print("Fit on removed simulated LC: {0} pm {1}, chisqu:{2}",removed_fit_param,removed_fit_param_errors,removed_chi_square)

  if interpolate == True:
    interp_time, interp_flux = LC.interpolate(Type=interpolate_method)
    interp_fit_param, interp_fit_param_errors, interp_chi_square = LC.fit_PSD(origin_data=False,delt=14,binning=binning,time=interp_time,flux=interp_flux,model=model)
    print("Input params for {0}: {1}".format(model,PSD_params))
    print("Fit on interpolated simulated LC: {0} pm {1}, chisqu:{2}",interp_fit_param,interp_fit_param_errors,interp_chi_square)

path = "/path/to/LC.csv"

analysis_LC(path,model="SPL",niters=10,interpolate=True,removal_threshold=0.1,interpolate_method="Linear")
