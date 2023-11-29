from pipe import *

def analysis_LC(path_orig,model="SPL",niters=10,binning=10,interpolate=False,interpolate_method="linear",removal_threshold=0,plot=False,PSD_params=[-1,1]):

  LC = Analysis_LC(path_orig)
  LC.fit_PDF()
  if plot == True:
    axLC_orig = plt.subplot(421)
    axPD_orig = plt.subplot(422)
    axLC_sim = plt.subplot(423)
    axPD_sim = plt.subplot(424)
    axLC_removed = plt.subplot(425)
    axPD_removed = plt.subplot(426)
    axLC_interp = plt.subplot(427)
    axPD_interp = plt.subplot(428)
  # if plot == True:
    # LC.plot_PDF()
  fit_param,fit_param_errors,chi_square = LC.fit_PSD(delt=14,binning=binning,model=model)
  orig_freq, orig_PSD = LC.calc_PSD(delt=14)

  if plot == True:
    axLC_orig.errorbar(LC.time,LC.flux,ls=' ',marker='.')
    axPD_orig.errorbar(orig_freq,orig_PSD,ls=' ',marker='.')
    axPD_orig.set_xscale('log')
    axPD_orig.set_yscale('log')

  print("Fit on original LC: {0} pm {1}, chisqu:{2}".format(fit_param,fit_param_errors,chi_square))

  # Check that params match the model

  time_sim,flux_sim_array,freq_sim, PSD_sim_array = LC.significance_calc(niters=1,PSD_params=PSD_params,PSD_model=model,best_fit=False)

  # Re-bin and fit the simulated LC (to confirm that we get back what we put in)
  sim_fit_param, sim_fit_param_errors, sim_chi_square = LC.fit_PSD(origin_data=False,delt=14,binning=binning,time=time_sim,flux=flux_sim_array,model=model)

  if plot == True:
    axLC_sim.errorbar(time_sim,flux_sim_array,ls=' ',marker='.')
    axPD_sim.errorbar(freq_sim,PSD_sim_array,ls=' ',marker='.')
    axPD_sim.set_xscale('log')
    axPD_sim.set_yscale('log')

  print("Input params for {0}: {1}".format(model,PSD_params))
  print("Fit on simulated LC: {0} pm {1}, chisqu:{2}".format(sim_fit_param,sim_fit_param_errors,sim_chi_square))

  # Remove data

  removed_time, removed_flux = LC.remove_below(threshold=removal_threshold)

  # print(removed_time[0])

  removed_time = np.array(removed_time)
  removed_flux = np.array(removed_flux)

  # Re-bin and fit the LC with data removed
  removed_fit_param, removed_fit_param_errors, removed_chi_square = LC.fit_PSD(origin_data=False,delt=14,binning=binning,time=removed_time[0],flux=removed_flux[0],model=model)
  removed_freq, removed_PSD = LC.calc_PSD(delt=14,time=removed_time[0],flux=removed_flux[0],origin_data=False)


  print("Input params for removed {0}: {1}".format(model,PSD_params))
  print("Fit on removed simulated LC: {0} pm {1}, chisqu:{2}".format(removed_fit_param,removed_fit_param_errors,removed_chi_square))

  if plot == True:
    axLC_removed.errorbar(removed_time[0],removed_flux[0],ls=' ',marker='.')
    axPD_removed.errorbar(removed_freq,removed_PSD,ls=' ',marker='.')
    axPD_removed.set_xscale('log')
    axPD_removed.set_yscale('log')

  if interpolate == True:
    interp_time, interp_flux, interp_freq, interp_PSD = LC.interpolate(interp_type=interpolate_method)
    interp_fit_param, interp_fit_param_errors, interp_chi_square = LC.fit_PSD(origin_data=False,delt=14,binning=binning,time=interp_time,flux=interp_flux,model=model,paramsSPL=PSD_params)
    if plot == True:
      axLC_interp.scatter(interp_time,interp_flux,marker='.')
      axPD_interp.scatter(interp_freq,interp_PSD,marker='.')
      axPD_interp.set_xscale('log')
      axPD_interp.set_yscale('log')
    print("Input params for interpolation {0}: {1}".format(model,PSD_params))
    print("Fit on interpolated simulated LC: {0} pm {1}, chisqu:{2}".format(interp_fit_param,interp_fit_param_errors,interp_chi_square))
    if plot == True:
      fig = plt.gcf()
      fig.set_size_inches(10,30)
      plt.show()

path = "./4FGL_J1048.4+7143_weekly_29_11_2023.csv"

analysis_LC(path,model="SPL",niters=10,interpolate=True,removal_threshold=0.2,interpolate_method="linear",plot=True,PSD_params=[0,-15])
