from pipe import *

def analysis_LC(path_orig, 
                model="SimplePL", 
                niters=10, 
                binning=10, 
                interpolate=False, 
                interpolate_method="Linear",
               removal_threshold=0):

  LC = Analysis_LC(path_orig)
  LC.fit_PDF()
  LC.plot_PDF()
								 

                  


