import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import Rbf
import readline

def remove_below(threshold, data):
    thre = threshold*max(data[1])
    cons = (data[1] > thre)
    new = data[cons]
    return new

def PSD(delt, time, flux):

    import numpy as np
    
    fj = []
    for i in range(len(flux)):
        fj.append((i)/(len(flux)*delt))
    xbar = flux - np.mean(flux)
    time = time - time.values[0]
    # dft_list = []
    # for i in range(len(flux)):
    #     dft = (np.sum(xbar*np.cos(2*np.pi*fj[i]*time)))**2 + \
    #         (np.sum(xbar*np.sin(2*np.pi*fj[i]*time)))**2
    #     dft_list.append(dft)
    dft = (np.sum(xbar*np.cos(2*np.pi*fj*time)))**2 + (np.sum(xbar*np.sin(2*np.pi*fj*time)))**2
    return fj[1:], dft_list[1:]

def interpolation(x, y, Type):
    import numpy as np
    from scipy.interpolate import Rbf, CubicSpline
  
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
