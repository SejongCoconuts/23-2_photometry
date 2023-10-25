import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv('4FGL_J1653.8+3945_weekly_10_11_2023.csv')
dd = df[~df['Photon Flux [0.1-100 GeV](photons cm-2 s-1)'].str.contains('<',na=False, case=False)]

dd.to_csv("Mkn501.csv")
data = pd.read_csv('Mkn501.csv')

#pd.options.display.float_format = '{:.10f}'.format
flux = data['Photon Flux [0.1-100 GeV](photons cm-2 s-1)']
flux_err = data['Photon Flux Error(photons cm-2 s-1)']
flux_m = np.mean(flux)
tt = data['Julian Date']
nn = len(data)
pi = 3.141592
del_t = 7
x_bar = flux - flux_m
x_err_bar = flux_err - np.mean(flux_err)

# Create text file with time, flux, error
df_r = pd.DataFrame({'time':tt, 'flux':flux,'errors':flux_err})
df_r.to_csv('Mkn501.txt',sep = '\t', index = False)

#plt.errorbar(tt, flux, yerr=flux_err, fmt='o')
#plt.show()

print(nn)

#fitting the lightcurve
xx = tt
yy = flux
plt.title('Original LC')
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.grid()
plt.plot(xx,yy)
plt.show()
#plt.savefig('Mkn501_LC1.png')

#calculate the PSD
ffj,PSD = [],[]
for j in range(nn-1):
    fj = j/(nn*del_t)
    ffj.append(fj)
    cj = np.cos((2*pi*fj*tt))
    sj = np.sin((2*pi*fj*tt))
    DFT_j = (np.sum(x_bar*cj))**2 + (np.sum(x_bar*sj))**2
    PSD.append(DFT_j)

freq = np.array(np.log10(ffj))[1:]
PSD = np.array(np.log10(PSD))[1:]

#print(freq)

plt.scatter(freq,PSD)
plt.xlabel('log_freq')
plt.ylabel('log_PSD')
plt.title('Original PSD')
plt.show()

# Poission noise
#Pn = []
#for i in range(nn-1):
#    err_sig = flux_err*flux_err
#    Pnoise = (2*del_t*err_sig) / (flux_m**2)
#    Pn.append(Pnoise)

#PSD_tru = []
#for j in range(len(Pn)):
#    PSD_c = PSD[i] - Pn[i]
#    PSD_tru.append(PSD_c)

# errorbar
binsize = 5
mean_freq, tru_psd, err_psd = [],[],[]
for i in range(0, nn, binsize):
    k = i + binsize -1
    mean_freq.append(np.nanmean(ffj[i:k]))
    tru_psd.append(np.nanmean(PSD[i:k])+0.25068) 
    err_psd.append(np.nanstd(PSD[i:k]))

mean_freq = np.array(mean_freq)
tru_psd = np.array(tru_psd)
err_psd = np.array(err_psd)

#plt.errorbar(np.log10(mean_freq), tru_psd, yerr=err_psd, fmt='o')
#fitting the PL - linear
xdata = np.log10(mean_freq)
ydata = tru_psd

def func(x, a, b):
    return a*x + b
popt,pcov = curve_fit(func, xdata, ydata)
print(popt)
plt.subplot(1,3,1)

plt.errorbar(np.log10(mean_freq), tru_psd, yerr=err_psd, fmt='o')
plt.plot(xdata, func(xdata, *popt))
plt.xlabel('log_freq')
plt.ylabel('log_PSD')
plt.legend(['Linear PL'])
plt.grid()
#plt.show()


#fitting the PL - broken
xdata1 = np.log10(mean_freq)
ydata1 = tru_psd

def func(x, a, b):
    return a*x + b

#getting cross point
xx1, yy1, xx2, yy2= [],[],[],[]
for i in range(len(xdata1)):
    if xdata1[i] < -1.782:
        x1 = xdata[i]
        xx1.append(x1)
        y1 = ydata[i]
        yy1.append(y1)
    elif xdata1[i] > -1.782:
        x2 = xdata[i]
        xx2.append(x2)
        y2 = ydata[i]
        yy2.append(y2)       


popt1,pcov1 = curve_fit(func, xx1, yy1)
popt2,pcov2 = curve_fit(func, xx2, yy2)  

a1 = popt1[0]
b1 = popt1[1]
a2 = popt2[0]
b2 = popt2[1]

cx = (b2-b1)/(a1-a2)
cy = a1*cx+b1

#print(cx, cy)

xxs = np.linspace(-3.6,cx)
plt.subplot(1,3,2)

xxl = np.linspace(cx,-0.75)
plt.errorbar(np.log10(mean_freq), tru_psd, yerr=err_psd, fmt='o')
plt.plot(xxs, func(xxs, *popt1))
plt.plot(xxl, func(xxl, *popt2))
plt.xlabel('log_freq')
plt.ylabel('log_PSD')
plt.legend(['Broken PL1','Broken PL2'])
plt.grid()
#plt.show()
print(popt1)
print(popt2)



#fitting the PL - curved

def func(x, a, b, c):
    return a*x**2 + b*x + c
xdata3 = np.log10(mean_freq)
ydata3 = tru_psd
popt3,pcov3 = curve_fit(func, xdata3, ydata3)
print(popt3)
plt.subplot(1,3,3)

plt.errorbar(np.log10(mean_freq), tru_psd, yerr=err_psd, fmt='o')
plt.plot(xdata3, func(xdata3, *popt3))
plt.xlabel('log_freq')
plt.ylabel('log_PSD')
plt.grid()
plt.legend(['Curved PL'])
#plt.show()
plt.savefig('Original_data_PLs.png')
