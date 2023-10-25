import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit


df2 = pd.read_csv("Mkn501_il.txt",sep="\t",names =['time','flux'], skiprows=[0])

#print(data)
tt = df2['time']
flux = df2['flux']
flux_m = np.mean(flux)

nn = len(df2)
pi = 3.141592
del_t = 7
x_bar = flux - flux_m

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

plt.title('Linear interpolation PSD')
plt.scatter(freq,PSD)
plt.xlabel('log_freq')
plt.ylabel('log_PSD')
#plt.show()
plt.savefig('Linear interpolation PSD.png')

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

plt.errorbar(np.log10(mean_freq), tru_psd, yerr=err_psd, fmt='o')

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
xxl = np.linspace(cx,-0.75)
plt.subplot(1,3,2)
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
plt.legend(['Curved PL'])
plt.grid()
#plt.show()

plt.savefig('Linear interpolation PLs.png')