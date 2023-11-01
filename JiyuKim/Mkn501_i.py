import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy import interpolate


#convert .dat -> .txt
dd = pd.read_csv("Mkn501_rs.dat", sep =" ", names =['time','flux'], skiprows=[0])

time = dd['time']
flux = dd['flux']

dd0 = pd.DataFrame({'time':[time], 'flux':[flux]})
df = dd0.apply(pd.Series.explode)
df.to_csv("Mkn501_rs.txt",sep="\t", index = False) 

#read input and simulation
df1 = pd.read_csv("Mkn501.txt", sep="\t", names=['time','flux','error'],skiprows=[0])

df2 = pd.read_csv("Mkn501_rs.txt",sep="\t",names =['time','flux'], skiprows=[0])
df3 = pd.read_csv("Mkn501_r0.txt",sep="\t",names = ['time','flux'], skiprows=[0])

#make file(input+simulation)
df0 = pd.concat([df2,df3])
df_s = df0.sort_values(by='time')

time1 = df1['time'] # time data of original LC
flux1 = df1['flux'] # flux data of original LC
time2 = df_s['time'] # time data of simulation LC
flux2 = round(df_s['flux'],10) # flux data of simulation LC

df4 = pd.DataFrame({'time':[time1],'flux_d':[flux1],'flux_f':[flux2]})
df_f = df4.apply(pd.Series.explode)
df_f.to_csv("Mkn501_f1.txt",sep="\t", index = False)
df5 = pd.read_csv("Mkn501_f1.txt",sep="\t",names = ['time','flux_d','flux_f'], skiprows=[0])
ee = df5[df5['flux_f']==0.0].index 
df6 = df5.drop(ee)

#ouput data

time = df6['time']
flux_d = df6['flux_d']
flux_f = df6['flux_f']

#original LC
x1 = np.array(time1)
y1 = np.array(flux1)
x1_min = np.min(x1)
x1_max = np.max(x1)
xint = np.linspace(x1_min, x1_max, len(x1))
fl = interpolate.interp1d(x1, y1, kind='linear') #linear interpolation
yintl = fl(xint)

plt.subplot(3,1,1)
plt.title('Original LC')
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.plot(xint,yintl, color = 'blue', linewidth=1)
plt.legend(['Linear'])
plt.grid()

fq = interpolate.interp1d(x1, y1,kind ='quadratic') #spline interpolation
yintq = fq(xint)
plt.subplot(3,1,2)
plt.plot(xint,yintq, color = 'green', linewidth=1)
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.legend(['Spline'])
plt.grid()
#plt.show()
#plt.savefig('Data_LC_interpolation.png')

fg = interpolate.Rbf(x1, y1, function ='gaussian')
yintg = fg(xint)
plt.subplot(3,1,3)
plt.plot(xint,yintq, color = 'red', linewidth=1)
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.legend(['Gaussian'])
plt.grid()
plt.show()


dd = pd.DataFrame({'time':[xint], 'flux':[yintl]})
dd1 = dd.apply(pd.Series.explode)
dd1.to_csv("Mkn501_l1.txt",sep="\t", index = False) 

dd2 = pd.DataFrame({'time':[xint], 'flux':[yintq]})
dd3 = dd2.apply(pd.Series.explode)
dd3.to_csv("Mkn501_q1.txt",sep="\t", index = False) 

dd4 = pd.DataFrame({'time':[xint], 'flux':[yintg]})
dd5 = dd4.apply(pd.Series.explode)
dd5.to_csv("Mkn501_g1.txt",sep="\t", index = False) 

#simulation LC
x2 = np.array(time2)
y2 = np.array(flux2)
x2_min = np.min(x2)
x2_max = np.max(x2)
xint = np.linspace(x2_min, x2_max, len(x2))
fl = interpolate.interp1d(x2, y2, kind='linear') #linear interpolation
yintl = fl(xint)
plt.subplot(3,1,1)
plt.title('Simulation_LC(removed data)')
plt.plot(xint,yintl, color = 'blue', linewidth=1)
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.legend(['Linear'])
plt.grid()

fq = interpolate.interp1d(x2, y2,kind ='quadratic') #spline interpolation
yintq = fq(xint)
plt.subplot(3,1,2)
plt.plot(xint,yintq, color = 'green', linewidth=1)
plt.legend(['Spline'])
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.grid()
#plt.show()
#plt.savefig('Removed_data_LC_interpolation.png')


fg = interpolate.Rbf(x2, y2, function ='gaussian')
yintg = fg(xint)
plt.subplot(3,1,3)
plt.plot(xint,yintq, color = 'red', linewidth=1)
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.legend(['Gaussian'])
plt.grid()
plt.show()




dd4 = pd.DataFrame({'time':[xint], 'flux':[yintl]})
dd5 = dd4.apply(pd.Series.explode)
dd5.to_csv("Mkn501_l2.txt",sep="\t", index = False) 

dd6 = pd.DataFrame({'time':[xint], 'flux':[yintq]})
dd7 = dd6.apply(pd.Series.explode)
dd7.to_csv("Mkn501_q2.txt",sep="\t", index = False) 


dd8 = pd.DataFrame({'time':[xint], 'flux':[yintg]})
dd9 = dd8.apply(pd.Series.explode)
dd9.to_csv("Mkn501_g1.txt",sep="\t", index = False) 




















