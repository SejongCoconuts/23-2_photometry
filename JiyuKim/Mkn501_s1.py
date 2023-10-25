import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import curve_fit

df1 = pd.read_csv("Mkn501.txt", sep="\t", names=['time','flux','flux_err'],skiprows=[0])
df2 = pd.read_csv("Mkn501_s.dat", sep =" ", names =['time','flux'], skiprows=[0])


time1 = df1['time']
flux1 = df1['flux']
time2 = df2['time']
flux2 = round(df2['flux'],10)

df = pd.DataFrame({'time':[time1], 'flux_d':[flux1],'flux_s':[flux2]})
df_s = df.apply(pd.Series.explode)
df_s.to_csv("Mkn501_ds.csv",sep="\t", index = False) 

#plt.plot(time2,flux2)
plt.title('Simulation LC')
plt.xlabel('Julian Date')
plt.ylabel('Photon Flux [0.1-100 GeV](photons cm-2 s-1)')
plt.grid()
#plt.show()
#plt.savefig('Simulation LC')

#removed data

df3 = pd.read_csv("Mkn501_ds.csv", sep="\t", names=['time','flux_d','flux_s'],skiprows=[0])

print(len(df3))

time = df3['time']
ff_d = df3['flux_d']
ff_s = df3['flux_s']

time_f,flux_f,time_f0,flux_f0 = [],[],[],[]
for i in range(len(df3)):
    sigma = 0.3
    ff1 =ff_s[i]
    ff2= ff_d[i]
    tt = time[i]
    dd = ff1*sigma
    dd_low = ff1-dd
    dd_high = ff1+dd
#    print([dd,dd_low,dd_high])
    if dd_low >= ff2 or ff2>= dd_high:
        time_f.append(tt)
        flux_f.append(ff1)
    else:
        time_f0.append(tt)
        flux_f0.append(0)

#creat "data-removed" file
df4 = pd.DataFrame({'time':[time_f], 'flux':[flux_f]})
df_f = df4.apply(pd.Series.explode)
df_f.to_csv("Mkn501_r.txt",sep="\t", index = False) 

df0 = pd.DataFrame({'time':[time_f0], 'flux_d':[flux_f0]})
df_0 = df0.apply(pd.Series.explode)
df_0.to_csv("Mkn501_r0.txt",sep="\t", index = False)  



df5 = pd.read_csv("Mkn501_r.txt", sep="\t", names=['time','flux_f'],skiprows=[0])
tt_f = df5['time']
ff_f = df5['flux_f']
plt.title('Lightcurves')
plt.plot(time,ff_d)
plt.plot(time,ff_s)
plt.plot(tt_f,ff_f)
plt.legend(['Original LC','Simulation LC','Simulation(removed data) LC'])
plt.grid()
#plt.show()
plt.savefig('Lightcurves.png')

