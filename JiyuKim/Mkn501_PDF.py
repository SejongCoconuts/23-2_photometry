from DELCgen import *
import scipy.stats as st
import matplotlib.pyplot as plt


df1 = "Mkn501_f0.txt" # Original data
df1lc = Load_Lightcurve(df1,tbin=100)
d1 = df1lc.Plot_PDF()
plt.savefig('Original_PDF.png')

df2 = "Mkn501_rsf.txt" # Simulation(removed data)
df2lc = Load_Lightcurve(df1,tbin=100)
#d2 = df2lc.Plot_PDF()
#plt.savefig('Removed_data _PDF.png')

Comparison_Plots([df1lc,df2lc],names=["Data LC","Removed data LC"],bins=25)

df3 = "Mkn501_l1f.txt" # Original data linear
df3lc = Load_Lightcurve(df1,tbin=100)
#d3 = df3lc.Plot_PDF()
#plt.savefig('Original_data linear_PDF.png')

df4 = "Mkn501_l2f.txt" # Simulation(removed data) linear
df4lc = Load_Lightcurve(df1,tbin=100)
#d4 = df4lc.Plot_PDF()
#plt.savefig('Removed_data_linear_PDF.png')

Comparison_Plots([df3lc,df4lc],names=["Data Linear LC","Removed data linear LC"],bins=25)

df5 = "Mkn501_q1f.txt" # Original data spline
df5lc = Load_Lightcurve(df1,tbin=100)
#d5 = df5lc.Plot_PDF()
#plt.savefig('Original_data spline_PDF.png')

df6 = "Mkn501_q2f.txt" # Simulation(removed data) spline
df6lc = Load_Lightcurve(df1,tbin=100)
#d6 = df6lc.Plot_PDF()
#plt.savefig('Removed_data_spline_PDF.png')

Comparison_Plots([df5lc,df6lc],names=["Data spline LC","Removed data spline LC"],bins=25)