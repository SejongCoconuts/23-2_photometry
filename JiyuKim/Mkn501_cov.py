import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Original data
df1 = pd.read_csv("Mkn501.txt", sep="\t", names=['time','flux','error'],skiprows=[0])
dd1 = df1.apply(pd.Series.explode)
dd1.to_csv("Mkn501_f0.txt",sep="\t", index = False, header=None) 

# Simulation(removed data)
df2 = pd.read_csv("Mkn501_rs.txt", sep="\t", names=['time','flux'],skiprows=[0])
dd2 = df2.apply(pd.Series.explode)
dd2.to_csv("Mkn501_rsf.txt",sep="\t", index = False,header=None) 

# Original data linear
df3 = pd.read_csv("Mkn501_l1.txt", sep="\t", names=['time','flux'],skiprows=[0])
dd3 = df3.apply(pd.Series.explode)
dd3.to_csv("Mkn501_l1f.txt",sep="\t", index = False,header=None) 

# Simulation(removed data) linear
df4 = pd.read_csv("Mkn501_l2.txt", sep="\t", names=['time','flux'],skiprows=[0])
dd4 = df4.apply(pd.Series.explode)
dd4.to_csv("Mkn501_l2f.txt",sep="\t", index = False,header=None)

# Original data spline
df5 = pd.read_csv("Mkn501_q1.txt", sep="\t", names=['time','flux'],skiprows=[0])
dd5 = df5.apply(pd.Series.explode)
dd5.to_csv("Mkn501_q1f.txt",sep="\t", index = False,header=None) 

# Simulation(removed data) spline
df6 = pd.read_csv("Mkn501_q2.txt", sep="\t", names=['time','flux'],skiprows=[0])
dd6 = df6.apply(pd.Series.explode)
dd6.to_csv("Mkn501_q2f.txt",sep="\t", index = False,header=None) 

