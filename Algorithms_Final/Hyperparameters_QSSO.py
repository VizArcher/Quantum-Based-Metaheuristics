from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random
import numpy as np
import math

df = pd.read_csv(r'C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\Htperparameters_QSSO.csv')

one = df.head(100)['1']
two = df.head(100)['2']
three = df.head(100)['3']
four = df.head(100)['4']
five = df.head(100)['5']
six = df.head(100)['6']
seven = df.head(100)['7'] 

#(g1,epsilon) :- (0.01, 500) , (0.1, 250) , (1 , 100) , (50, 50) , (100, 1) , (250,  0.1) , (500, 0.01) 

X_axis = t = np.linspace(0, 100, 100)

plt.figure(figsize=(6.4,4.8),dpi=150)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 2

plt.plot(X_axis , one, color='r', label=r'g$_1$ = 0.01     $\xi_{\omega m}$ = 500',linewidth=1.5)
plt.plot(X_axis , two, color='c', label=r'g$_1$ = 0.1       $\xi_{\omega m}$ = 250' ,linewidth=1.5)
plt.plot(X_axis , three, color='y', label=r'g$_1$ = 1          $\xi_{\omega m}$ = 100' ,linewidth=1.5)
plt.plot(X_axis , four, color='crimson', label=r'g$_1$ = 50        $\xi_{\omega m}$ = 50' ,linewidth=1.5)
plt.plot(X_axis , five, color='g', label=r'g$_1$ = 100      $\xi_{\omega m}$ = 1' ,linewidth=1.5)
plt.plot(X_axis , six, color='darkkhaki', label=r'g$_1$ = 250      $\xi_{\omega m}$ = 0.1' ,linewidth=1.5)
plt.plot(X_axis , seven, color='royalblue', label=r'g$_1$ = 500      $\xi_{\omega m}$ = 0.01' ,linewidth=1.5)


plt.legend(fontsize=16)

plt.xlabel('Iterations [-]' ,fontsize = 20 )
plt.ylabel('Objective Function ($C_{p}$) [-]' ,fontsize = 20)

plt.tick_params(axis='both',size=8, labelsize=17,direction='inout')
plt.yticks(np.array([0.0,0.05,0.1,0.15,0.20,0.25,0.30,0.35]), fontsize=17)
plt.tight_layout()
plt.show()
