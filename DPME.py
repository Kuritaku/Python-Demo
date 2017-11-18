# -*- coding: utf-8 -*-
"""
 Grad-School Lecture Assignment 1
  " Meteorology Research Method"

 This Program computes 
 1. Dry Static Energy                : hd
 2. Potential Temperature            : theta
 3. Moist Static Energy              : hm
 4. Equivalent Potential Temperature : theta_e
 
 Data set is assumed JMA Sonde Data preprocessed as CSV file
 The following application assumes the data set below

  | 0. Pres./hPa | 1. Height/m | 2. Temp./T | 3. Relat. Hum./ phi |
 
"""

import numpy as np
import math 
import csv
import pandas as pd
import matplotlib.pyplot as plt

class DPME:
    
    def __init__(self,ifile, header):
         self.file=ifile
         self.head=header
         
    def read_data(self):
        # csv read
        data = []
        data=pd.read_csv(self.file, dtype=np.float64, header = None, 
                         encoding="utf_8")
        
        # set number of matrix & number of components
        num = data.shape[0] ; elem= data.shape[1]
                                                                              
        return data, num, elem
    
    def main_compute(self):
        data0, num, elem = self.read_data()
        
        # convert DataFrame to array 
        data = np.asarray([[None for col in range(elem)] for row in range(num)])
        for i in range(elem):
            data[:,i] = data0.ix[:,i]
         
        # set coeffcients
        Cp = 1004.0
        Rd = 287.0
        P0 = 1000.0 #hPa
        T0 = 273.16
        g  = 9.8
        L  = 2.5e+6
        
        
        # Results
        hd      = self.dse(data,num,Cp,T0,g)
        theta   = self.pt(data,num,Cp,P0,T0,Rd)
        hm      = self.hm(data,hd,num,T0,L)
        theta_e = self.ept(data,theta,num,Cp,T0,L)
        
        
        return hd,theta, hm, theta_e
    
    def dse(self,data,num,Cp,T0,g):
        # ----------------------------
        # Dry Static Energy
        # ----------------------------
        hd  = [None for row in range(num)]
        for i in range(num):
            a = T0+data[i,2] ; b = data[i,1]
            hd[i] = (Cp*a + g*b)*0.001 # J to kJ
            
        return hd

    def pt(self,data,num,Cp,P0,T0,Rd):
        # ----------------------------
        # Potential Temperature
        # ----------------------------
        theta = [None for row in range(num)]
        alpha = Rd/Cp
        for i in range(num):
            a = P0/data[i,0]
            theta[i] =  (T0+data[i,2])*math.pow(a,alpha) 

        return theta
    
    def hm(self,data,hd,num,T0,L):
        # ----------------------------
        # Moist Static Energy
        # ----------------------------
        Md = 28.96 ; Mv = 18.0 # ave. molecular weight
        alpha = Mv/Md
        
        hm  = [None for row in range(num)]
        for i in range(num):
            # set saturation vaper pressure
            a0 = T0+data[i,2] - 273.16
            b0 = T0+data[i,2] - 35.86
            es = 611 * math.exp(17.27*a0/b0)
            #print(es*0.01,  'Pa')
            
            #actual vaper pressure
            h_ratio = data[i,3]
            if math.isnan(h_ratio) is True : h_ratio = 100.0 
            e = es*h_ratio*0.01
            
            # mixing ratio
            r = alpha * (e/(data[i,0]*100-e))
            #print(r)
            
            hm[i] = hd[i] + L*r*1.0e-3
        
        return hm
    
    def ept(self,data,theta,num,Cp,T0,L):
        # ----------------------------
        # Equivalent Potential Temperature
        # ----------------------------
        theta_e = [None for row in range(num)]

        for i in range(num):
            # set saturation vaper pressure
            a0 = T0+data[i,2] - 273.16
            b0 = T0+data[i,2] - 35.86
            es = 611 * math.exp(17.27*a0/b0)
        
            # set mixing ratio of saturate atm.
            ws = 0.622*es/(data[i,0]*100.0)
            
            # equivalent potential temp.
            alpha = Cp*(data[i,2] + 273.16)
            theta_e[i] = theta[i]*math.exp(L*ws/alpha)
        
        return theta_e
     
if __name__ == "__main__":
    filelist=['kadai1_1.csv' , 'kadai1_2.csv']
    
    for i in range(2):
        app=DPME(ifile=filelist[i], header=1)
        if i   == 0 :
              hd1,theta1, hm1, theta_e1 = app.main_compute()
        elif i == 1 :
              hd2,theta2, hm2, theta_e2 = app.main_compute()
    print("---------------------------------------")
    print("     DPME COMPUTATION  NORMAL END      ")
    print("---------------------------------------")
    
    x1 = np.linspace(1000,100, len(hd1))
    x2 = np.linspace(1000,100, len(hd2))
    plt.figure(figsize=(16,8), dpi=200)
    
    # dry static energy
    plt.subplot(111)
    plt.plot(hd1, x1, c='blue',   label='Down Wind', linewidth=2.8)
    plt.plot(hd2, x2, c='orange', label='Up Wind',   linewidth=2.8)
    plt.gca().invert_yaxis()
    plt.xlabel('Energy  [kJ]',       fontsize=24)
    plt.ylabel('Pressure  [hPa]',    fontsize=24)
    plt.title(' Dry Static Energy ', fontsize=24)
    plt.tick_params(labelsize=18)
    plt.savefig('dse.jpg')
    plt.legend(fontsize=22)
    
    # potential temp.
    plt.figure(figsize=(16,8), dpi=200)
    plt.subplot(111)
    plt.plot(theta1, x1, c='blue',   label='Down Wind', linewidth=2.8)
    plt.plot(theta2, x2, c='orange', label='Up Wind',   linewidth=2.8)
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature [K]',        fontsize=24)
    plt.ylabel('Pressure  [hPa]',        fontsize=24)
    plt.title(' Potential Temperature ', fontsize=24)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=22)
    plt.savefig('theta.jpg')
    
    # Moist static energy
    plt.figure(figsize=(16,8), dpi=200)
    plt.subplot(111)
    plt.plot(hm1, x1, c='blue',   label='Down Wind', linewidth=2.8)
    plt.plot(hm2, x2, c='orange', label='Up Wind',   linewidth=2.8)
    plt.gca().invert_yaxis()
    plt.xlabel('Energy  [kJ]',         fontsize=24)
    plt.ylabel('Pressure  [hPa]',      fontsize=24)
    plt.title(' Moist Static Energy ', fontsize=24)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=22)
    plt.savefig('mse.jpg')
        
    # Equivalent potential temp.
    plt.figure(figsize=(16,8), dpi=200)
    plt.subplot(111)
    plt.plot(theta_e1, x1, c='blue',   label='Down Wind', linewidth=2.8)
    plt.plot(theta_e2, x2, c='orange', label='Up Wind',   linewidth=2.8)
    plt.gca().invert_yaxis()
    plt.xlabel('Temperature [K]',                   fontsize=24)
    plt.ylabel('Pressure  [hPa]',                   fontsize=24)
    plt.title(' Equivalent Potential Temperature ', fontsize=24)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=22)
    plt.savefig('theta_e.jpg')
    plt.show()
    #plt.plot(x, theta1, c='orange', label='Potential Temperature')
    