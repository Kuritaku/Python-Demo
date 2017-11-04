# -*- coding: utf-8 -*-
"""
       EnKF - Lorenz 63
  coded by Takuya Kurihana
  
  About EnKF,      https://en.wikipedia.org/wiki/Ensemble_Kalman_filter
  About Lorenz 63, https://en.wikipedia.org/wiki/Lorenz_system
  
  Further thorough theory & algorithm, check scientific papers
"""

import numpy as np
import scipy as sp
import random as rn
from math import sqrt 

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation 

class lorenz_63:
    ####################################################
    #                                                  #
    #                Lorenz 63 Model                   #
    #                                                  #
    ####################################################
    def __init__(self):
        #  coefficeint not change
        self.alpha = 10.0
        self.beta  = 28.0
        self.gamma = 8.0/3.0
        
        
    def lorenz_deriv(self,params,t0):
        x,y,z = params
        # ===== compute the time derivation of lorenz 63 model ======= #
        return [self.alpha*(y-x),
                x*(self.beta-z)-y, 
                x*y -self.gamma*z ]
   
    ###################################################
    #
    #              Discritized methods
    #
    ###################################################
    
    def euler(self, params,t):       
        x_vec = params
        x = x_vec[0]
        y = x_vec[1]
        z = x_vec[2]
        dt = 1.0e-4
        
        if t == 0:
               M=np.array([[-self.alpha, self.alpha,0],
                           [self.beta-z, -1, 0] ,[y , 0, -self.gamma]])
               Mdt = M*dt
               x_vec = x_vec + np.dot(Mdt,x_vec)
        else:
            step = int(1/dt)
            for i in range(step) :             
               M=np.array([[-self.alpha, self.alpha,0],
                           [self.beta-z, -1, 0] ,[y , 0, -self.gamma]])
               Mdt = M*dt
 
               x_vec = x_vec + np.dot(Mdt,x_vec)
               x = x_vec[0]
               y = x_vec[1]
               z = x_vec[2]
        x_t = x_vec
        return x_t

    def runge_kutta_4(self,params,t):
        x_vec = params
        dt = 2.0e-4  # dt must be sufficiently small value
        step = 1     #int(1/dt)
        
        if t == 0:
                s1 = np.asarray(self.lorenz_deriv(x_vec,t))               
                s2 = np.asarray(self.lorenz_deriv(x_vec+dt*s1[:]/2,t+dt/2))             
                s3 = np.asarray(self.lorenz_deriv(x_vec+dt*s2[:]/2,t+dt/2))
                s4 = np.asarray(self.lorenz_deriv(x_vec+dt*s3[:],t+dt))
                x_vec = x_vec + dt/6*(s1+2*s2+2*s3+s4)
        else:
              for i in range(step) :         
                s1 = np.asarray(self.lorenz_deriv(x_vec,t))               
                s2 = np.asarray(self.lorenz_deriv(x_vec+dt*s1[:]/2,t+dt/2))             
                s3 = np.asarray(self.lorenz_deriv(x_vec+dt*s2[:]/2,t+dt/2))
                s4 = np.asarray(self.lorenz_deriv(x_vec+dt*s3[:],t+dt))
                
                x_vec = x_vec + dt/6*(s1+2*s2+2*s3+s4)
            
        x_t = x_vec
        return x_t

if __name__ == "__main__":
    lorenz63 = lorenz_63()
    init_param = 0.8
    N = 1               # 1 trajectory particle
    ens_num = 3
    gues_coef = 0.0001  # pertb noize adjust
    obs_coef = 0.8
    inflation = 1.1
    R_conf = 1.0
    dt = 20000          # discrtized scale
    dtime = 200         # time span 
    ntime = np.linspace(0,dtime,dt)
    etime = len(ntime)
    
    # initial condition
    x0 = np.reshape([ 13.78301946, -12.52368443, 9.55948302], (1,3))
    
    # ======================================= #
    #                EnKF 
    # ======================================= #  
    
    def nature():
         t = ntime
         return np.asarray([sp.integrate.odeint(lorenz63.lorenz_deriv, x0_num, t) 
                          for x0_num in x0])       
    
    def obs_noize():
        #set gaussian distribution : mean = 0, deviation = 1.0
        mean  = 0.0
        sigma = 1.0
        noize = np.random.normal(mean,sigma,10*etime)
        nlist = noize.tolist()
        nselect = rn.sample(nlist,3*etime)
        noize = np.reshape(nselect,(1,etime,3))
        return noize
    
    x_t = nature() #true(1,time,3)
    gnoize = obs_noize()
    obs = x_t + obs_coef*gnoize # observation
    obs = np.reshape(obs,(etime,3)) #obs(time,3)
   
    #--------------------------------------------------------#    
    #                  foreword model  methods
    #--------------------------------------------------------#
    def elforecast(itime,x):
        # euler     
        return lorenz63.euler(x,itime) 
    
    def RKforecast(itime,x):
        # 4th Runge Kutta
        return lorenz63.runge_kutta_4(x,itime)
    
    def RKaforecast(itime,x_solver):
        # 4th Runge Kutta
        return lorenz63.runge_kutta_4(x_solver,itime)
    
    def odeint(itime,x):
        tstep=np.arange(0,itime+1)
        x = np.reshape(x,(1,3))
        return np.asarray([sp.integrate.odeint(lorenz63.lorenz_deriv, x0_num, tstep) 
                          for x0_num in x])   
    #--------------------------------------------------------#

    def ensemble_pertb():
        # set ensemble forcast perturbation
        mean  = 0.0
        sigma = 1.0
        pertb = np.random.normal(mean,sigma,ens_num*3) # gaussian
        pertb = np.reshape(pertb, (ens_num,3))
        pertbf = gues_coef * pertb # forecast perturbation   
        return pertbf
 
    def operator():
        # forecast ensemble perturbation
        ens_vec =ens_mean
        ens_vec = np.reshape(ens_vec,(3,1))
        X =  ens_vec - xtrue
       
        # Baack gorund error covariance
        unbias_var_coef = 1.0/float(ens_num-1)
        Pf = unbias_var_coef * np.outer(X, X)
        Pf = Pf * inflation 
        
        # obsercvation error covariance off-set off diagonal components
        R = ([0.5, 0.0, 0.0],[0.0, 0.2,0.0],[0.0,0.0,0.1])
        return Pf,R
     
    def kalman_gain():
        Pf, R= operator()
        # compute Kalman Gain 
        # =============================
        #    K = Pf * (R + Pf)^-1
        # =============================
        A = R + Pf
        KK = np.dot(Pf,np.linalg.inv(A))
        return KK
    
    # No DA exp in middlle of the timestep
    def no_da_exp1(xf_nd):
         x0_nd = xf_nd
         t = ntime[200:5000]   # chose No da period 
         return np.asarray([sp.integrate.odeint(lorenz63.lorenz_deriv, x0_num, t) 
                          for x0_num in x0_nd])  
    
    xb = np.zeros((etime,3))
    xa = np.zeros((etime,3))
    xe = np.zeros((etime+1,3))
    ens_mem = np.zeros((ens_num,3))
    ferror = np.zeros((etime))
    aerror = np.zeros((etime))
    oerror = np.zeros((etime))
    
    for itime in range(0,etime):
        if itime == 0 : x = init_param*np.reshape(x0,(3,1))  
        
        # select disctizetion Euler or Runge Kutta
        x = RKforecast(itime,x)
        
        xtrue =x_t[0,itime,:] 
        xtrue = np.reshape(xtrue,(3,1))
        xf = np.reshape(x,(1,3))
        xb[itime,:] = xf
        
        #----------------------------
        #    No DA exp. 
        #----------------------------
        if itime == 200: 
            xf_nd = xf
            xt_nd = no_da_exp1(xf_nd)
            
        # set ensemble forecast vector
        vecf = [ xf for i in range(ens_num)]
        farray = np.asarray(vecf).reshape(ens_num,3) # list to array
        farray = np.reshape(farray, (ens_num,3))
        
        # ==========================================
        # ens(x_forecast) = ens_mean + perturbation 
        # ==========================================
        prtb_f = ensemble_pertb()
        ensf =  farray + prtb_f 
        for ienum in range(ens_num):
                  x_solver=ensf[ienum,:]
                  ens_mem[ienum,:] = RKaforecast(itime,x_solver)
        ens_mean = sum(ens_mem)/float(ens_num)
        xe_mean = np.reshape(ens_mean,(1,3))
        
        xe[0,:] = x0
        xe[itime,:] = xe_mean
        K = kalman_gain() 
        
        #  analysis equation in EnKF
        # ===========================
        #   xa = xb + K*(y - xb) 
        # ===========================            
        Inc = obs[itime,:] - xf[:]
        Inc = np.reshape(Inc,(3))
        anl = xf + np.dot(K,Inc)
        xa[itime,:] = anl
        x_a = np.reshape(anl,(3,1))
          
        #oevrlap data 
        x = x_a
   
        #rmse
        ferror[itime] = sqrt((xb[itime,0]-x_t[0,itime,0])**2)
        aerror[itime] = sqrt((xa[itime,0]-x_t[0,itime,0])**2)
        oerror[itime] = sqrt((obs[itime,0]-x_t[0,itime,0])**2)
    
    # Show Results 
    for it in range(100) : print(x_t[0,it,0], xb[it,0])
    
    # No DA forecast 
    def diff():
         x_diff = init_param*x0 
         t = ntime
         return np.asarray([sp.integrate.odeint(lorenz63.lorenz_deriv, x0_num, t) 
                          for x0_num in x_diff])      
    xf_diff = diff()

    #==================================================================#
    
    end_time = 5000
    time = np.arange(0,end_time)
    
    xf_nd = np.asarray([[None for col in range(3)] for row in range(end_time)])
    nderror=np.asarray([None for row in range(etime)])
    for j in range(4800):
        xf_nd[200+j,:] = xt_nd[0,j,:]
        nderror[200+j] = sqrt((xf_nd[200+j,0]-x_t[0,200+j,0])**2)
    
    
    fig =plt.figure(figsize=(16,8),)
    plt.subplot(111)
    plt.xlim(0,end_time)
    plt.plot(time[:],xb[0:end_time,0],c="Red",linewidth=2.8
             , label="DA ")
    plt.plot(time[:],xf_nd[:end_time,0],c="blue",linewidth=2.8
             , label="No DA ")

    plt.title("Trajectory Lorenz 63")
    plt.legend(loc='lower right')
    filename = "trajectoryl63_enkf_1.png"
    #plt.savefig(filename)
    
    
    fig =plt.figure(figsize=(16,8),)
    plt.subplot(111)
    end_time = 1000
    time = np.arange(0,end_time)
    plt.xlim(0,end_time)
    plt.plot(time[:],nderror[:end_time],c="blue",linewidth=2.8
              ,label="No DA")
    plt.plot(time[:],ferror[0:end_time],c="red"
             ,label="DA",linewidth=2.8)
    plt.title("RMSE Lorenz 63 EnKF")
    plt.legend(loc='higher right')
    filename = "rmsel63_enkf_1.png"
    #plt.savefig(filename)
    #plt.show()
    
    jx = xf_diff[0,:-1,0]
    jy = xf_diff[0,:-1,1]
    jz = xf_diff[0,:-1,2]
    ax = xa[:etime,0]
    ay = xa[:etime,1]
    az = xa[:etime,2]
    
    #fig =plt.figure(figsize=(16,8), projection='3d')
    plt.subplot(111, projection='3d')
    plt.plot(jx,jy, jz, c="blue", label='No DA')
    plt.plot(ax,ay, az, c="red",  label='DA')
    plt.title("Lorenz 63 3D Trajectory")
    plt.legend(loc='lower right')
    filename = "3Dl63_enkf_1.png"
    #plt.savefig(filename)
    plt.show()
 
    end_time = 5000
    time = np.arange(0,end_time)  
    fig =plt.figure(figsize=(16,8),)
    plt.subplot(111)
    plt.plot(time[:],xb[0:end_time,0],c="Red",linewidth=2.8
             , label="DA ")
    plt.plot(time[:],xf_diff[0,0:end_time,0],c="k",linewidth=2.8
             , label="Forecast from initial")
    plt.legend(loc='lower right')
    filename = "forecast.png"
    #plt.savefig(filename)
    
    ########################################################################
    #
    #                            Animation
    #
    ########################################################################
    # Set up figure & 3D axis for animation
    fig = plt.figure(dpi = 400)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')
    end_time = 100


    # set up lines and points
    lines = sum([ax.plot([],[],[],'-', c="blue") for c in range(end_time)] , [] )
    pts   = sum([ax.plot([],[],[], 'o', c="blue") for c in range(end_time)], [] )
    alines = sum([ax.plot([],[],[],'-', c="red") for c in range(end_time)] , [] )
    apts   = sum([ax.plot([],[],[], 'o', c="red") for c in range(end_time)], [] )
    tlines = sum([ax.plot([],[],[],'-', c="k",linewidth=0.5) for c in range(end_time)] , [] )
    tpts   = sum([ax.plot([],[],[], 'o', c="k") for c in range(end_time)], [] )

    # prepare the axes limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # initialization function: plot the background of each frame
    def init():
        for line, pt, aline, apt in zip(lines, pts, alines, apts):
            line.set_data([], [])
            line.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])

            aline.set_data([], [])
            aline.set_3d_properties([])
            apt.set_data([], [])
            apt.set_3d_properties([])
        
        for tline, tpt in zip(tlines, tpts):
            tline.set_data([], [])
            tline.set_3d_properties([])
            tpt.set_data([], [])
            tpt.set_3d_properties([])

        return lines + pts
  
    xb = np.reshape(xa,(1,etime,3))
    # animation function.  This will be called sequentially with the frame number
    def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
        i = (2 * i) % 2000

        for line, pt, xi in zip(lines, pts, xf_diff):
            x, y, z = xi[:i].T
            line.set_data(x, y)
            line.set_3d_properties(z)
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])

        for aline, apt, axi in zip(alines, apts, xb):
            iax, iay, iaz = axi[:i].T
            aline.set_data(iax, iay)
            aline.set_3d_properties(iaz)
            apt.set_data(iax[-1:], iay[-1:])
            apt.set_3d_properties(iaz[-1:])

        for tline, tpt, txi in zip(tlines, tpts, x_t):
            itx, ity, itz = txi[:i].T
            tline.set_data(itx, ity)
            tline.set_3d_properties(itz)
            tpt.set_data(itx[-1:], ity[-1:])
            tpt.set_3d_properties(itz[-1:])

        ax.view_init(30, 0.3 * i)
        fig.canvas.draw()
        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1000, interval=50, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    anim.save('lda_noda_L63.mp4', fps=15,dpi=200, extra_args=['-vcodec', 'libx264'])

    plt.show()

    