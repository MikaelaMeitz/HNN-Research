import torch
import numpy as np
import os, copy, time
import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

#find amplitudes of each trajectory in data set from origin
def get_AMP(data): 
    dist = []
    for i in range(len(data)): #first loop goes into each trajecotry "sheet"
        dist2=[]
        for j in range(len(data[i])):#second loop goes into seach trajectory "row" which holds an (x,y)
            temp = np.sqrt((data[i,j,0]) ** 2 + (data[i,j,1]) ** 2) #distance from the origin to the point 
            dist2.append(temp)
            Amp = np.mean(dist2)
        dist.append(Amp)
    dist = np.array(dist)
    return dist  

#scale all amplitudes in dist array 
def get_scale(dist, i): 
    scale = []
    for k in range(len(dist)): 
        x = dist[k]/dist[i]
        x = np.round(x, 2)
        scale.append(x)
    scale = np.array(scale)
    return scale

#to search amp array and find the scaling trajectories
def find_amp_index(scale_amp, i):
    perc = [0.9, 0.95,1.05, 1.10]
    amp_index = []
    print(i)
    amp_perc = []
    for p in perc:
        for k in range(len(scale_amp)): #k is the index for the amplitude we want to use 
            if scale_amp[k] == p: #or k == i: 
                print(k, scale_amp[k])
                amp_index.append(k)
                amp_perc.append(p)
                break
 
            if k == i: 
                if k not in amp_index: 
                    print(k, scale_amp[k])
                    amp_index.append(k)
                    amp_perc.append(1)
            
    return amp_index, amp_perc #returns the amplidue array index 


def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,2)
        t = torch.zeros_like(x[...,:1]) 
        dx = model(x, t=t).data.numpy().reshape(-1) 
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

def get_true_pred(data_true, data_pred): 
    NEWdist = []
    MSE = []
    index_pred = []
    for k in range(len(data_true)): #giant loop to calculate distance 
        dist = []
        dist2 = []
        for i in range(len(data_pred)): #calculating the euclidian distance between A TRUE pt and ALL prediction pts
            temp = np.sqrt(((data_true.iloc[k, 0] - data_pred.iloc[i, 0]) ** 2) + ((data_true.iloc[k, 1] - data_pred.iloc[i, 1]) **2)) #euclidean dist
            temp2 = ((data_true.iloc[k, 0] - data_pred.iloc[i, 0]) ** 2) + ((data_true.iloc[k, 1] - data_pred.iloc[i, 1]) **2)#similar to L2 norm, SE (not sqrt)
            dist.append(temp) #appaned and make array of all calculated distances
            dist2.append(temp2)
        #out of for loop 
        dist = np.array(dist)
        dist2 = np.array(dist2)
        index_dist = np.argsort(dist)
        dist.sort() #next, pick which distance is the smallest from the "dist" array
        dist2.sort()
        NEWdist.append(dist[0]) #append "smallest value" to list every loop 
        MSE.append(dist2[0])
        index_pred.append(data_pred.iloc[index_dist[0]])
    #out of big loop
    index_pred = np.array(index_pred)
    NEWdist = np.array(NEWdist)
    MSE = np.array(MSE)
    # MSE = np.mean(MSE)
    return NEWdist, index_pred, MSE
