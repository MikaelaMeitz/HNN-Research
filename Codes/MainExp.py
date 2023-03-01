import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.image as mpimg
import os, copy, time, pickle
from urllib.request import urlretrieve
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import h5py as h5

from models import MLP, DHNN, HNN
from train import  get_args, train 

#added by Mikaela Meitz:
from Meitz_utils import get_AMP, get_scale, find_amp_index, integrate_model, get_true_pred


def load_data(file):   
    f = h5.File(file, 'r')
    data = np.array(f['Dataset1'][:,:,0:2])
    return data

def get_data(state, args, j, i, save_path=None):
    
    data = {} #dictionary 
    data['t'] = np.array(range(np.shape(state)[0])).reshape(np.shape(state)[0],1)
    x = state
    dx = (x[1:] - x[:-1]) / (data['t'][1:] - data['t'][:-1])
    dx[:-1] = (dx[:-1] + dx[1:]) / 2  # midpoint rule
    x, t = x[1:], data['t'][1:]

    split_ix = int(state.shape[0] * args.train_split) # train / test split
    if j == i: #original train/test split
        data['x'], data['x_test'] = x[:split_ix], x[split_ix:]
        data['t'], data['t_test'] = 0*x[:split_ix,...,:1], 0*x[split_ix:,...,:1] # H = not time varying
        data['dx'], data['dx_test'] = dx[:split_ix], dx[split_ix:]
        data['time'], data['time_test'] = t[:split_ix], t[split_ix:]
    else: #swaped train/test split so that the testing data starts at the first point
        data['x'], data['x_test'] = x[split_ix:], x[:split_ix]
        data['t'], data['t_test'] = 0*x[split_ix:,...,:1], 0*x[:split_ix,...,:1] # H = not time varying
        data['dx'], data['dx_test'] = dx[split_ix:], dx[:split_ix]
        data['time'], data['time_test'] = t[split_ix:], t[:split_ix]
    return data

def main(MAJOR_FOLDER, FILE):
    main_start_time = time.time()

    data = load_data(FILE)

    #calculate all amplitudes in data array
    dist = get_AMP(data)
    print("Average Amplitudes for each full Trajectory = ", dist, flush = True)

    verbose = False

    #main for loop for all experiments 
    for i in range(20, 400): #change from range(len(data))
       
        scale_amp = get_scale(dist, i)
        all_trials = {}

        #choose new array with scaled amplitudes
        perc=[0.8, 0.9, 1, 1.1, 1.2]
        amp_index = []
        amp_index = find_amp_index(scale_amp, perc)
        print(amp_index, flush = True)

        #make individual folders for each experiment 
        if len(amp_index) > 1: #only make experiment folders for arrays with more than one amplitude
            experiment_path = os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER)
            experiment_file = "Experiment{0}".format(str(i).zfill(3))  
            ex_file = os.makedirs(experiment_file, exist_ok = True)
            new_experiment_path = os.path.join(experiment_path, experiment_file)

            #Spliting data to use for testing on jth trajecotry
            main_dict = {}

            for j in amp_index:
                args = get_args() 
                args.train_split = 0.5 
                j_traj = data[j, :, :]
                j_traj = get_data(j_traj, args, j, i)
                main_dict[j] = j_traj   

            #Training loop on ith trajectory
            for j in amp_index: 

                trial_path = "Trial{0}".format(str(j).zfill(3))
                #new_trial_exp_path = os.path.join(new_experiment_path, trial_path)
                model_save_path = os.path.join(new_experiment_path, trial_path, "models/")
                os.makedirs(model_save_path, exist_ok = True)

                # start_time = time.time()
                if j == i:
                    data_train = main_dict[j]
                    [f(args.seed) for f in [np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all]]
                    #MLP
                    mlp_model = MLP(args.input_dim, args.output_dim, args.hidden_dim) 
                    mlp_results = train(mlp_model, args, data_train)
                    #HNN
                    hnn_model = HNN(args.input_dim, args.hidden_dim)
                    hnn_results = train(hnn_model, args, data_train) 
                    #DHNN
                    dhnn_model = DHNN(args.input_dim, args.hidden_dim)
                    dhnn_results = train(dhnn_model, args, data_train)

                    torch.save(mlp_model.state_dict(), os.path.join(model_save_path, "MLP.pkl"))
                    torch.save(hnn_model.state_dict(), os.path.join(model_save_path, "HNN.pkl"))
                    torch.save(dhnn_model.state_dict(), os.path.join(model_save_path, "DHNN.pkl"))

                    # DHNN_model.save(os.path.join(experiment_path, trial_path, "models/", "DHNN.pkl"))

                    # print("--- %s seconds --- for traning" % (time.time() - start_time))

           #data anylysis on all trajectories 
            for j in amp_index: 
                test_time = []
                test_x = []

                #testing on all trajectories 
                test_time = main_dict[j].get('time_test') #using the jth traj 
                test_x = main_dict[j].get('x_test')

                # get trajectory of true test data
                t_eval = np.squeeze(test_time - test_time.min())
                t_span = [t_eval.min(), t_eval.max()]
                x0 = test_x[0] #0
                true_x = test_x  #reserved test data 

                # integrate along baseline vector field
                integrate_time = time.time()
                mlp_path = integrate_model(mlp_model, t_span, x0, t_eval=t_eval)
                mlp_x = mlp_path['y'].T

                # integrate along HNN vector field
                hnn_path = integrate_model(hnn_model, t_span, x0, t_eval=t_eval)
                hnn_x = hnn_path['y'].T

                # integrate along D-HNN vector field
                dhnn_path = integrate_model(dhnn_model, t_span, x0, t_eval=t_eval)
                dhnn_x = dhnn_path['y'].T

                # print("--- %s seconds --- for integration" % (time.time() - integrate_time))

                #Calculate the average distance 
                data_true = pd.DataFrame(true_x)
                # MSE_time = time.time()

                #HNN average distance of each point from truth
                #np.savetxt('hnn_x.csv', hnn_x[:,:], delimiter=',')
                data_pred = pd.DataFrame(hnn_x)
                HNN_pred, HNN_index, HNN_MSE = get_true_pred(data_true, data_pred) #HNN_pred = distances, HNN_index = coordinates
                HNN_pred2 = np.mean(HNN_pred)

                #DHNN average distance of each point from truth
                data_pred = pd.DataFrame(dhnn_x)
                DHNN_pred, DHNN_index, DHNN_MSE = get_true_pred(data_true, data_pred)
                DHNN_pred2 = np.mean(DHNN_pred)

                #MLP average distance of each point from truth
                data_pred = pd.DataFrame(mlp_x)
                MLP_pred, MLP_index, MLP_MSE = get_true_pred(data_true, data_pred)
                MLP_pred2 = np.mean(MLP_pred)

                #print("--- %s seconds --- for MSE calculation" % (time.time() - MSE_time))

                if verbose == True:
                    print('Average distance between TRUE pts and the closest prediction pts(not consecutive)', flush = True)
                    print('HNN =',HNN_pred2 , flush = True) 
                    print('DHNN =', DHNN_pred2 , flush = True) 
                    print('MLP =', MLP_pred2 , flush = True) 

                    print("\n", flush = True)

                    print('The most accurate model when predicting values closer to the truth values for trajectory', j, 'is:', flush = True)
                    if HNN_pred2 < DHNN_pred2: 
                        if HNN_pred2 < MLP_pred2:
                            print('HNN', flush = True)
                    else:
                        if DHNN_pred2 < MLP_pred2:
                            print('DHNN', flush = True)
                        else: 
                            print('MSE', flush = True)

                    print("\n", flush = True)

                    #Printing  average MSEs
                    print('Average MSE for each model', flush = True)
                    HNN_MSE_avg = np.mean(HNN_MSE)
                    DHNN_MSE_avg = np.mean(DHNN_MSE)
                    MLP_MSE_avg = np.mean(MLP_MSE)

                    print('HNN = ', HNN_MSE_avg, flush = True)
                    print('DHNN = ', DHNN_MSE_avg, flush = True)
                    print('MLP = ', MLP_MSE_avg, flush = True)

                def hamiltonian_fn(coords): # x=q and px=p
                    q, p = np.split(coords,2)
                    alfa = 0.113
                    alfa = alfa/3
                    V = alfa*(q**3)
                    H = np.divide((q**2), 2) + np.divide((p**2), 2) + V #Hamiltonian 
                    #print (q,p)
                    return H

                #plotting
                tpad = 7

                fig = plt.figure(figsize=[20,15], dpi=300)
                plt.subplot(1,3,1)
                plt.title('Predictions on %s' % j, pad=tpad) ; plt.xlabel('$x$') ; plt.ylabel('$p_x$')
                plt.plot(true_x[:,0], true_x[:,1], 'k.', label='Ground truth', markersize = 3)
                # plt.plot(mlp_x[:,0], mlp_x[:,1], 'r.', label='MLP', markersize = 3)
                plt.plot(hnn_x[:,0], hnn_x[:,1], 'g.', label='HNN', markersize =3)
                plt.plot(dhnn_x[:,0], dhnn_x[:,1], 'b.', label='D-HNN', markersize = 3)
                # plt.xlim(-1,1) ; plt.ylim(-1,1)
                plt.legend(fontsize=7, loc='lower right')

                plt.subplot(1,3,2)
                #Plt MSE 
                plt.title('MSE on %s' % j)
                plt.xlabel('Time step') 
                plt.plot(HNN_MSE, 'g-',label ='HNN MSE', linewidth=0.75)
                plt.plot(DHNN_MSE, 'b-',label ='DHNN MSE', linewidth=0.75)
                # plt.plot(MLP_MSE, 'r-',label = 'MLP MSE', linewidth=0.75)
                plt.legend(fontsize=7, loc='lower right')
                # print(HNN_MSE)
                plt.legend(fontsize=7)

                plt.subplot(1,3,3)
                plt.title('Total energy on %s' % j, pad=tpad)
                plt.xlabel('Time step')
                true_e = np.stack([hamiltonian_fn(c) for c in true_x])
                mlp_e = np.stack([hamiltonian_fn(c) for c in mlp_x])
                hnn_e = np.stack([hamiltonian_fn(c) for c in hnn_x])
                dhnn_e = np.stack([hamiltonian_fn(c) for c in dhnn_x])
                plt.plot(t_eval, true_e, 'k-', label='Ground truth', linewidth=1)
                # plt.plot(t_eval, mlp_e, 'r-', label='MLP', linewidth=1)
                plt.plot(t_eval, hnn_e, 'g-', label='HNN', linewidth=1)
                plt.plot(t_eval, dhnn_e, 'b-', label='D-HNN', linewidth=1)
                plt.legend(fontsize=7, loc='lower right')

                plt.tight_layout() 

                #save figures 
                trial_path = "Trial{0}".format(str(j).zfill(3))
                plt.savefig(os.path.join(new_experiment_path, trial_path, "Trial{0}_Figures.png".format(str(j).zfill(3))))
                plt.close()
                hnn = {"Predictions": hnn_x, "MSE": HNN_MSE, "Total Energy": hnn_e} 
                dhnn = {"Predictions": dhnn_x, "MSE": DHNN_MSE, "Total Energy": dhnn_e} 
                mlp = {"Predictions": mlp_x, "MSE": MLP_MSE, "Total Energy": mlp_e} 
                trial_data = {"hnn": hnn, "dhnn": dhnn, "mlp": mlp}

                trial_name = str(j) 
                jth_trial = {trial_name: trial_data}
                
                #save data using pkl
                
                with open(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "Experiment{0}/Trial{1}/Trial{2}_Data.pkl".format(str(i).zfill(3),str(j).zfill(3),str(j).zfill(3))), 'wb') as f:
                    pickle.dump(jth_trial, f)

                all_trials.update(jth_trial)
                
                #file2 = os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "Experiment.{0}.pkl".format(str(i).zfill(3)))
                                     
                with open(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "Experiment.{0}.pkl".format(str(i).zfill(3))), 'wb') as f:
                    pickle.dump(all_trials, f)
                    
            #plot the average MSE for each experiment 
            FILES = glob.glob(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/",MAJOR_FOLDER,'Experiment.*.pkl'), recursive=True)
            FILES.sort()
            all_data = {}
            
            for file in FILES:
                i = file.split(".")[-2]
                with open(file ,'rb') as read_file:
                    trials = pickle.load(read_file)
                all_data[i] = trials
                
            for i in all_data:
                available_js = all_data[i].keys()
    
                tpad = 7
                fig, axs = plt.subplots(2,figsize=[10,10], dpi=300)
    
                js = list(all_data[i].keys())
    
                hnn_MSE = []
                dhnn_MSE = []
                mlp_MSE = []
    
                for j in available_js:  #this one
                    #print(j)
                    mse_hnn = (np.mean(all_data[str(i)][str(j)]["hnn"]["MSE"]))
                    hnn_MSE.append(mse_hnn)
                    mse_dhnn = (np.mean(all_data[str(i)][str(j)]["dhnn"]["MSE"]))
                    dhnn_MSE.append(mse_dhnn)
                    mse_mlp = np.mean(all_data[str(i)][str(j)]["mlp"]["MSE"])                      
                    mlp_MSE.append(mse_mlp)
        
                #plot
                #plt.subplot(1,2,1)
                fig.suptitle('Average MSE for Experiment %s' % i)
                axs[1].plot(js, hnn_MSE, 'og-', label='HNN', markersize = 8)
                axs[1].plot(js, dhnn_MSE, '^b-', label='D-HNN', markersize = 8)
                axs[1].plot(js, mlp_MSE, '*r-', label='MLP', markersize = 8)
                plt.legend(fontsize=7, bbox_to_anchor=(0,0))
                #without MLP
                #plt.subplot(1,2,2) 
                axs[0].plot(js, hnn_MSE, 'og-', label='HNN', markersize = 8)
                axs[0].plot(js, dhnn_MSE, '^b-', label='D-HNN', markersize = 8)
    
    
                for ax in axs.flat:
                    ax.set(xlabel='Trajectory Number', ylabel='Average MSE')
                
                
                trial_path = "Trial{0}".format(str(j).zfill(3))
                plt.savefig(os.path.join(new_experiment_path, "Average_MSE_Fig.png"))
                plt.close()
             
                
    end = time.time()
    print("Time to complete: %s minutes" % ((end-main_start_time)/60), flush = True)
        
if __name__ == '__main__':
    print("START", flush = True)
    FILE = 'Trajectories400.hdf5'
    MAJOR_FOLDER = "02242023"                           
    main(MAJOR_FOLDER, FILE)
    print("DONE!!!", flush = True)
    
    
    