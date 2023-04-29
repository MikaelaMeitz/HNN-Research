#Code adapted from:  S. Greydanus, M. Dzamba, J. Yosinski. Hamiltonian Neural Networks. arXiv preprint arXiv:1906.01563, 2019.

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
    main_times = [] #store all times for all experiments 
    training_time = [] #store training times for all experiments
    training_js = [] #store training index times for all experiments
        
    data = load_data(FILE)

    #calculate all amplitudes in data array
    dist = get_AMP(data)
    print("Average Amplitudes for each full Trajectory = ", dist, flush = True)

    verbose = False

    #main for loop for "all experiments"
    traj_list = [20,50,75,100,125,129,130,135,141,150,200,249,250,298,300]
    
    for i in range(20,25): #change from range(len(data)) 
       
        scale_amp = get_scale(dist, i)
        all_trials = {}
        
        experiment_start_time = time.time()
        experiment_time = [] #store experiment times for all experiments
        experiment_is = [] #store experiment index times for all experiments
        
        integration_times = [] #store integration times for each trial 
        MSE_times = [] #store MSE times for each trial 
         
        #choose new array with scaled amplitudes
        perc=[0.9, 0.95, 1.0, 1.05, 1.10]
        amp_index = []
        amp_perc = []
        amp_index, amp_perc = find_amp_index(scale_amp, i)
        print(amp_index, amp_perc, flush = True)

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

                #start training time
                training_start_time = time.time()
                
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
                    
                    training_time.append((time.time() - training_start_time)/60)
                    training_js.append(j)
                    print("--- %d minutes --- for traning" % ((time.time() - training_start_time)/60))
            #out of training loop 

            #data anylysis on each trajectory 
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

                #start integration time
                integration_start_time = time.time()
                
                # integrate along baseline vector field
                #mlp_model = MLP(args.input_dim, args.output_dim, args.hidden_dim) 
                mlp_path = integrate_model(mlp_model, t_span, x0, t_eval=t_eval)
                mlp_x = mlp_path['y'].T

                # integrate along HNN vector field
                #hnn_model = HNN(args.input_dim, args.hidden_dim)
                hnn_path = integrate_model(hnn_model, t_span, x0, t_eval=t_eval)
                hnn_x = hnn_path['y'].T

                # integrate along D-HNN vector field
                #dhnn_model = DHNN(args.input_dim, args.hidden_dim)
                dhnn_path = integrate_model(dhnn_model, t_span, x0, t_eval=t_eval)
                dhnn_x = dhnn_path['y'].T
        
                integration_times.append((time.time() - integration_start_time)/60)
                print("--- %d minutes --- for integration" % ((time.time() - integration_start_time)/60))
                

                #Calculate the average distance 
                data_true = pd.DataFrame(true_x)
                
                #start mse time
                MSE_start_time= time.time()

                #HNN average distance of each point from truth
                data_pred = pd.DataFrame(hnn_x)
                HNN_pred, HNN_index, HNN_MSE = get_true_pred(data_true, data_pred) 
                HNN_pred2 = np.mean(HNN_pred)

                #DHNN average distance of each point from truth
                data_pred = pd.DataFrame(dhnn_x)
                DHNN_pred, DHNN_index, DHNN_MSE = get_true_pred(data_true, data_pred)
                DHNN_pred2 = np.mean(DHNN_pred)

                #MLP average distance of each point from truth
                data_pred = pd.DataFrame(mlp_x)
                MLP_pred, MLP_index, MLP_MSE = get_true_pred(data_true, data_pred)
                MLP_pred2 = np.mean(MLP_pred)

                MSE_times.append((time.time() - MSE_start_time)/60)
                print("--- %d minutes --- for MSE calculation" % ((time.time() - MSE_start_time)/60))
                
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

                #plotting w/MLP
                tpad = 7
                fig = plt.figure(figsize=[9,3], dpi=300)
                
                #plt predictions 
                plt.subplot(1,3,1)
                plt.title('Predictions on %s' % j, pad=tpad) ; plt.xlabel('$x$') ; plt.ylabel('$p_x$')
                plt.plot(true_x[:,0], true_x[:,1], 'ks', label='Ground truth', markersize = 2)
                plt.plot(hnn_x[:,0], hnn_x[:,1], 'go', label='HNN', markersize =2)
                plt.plot(dhnn_x[:,0], dhnn_x[:,1], 'b^', label='D-HNN', markersize = 2, alpha=0.4)
                plt.plot(mlp_x[:,0], mlp_x[:,1], 'r*', label='MLP', markersize = 3, alpha=0.2)
                plt.legend(fontsize=7, loc='upper left')
                           
                #plot MSE 
                plt.subplot(1,3,2)
                plt.title('SE on %s' % j)
                plt.xlabel('Time step') 
                plt.plot(HNN_MSE, 'g--',label ='HNN SE', linewidth=0.75, markersize = 2)
                plt.plot(DHNN_MSE, 'b:',label ='DHNN SE', linewidth=0.75, markersize = 2, alpha=0.4)
                plt.plot(MLP_MSE, 'r-.',label = 'MLP SE', linewidth=0.75, markersize = 2, alpha=0.2)
                plt.legend(fontsize=7, loc='upper left')

                #plot total energy 
                plt.subplot(1,3,3)
                plt.title('Total energy on %s' % j, pad=tpad)
                plt.xlabel('Time step')
                true_e = np.stack([hamiltonian_fn(c) for c in true_x])
                mlp_e = np.stack([hamiltonian_fn(c) for c in mlp_x])
                hnn_e = np.stack([hamiltonian_fn(c) for c in hnn_x])
                dhnn_e = np.stack([hamiltonian_fn(c) for c in dhnn_x])
                plt.plot(t_eval, true_e, 'k-', label='Ground truth', linewidth=1, markersize = 2)
                plt.plot(t_eval, hnn_e, 'g--', label='HNN', linewidth=1, markersize = 2)
                plt.plot(t_eval, dhnn_e, 'b:', label='D-HNN', linewidth=1, markersize = 2, alpha=0.4)
                plt.plot(t_eval, mlp_e, 'r-.', label='MLP', linewidth=1, markersize = 2, alpha=0.2)
                plt.legend(fontsize=7, loc='upper left')

                plt.tight_layout() 

                #save figures 
                trial_path = "Trial{0}".format(str(j).zfill(3))
                plt.savefig(os.path.join(new_experiment_path, trial_path, "Trial{0}_Figures_W_MLP.png".format(str(j).zfill(3))))
                plt.close()
                
                #NEW plotting NO MLP
                tpad = 7
                fig = plt.figure(figsize=[9,3], dpi=300)
                
                #plt predictions 
                plt.subplot(1,3,1)
                plt.title('Predictions on %s' % j, pad=tpad) ; plt.xlabel('$x$') ; plt.ylabel('$p_x$')
                plt.plot(true_x[:,0], true_x[:,1], 'ks', label='Ground truth', markersize = 2)
                plt.plot(hnn_x[:,0], hnn_x[:,1], 'go', label='HNN', markersize =2)
                plt.plot(dhnn_x[:,0], dhnn_x[:,1], 'b^', label='D-HNN', markersize = 2, alpha=0.4)
                plt.legend(fontsize=7, loc='upper left')

                #plot MSE 
                plt.subplot(1,3,2)
                plt.title('SE on %s' % j)
                plt.xlabel('Time step') 
                plt.plot(HNN_MSE, 'g--',label ='HNN SE', linewidth=0.75, markersize = 2)
                plt.plot(DHNN_MSE, 'b:',label ='DHNN SE', linewidth=0.75, markersize = 2, alpha=0.4)
                plt.legend(fontsize=7, loc='upper left')

                #plot total energy
                plt.subplot(1,3,3)
                plt.title('Total energy on %s' % j, pad=tpad)
                plt.xlabel('Time step')
                true_e = np.stack([hamiltonian_fn(c) for c in true_x])
                mlp_e = np.stack([hamiltonian_fn(c) for c in mlp_x])
                hnn_e = np.stack([hamiltonian_fn(c) for c in hnn_x])
                dhnn_e = np.stack([hamiltonian_fn(c) for c in dhnn_x])
                plt.plot(t_eval, true_e, 'k-', label='Ground truth', linewidth=1, markersize = 2)
                plt.plot(t_eval, hnn_e, 'g--', label='HNN', linewidth=1, markersize = 2)
                plt.plot(t_eval, dhnn_e, 'b:', label='D-HNN', linewidth=1, markersize = 2, alpha=0.4)
                plt.legend(fontsize=7, loc='upper left')
                

                plt.tight_layout() 

                #save figures 
                trial_path = "Trial{0}".format(str(j).zfill(3))
                plt.savefig(os.path.join(new_experiment_path, trial_path, "Trial{0}_Figures.png".format(str(j).zfill(3))))
                plt.close()
                
                
                #NEW plotting NO MLP or DHNN
                tpad = 7
                fig = plt.figure(figsize=[9,3], dpi=300)
                
                #plt predictions 
                plt.subplot(1,3,1)
                plt.title('Predictions on %s' % j, pad=tpad) ; plt.xlabel('$x$') ; plt.ylabel('$p_x$')
                plt.plot(true_x[:,0], true_x[:,1], 'ks', label='Ground truth', markersize = 2)
                plt.plot(hnn_x[:,0], hnn_x[:,1], 'go', label='HNN', markersize =2)
                plt.legend(fontsize=7, loc='upper left')

                #plot MSE 
                plt.subplot(1,3,2)
                plt.title('SE on %s' % j)
                plt.xlabel('Time step') 
                plt.plot(HNN_MSE, 'g--',label ='HNN SE', linewidth=0.75, markersize = 2)
                plt.legend(fontsize=7, loc='upper left')

                #plot total energy
                plt.subplot(1,3,3)
                plt.title('Total energy on %s' % j, pad=tpad)
                plt.xlabel('Time step')
                true_e = np.stack([hamiltonian_fn(c) for c in true_x])
                hnn_e = np.stack([hamiltonian_fn(c) for c in hnn_x])
                plt.plot(t_eval, true_e, 'k-', label='Ground truth', linewidth=1, markersize = 2)
                plt.plot(t_eval, hnn_e, 'g--', label='HNN', linewidth=1, markersize = 2)
                plt.legend(fontsize=7, loc='upper left')
                

                plt.tight_layout() 

                #save figures 
                trial_path = "Trial{0}".format(str(j).zfill(3))
                plt.savefig(os.path.join(new_experiment_path, trial_path, "Trial{0}_Figures_HNN.png".format(str(j).zfill(3))))
                plt.close()
                
                amp_percentage = {"Amp Percentage": amp_perc} #added 03222023
                ground_truth = {"Predictions": true_x, "Total Energy": true_e, "Time Step": t_eval} #added 03142023
                hnn = {"Predictions": hnn_x, "MSE": HNN_MSE, "Total Energy": hnn_e} 
                dhnn = {"Predictions": dhnn_x, "MSE": DHNN_MSE, "Total Energy": dhnn_e} 
                mlp = {"Predictions": mlp_x, "MSE": MLP_MSE, "Total Energy": mlp_e} 
                trial_data = {"amp_percentage": amp_percentage, "ground_truth": ground_truth, "hnn": hnn, "dhnn": dhnn, "mlp": mlp}

                trial_name = str(j) 
                jth_trial = {trial_name: trial_data}
                
                #save data using pkl
                with open(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "Experiment{0}/Trial{1}/Trial{2}_Data.pkl".format(str(i).zfill(3),str(j).zfill(3),str(j).zfill(3))), 'wb') as f:
                    pickle.dump(jth_trial, f)
                
                all_trials.update(jth_trial)
                #out of loop
                                         
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
                
                available_js = all_data[str(i)].keys()
                
                js = list(all_data[str(i)].keys())

                hnn_MSE = []
                dhnn_MSE = []
                mlp_MSE = []
                amp_percent =[]

                for j in available_js: 
                    #print('in loop')
                    amp_percent = all_data[str(i)][str(j)]["amp_percentage"]["Amp Percentage"]
                    mse_hnn = (np.mean(all_data[str(i)][str(j)]["hnn"]["MSE"]))
                    hnn_MSE.append(mse_hnn)
                    mse_dhnn = (np.mean(all_data[str(i)][str(j)]["dhnn"]["MSE"]))
                    dhnn_MSE.append(mse_dhnn)
                    mse_mlp = np.mean(all_data[str(i)][str(j)]["mlp"]["MSE"])                      
                    mlp_MSE.append(mse_mlp)

                #plot w/MLP
                tpad = 7
                fig = plt.figure(figsize=[15,10], dpi=300)
                plt.title('Average MSE for Experiment %s' % i, pad=tpad) 
                plt.xlabel('Trajectory Number') 
                plt.ylabel('Average MSE') 
                plt.plot(js, hnn_MSE, 'og-', label='HNN', markersize = 8)
                plt.plot(js, dhnn_MSE, '^b-', label='D-HNN', markersize = 8)
                plt.plot(js, mlp_MSE, '*r-', label='MLP', markersize = 8)
                plt.legend(fontsize=7, loc='upper right')
                plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns",MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "Average_Traj_MSE_W_MLP_Fig.png"))
                plt.close()

                #without MLP
                tpad = 7
                fig = plt.figure(figsize=[15,10], dpi=300)
                plt.title('Average MSE for Experiment %s' % i, pad=tpad) 
                plt.xlabel('Trajectory Number') 
                plt.ylabel('Average MSE') 
                plt.plot(js, hnn_MSE, 'og-', label='HNN', markersize = 8)
                plt.plot(js, dhnn_MSE, '^b-', label='D-HNN', markersize = 8)
                plt.legend(fontsize=7, loc='upper right')
                trial_path = "Trial{0}".format(str(j).zfill(3))
                plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "Average_Traj_MSE_Fig.png"))
                plt.close()
       
                #plot w/MLP
                tpad = 7
                fig = plt.figure(figsize=[15,10], dpi=300)
                plt.title('Average MSE for Experiment %s' % i, pad=tpad) 
                plt.xlabel('Scaled Amplitude') 
                plt.ylabel('Average MSE') 
                plt.plot(amp_percent, hnn_MSE, 'og-', label='HNN', markersize = 8)
                plt.plot(amp_percent, dhnn_MSE, '^b-', label='D-HNN', markersize = 8)
                plt.plot(amp_percent, mlp_MSE, '*r-', label='MLP', markersize = 8)
                plt.legend(fontsize=7, loc='upper right')
                plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "Average_Scale_MSE_W_MLP_Fig.png"))
                plt.close()
                                        
                #without MLP
                tpad = 7
                fig = plt.figure(figsize=[15,10], dpi=300) 
                plt.title('Average MSE for Experiment %s' % i, pad=tpad) 
                plt.xlabel('Scaled Amplitude') 
                plt.ylabel('Average MSE') 
                plt.plot(amp_percent, hnn_MSE, 'og-', label='HNN', markersize = 8)
                plt.plot(amp_percent, dhnn_MSE, '^b-', label='D-HNN', markersize = 8)
                plt.legend(fontsize=7, loc='upper right')
                plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "Average_Scale_MSE_Fig.png"))
                plt.close()

            #plot all tiral integration times per experiment
#             tpad = 7
#             fig = plt.figure(figsize=[10,10], dpi=300) 
#             plt.title('Integration Time for Experiment %s' % i, pad=tpad) 
#             plt.xlabel('Trial Number') 
#             plt.ylabel('Time (minutes)') 
#             plt.plot(js , integration_times, 'ok-', markersize = 8)
#             plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "Integration_Times_Fig.png"))
#             plt.close()

#             #plot all tiral integration times per experiment SCALED
#             tpad = 7
#             fig = plt.figure(figsize=[10,10], dpi=300) 
#             plt.title('Integration Time for Experiment %s' % i, pad=tpad) 
#             plt.xlabel('Trial Number') 
#             plt.ylabel('Time (minutes)') 
#             plt.plot(amp_percent, integration_times, 'ok-', markersize = 8)
#             plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "Integration_Times_Scaled_Fig.png"))
#             #plt.xlim((0.94, 1.06))
#             plt.close()

#             #plot all tiral MSE times per experiment
#             tpad = 7
#             fig = plt.figure(figsize=[10,10], dpi=300) 
#             plt.title('Time to Calculate MSE for Experiemnt %s' % i, pad=tpad) 
#             plt.xlabel('Trial Number') 
#             plt.ylabel('Time (minutes)') 
#             plt.plot(js, MSE_times, 'ok-', markersize = 8)
#             plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "MSE_Times_Fig.png"))
#             plt.close()

#             #plot all tiral MSE times per experiment SCALED
#             tpad = 7
#             fig = plt.figure(figsize=[10,10], dpi=300) 
#             plt.title('Time to Calculate MSE for Experiemnt %s' % i, pad=tpad) 
#             plt.xlabel('Trial Number') 
#             plt.ylabel('Time (minutes)') 
#             plt.plot(amp_percent, MSE_times, 'ok-', markersize = 8)
#             plt.savefig(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns", MAJOR_FOLDER, "Experiment{0}".format(str(i).zfill(3)), "MSE_Times_Scaled_Fig.png"))
#             plt.close()
                
            #end of "if"
            
        experiment_time.append((time.time() - experiment_start_time)/60)
        experiment_is.append(i)
        print("--- %d minutes --- to complete experiemnt" % ((time.time() - experiment_start_time)/60))
        
        #out of "each experiment" loop
     
    #save training time data using pkl
    with open(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "Train_Time.pkl"), 'wb') as f:
        pickle.dump(training_time, f)

    with open(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "Train_Js_Time.pkl"), 'wb') as f:
        pickle.dump(training_js, f)

    #save experiment time data using pkl
    with open(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "EXP_Time.pkl"), 'wb') as f:
        pickle.dump(experiment_time, f)

    with open(os.path.join("/global/cfs/cdirs/m3792/mmeitz/dissipative_hnns/", MAJOR_FOLDER, "EXP_Is_Time.pkl"), 'wb') as f:
        pickle.dump(experiment_is, f) 
    
    end = time.time()
    print("Time to complete all experiemtns within the certrain range chosen: %d minutes" % ((end-main_start_time)/60), flush = True)
         
        
if __name__ == '__main__':
    print("START", flush = True)
    FILE = 'Trajectories400.hdf5'
    MAJOR_FOLDER = "04272023-2"                           
    main(MAJOR_FOLDER, FILE)
    print("DONE!!!", flush = True)
    
    
    