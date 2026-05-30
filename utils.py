import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from mat4py import loadmat
import time
from tqdm import tqdm

import copy
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR

def train_on_policy_agent(env, agent, utils_param):
    num_episodes = utils_param["num_episodes"] 
    alpha = utils_param["alpha"]
    track_length = utils_param["track_length"]
    mini_batch = utils_param["mini_batch"]
    epochs = utils_param["epochs"]

    episodes_per_epoch = num_episodes // epochs
    return_list = []
    omega_outside_list = []
    lambda_upper_list = []
    lambda_lower_list = []    
    eta_upper_list = []
    eta_lower_list = []  
    
    #####initial lambda and eta
    omega_dim = env.bus_dim  
    lambda_lower=np.zeros((omega_dim,),dtype=np.float32)
    lambda_upper=np.zeros((omega_dim,),dtype=np.float32)
    eta_lower = np.ones((omega_dim,), dtype=np.float32) * 0.005
    eta_upper = np.ones((omega_dim,), dtype=np.float32) * 0.005

    for epoch in range(epochs):
        desc = f'Epoch {epoch+1}/{epochs}'  
        with tqdm(total=episodes_per_epoch, desc=desc) as pbar:
            for episode in range(episodes_per_epoch):   
                batch_trajectory = {
                    "states": [],      # [mini_batch, track_length, state_dim]
                    "actions": [],     # [mini_batch, track_length, gen_dim, 2 ]
                    "next_states": [], # [mini_batch, track_length, state_dim]
                    "rewards": [],     # [mini_batch, track_length]
                    "Pm_trains": []     # [mini_batch, track_length]
                }              
                total_episode_return = 0

                for traj_idx in range(mini_batch):         
                    trajectory = {
                        "states": [],  "actions":[],  "next_states": [], "rewards": [], "Pm_trains": []
                    }
                    trajectory_return = 0       

                    state = env.reset() #(85,)
                    ###set Pm_train
                    Pm_train = env.Pm[0].copy() #(39,)

                    for step in range(track_length):
                        ########actions from actor are float32
                        action = agent.take_action(state)
                        action = action.astype(np.float64)  

                        ##### safety layer
                        action_pro = env.action_projection(action, Pm_train)

                        next_state_env, reward_env = env.step(action_pro, Pm_train)

                        trajectory['states'].append(state)
                        trajectory['actions'].append(action_pro)                      
                        trajectory['next_states'].append(next_state_env)
                        trajectory['rewards'].append(np.array([reward_env]))
                        trajectory['Pm_trains'].append(Pm_train)
                        state = next_state_env
                        trajectory_return += reward_env.item()

                    for key in batch_trajectory:
                        batch_trajectory[key].append(trajectory[key])
                    
                    total_episode_return += trajectory_return

                return_list.append(total_episode_return)

                ###### record omega_outside_ratio
                next_states_batch = np.array(batch_trajectory['next_states'])
                next_omega_batch = next_states_batch @ env.select_omega
                omega_outside_pertime = np.any((next_omega_batch < -0.201)|(next_omega_batch > 0.201), axis=2)
                omega_outside_ratio = omega_outside_pertime.mean()
                omega_outside_list.append(omega_outside_ratio)

                ###### updat eta and lambda
                states_batch = np.array(batch_trajectory['states'], dtype=np.float32)
                omega_batch = states_batch @ env.select_omega.astype(np.float32)

                omg_low_th = env.omg_low_th.astype(np.float32)
                omg_upp_th = env.omg_upp_th.astype(np.float32)

                grad_lambda_lower = np.maximum((eta_lower + omg_low_th - omega_batch), 0) - alpha * eta_lower
                grad_lambda_lower = np.mean(grad_lambda_lower, axis=(0, 1))
                lambda_lower = np.maximum(lambda_lower + agent.mu_lambda * grad_lambda_lower, 0)

                grad_lambda_upper = np.maximum((eta_upper + omega_batch - omg_upp_th), 0) - alpha * eta_upper
                grad_lambda_upper = np.mean(grad_lambda_upper, axis=(0, 1))
                lambda_upper = np.maximum(lambda_upper + agent.mu_lambda * grad_lambda_upper, 0)

                grad_eta_lower = ((eta_lower + omg_low_th - omega_batch) >= 0).astype(np.float32) - alpha
                grad_eta_lower = np.mean(grad_eta_lower, axis=(0, 1))
                eta_lower = np.clip(eta_lower - agent.mu_eta * lambda_lower * grad_eta_lower, -0.005, 0.005)

                grad_eta_upper = ((eta_upper + omega_batch - omg_upp_th) >= 0).astype(np.float32) - alpha
                grad_eta_upper = np.mean(grad_eta_upper, axis=(0, 1)) 
                eta_upper = np.clip(eta_upper - agent.mu_eta * lambda_upper * grad_eta_upper, -0.005, 0.005)              

                agent.update(batch_trajectory, eta_upper,eta_lower, lambda_upper, lambda_lower)

                lambda_upper_list.append(lambda_upper)
                lambda_lower_list.append(lambda_upper)                
                eta_upper_list.append(eta_upper)
                eta_lower_list.append(eta_lower)

                if (episode+1) % epochs == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/epochs * epoch + episode+1), 'return': '%.3f' % np.mean(return_list[-epochs:])})
                pbar.update(1)

    # eta_upper_list, eta_lower_list
    return return_list, omega_outside_list, lambda_upper_list, lambda_lower_list, eta_upper_list, eta_lower_list

