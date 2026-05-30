import numpy as np
import random
import torch
import os

import time
from mat4py import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

from data import IEEE_39_config
import utils
import TD3PD
from env_tran_39 import env_tran_IEEE39_cont

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:1")
current_dir = os.path.dirname(os.path.abspath(__file__))

######### initialize Env 
IEEE39_config = IEEE_39_config.load_ieee39_config()
IEEE39_config.update({
    "delta_t": 0.01,
    "omega_bound_th":0.1, 
    "omega_bound_sf":0.2,  
    "reset_omega_bound":0.3,  
    "reset_delta_bound":0.3,  
    "cost_action_scaler":0.001,
})

Env = env_tran_IEEE39_cont(IEEE39_config)

######### load policy TD3PD
policyname = "TD3PD"
if policyname == "TD3PD":
    policy_param = {
        "state_dim": Env.bus_dim + Env.bus_delta_dim,
        "action_dim": Env.bus_dim,
        "max_action": 1.0,
        "discount": 0.98,
        "tau": 0.005,
        "policy_freq":2,
        "hidden_dim": 128, 
        "actor_lr": 1e-4,
        "critic_lr": 5e-3,   
        "mu_eta":10,   
    }
    agent = TD3PD.TD3PD(Env, policy_param, device) 

filename_test = "TD3PD_20"
filename_actor = os.path.join(current_dir, 'Model','TD3PD', filename_test + '_actor.pth')
filename_critic = os.path.join(current_dir, 'Model','TD3PD', filename_test + '_critic.pth')
agent.actor = torch.load(filename_actor, map_location=device)
agent.critic = torch.load(filename_critic, map_location=device)

##########################begin to compare
plt.rcParams.update({'font.size': 24}) 

dim_delta = Env.bus_delta_dim
dim_omega = Env.bus_dim
Pm = Env.Pm.copy()
# state0 = np.zeros(Env.equilibrium_init.shape)
# state0_compare = Env.equilibrium_init
state0 = Env.equilibrium_init

############ test action

omega_range = np.arange(-0.5,0.5,0.01)
action_td3pd_all = np.zeros(len(omega_range))

fig = plt.figure(figsize = (12,9), dpi = 200)
idx_plot_i=0
for idx_plot in [29,32,35,38] :
    gen_idx = idx_plot
    state = state0.copy()
    for j in range(len(omega_range)):
        state[0,dim_delta+gen_idx] = omega_range[j]

        u_td3pd = agent.take_action(state) 
        action_td3pd_all[j] = u_td3pd[0][gen_idx]

    plt.subplot(2,2,idx_plot_i+1)
    plt.plot(omega_range, action_td3pd_all, label = 'TD3PD',linewidth=2)
    plt.xlim(-0.5,0.5)
    plt.ylim(-3.4,4.5)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1.5))
    plt.title('Gen'+str(gen_idx+1), fontsize=24)
    plt.xlabel('$\omega (Hz)$')
    plt.ylabel('u (p.u.)') 
    plt.legend(fontsize=20, framealpha=0.3, bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0., handlelength=1.0, handletextpad=0.4)   
    plt.tick_params(axis='both', labelsize=20)
    plt.grid(True, alpha=0.5)
    idx_plot_i+=1
fig.tight_layout() 

save_dir = os.path.join(current_dir, 'Picture')

with PdfPages(os.path.join(save_dir, f'{filename_test}_action.pdf')) as pdf:
    pdf.savefig()
    plt.close()

############ test trajectory

state_test = state0.copy()
Env.set_state(state_test)

s_record_all_omega = state_test @ Env.select_omega
s_record_all_omega_true = s_record_all_omega + 60*np.ones((1,dim_omega))
s_record_all_lambda = state_test @ Env.select_lambda
Trajectory_omega=[]
Trajectory_omega_true=[]
Trajectory_lambda=[]
Trajectory_omega.append(s_record_all_omega)
Trajectory_omega_true.append(s_record_all_omega_true)
Trajectory_lambda.append(s_record_all_lambda)

Trajectory_action=[]
Trajectory_return=[] 
return_td3pd=0

Test_time=50    
SimulationLength=int(Test_time/Env.delta_t)

change_node_Pm = 38
change_start_time = 1.0
change_end_time = 21.0


for i in range(SimulationLength):
    current_time = i * Env.delta_t

    Pm_test = Pm.copy()

    if current_time >= change_start_time and current_time < change_end_time:
        Pm_test[0,change_node_Pm-1] = 0 

    u_td3pd = agent.take_action(state_test)
    next_s_td3pd, r_td3pd= Env.step(u_td3pd,Pm_test)
    return_td3pd += r_td3pd
    state_test = next_s_td3pd
    s_record_all_omega = state_test@Env.select_omega
    s_record_all_omega_true = s_record_all_omega + 60*np.ones((1,dim_omega))
    s_record_all_lambda = state_test@Env.select_lambda
    Trajectory_omega.append(s_record_all_omega)
    Trajectory_omega_true.append(s_record_all_omega_true)
    Trajectory_lambda.append(s_record_all_lambda)
    Trajectory_action.append(u_td3pd)
    Trajectory_return.append(np.squeeze(r_td3pd))

print(f"return_td3pd: {return_td3pd/SimulationLength:.5f}")

Trajectory_omega=np.squeeze(np.asarray(Trajectory_omega))
Trajectory_omega_true=np.squeeze(np.asarray(Trajectory_omega_true))
Trajectory_lambda=np.squeeze(np.asarray(Trajectory_lambda))
Trajectory_action=np.squeeze(np.asarray(Trajectory_action))

plt.figure(figsize=(8,16), dpi=200)

TimeRecord=np.arange(1,SimulationLength+1)
TimeRecord=Env.delta_t*TimeRecord

plt.subplot(3,1,1)
# plt.subplot(1,3,1)
plt.plot(TimeRecord,Trajectory_action)
plt.xlim(0, 50)
plt.ylim(-1.6, 1.8)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.6)) # 0.1 for noise
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlabel('Time (s)')
plt.ylabel('u (p.u.)')
plt.tick_params(axis='both', labelsize=20)
plt.grid(True, alpha=0.5)


plt.subplot(3,1,2)
TimeRecord=np.arange(1,SimulationLength+2)
TimeRecord=Env.delta_t*TimeRecord

upp_bound_omg = 60.2
low_bound_omg = 59.8
show_scale = 0.05 

total_timesteps = Trajectory_omega_true.shape[0]  
outside_range_per_timestep = np.any((Trajectory_omega_true < low_bound_omg)|(Trajectory_omega_true > upp_bound_omg), axis=1)
outside_timesteps = np.sum(outside_range_per_timestep)
proportion_outside = float(outside_timesteps) / float(total_timesteps)

print(filename_test)
print(f"Transboundary ratio of TD3PD : {proportion_outside:.4f}")
print(f"Max frequency deviation of TD3PD : {Trajectory_omega_true.max()-60:.4f}")
print(f"Min frequency deviation of TD3PD : {Trajectory_omega_true.min()-60:.4f}")

plt.plot(TimeRecord,Trajectory_omega_true)

plt.axhline(y=low_bound_omg, color='r', linestyle='--', linewidth=1.5, label='Lower bound')
plt.axhline(y=upp_bound_omg, color='r', linestyle='--', linewidth=1.5, label='Upper bound')

plt.xlim(0, 50)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.ylim(low_bound_omg-show_scale, upp_bound_omg + show_scale)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlabel('Time (s)')
plt.ylabel('$\omega$ (Hz)')
plt.tick_params(axis='both', labelsize=20)
plt.grid(True, alpha=0.5)


plt.subplot(3,1,3)
TimeRecord=np.arange(1,SimulationLength+2)
TimeRecord=Env.delta_t*TimeRecord

plt.plot(TimeRecord,Trajectory_lambda)
plt.grid()
plt.xlim(0, 50)
plt.ylim(-0.24, 0.21)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.07)) #0.05
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.xlabel('Time (s)')
plt.ylabel('$\delta$ (rad)')
plt.tick_params(axis='both', labelsize=20)
plt.grid(True, alpha=0.5)

plt.tight_layout() 

save_dir = os.path.join(current_dir, 'Picture')
with PdfPages(os.path.join(save_dir, f'{filename_test}_test.pdf')) as pdf:
    pdf.savefig()
    plt.close()

print("test end")