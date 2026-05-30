import numpy as np
import random
import torch
import os

from data import IEEE_39_config
from env_tran_39 import env_tran_IEEE39_cont
import TD3PD
import utils

import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cuda:1")
print(f"Training on device: {device}")
current_dir = os.path.dirname(os.path.abspath(__file__))

######### initialize env 
IEEE39_config = IEEE_39_config.load_ieee39_config()
IEEE39_config.update({
    "delta_t": 0.0008,    
    "omega_bound_th":0.1, 
    "omega_bound_sf":0.2, 
    "reset_omega_bound":0.3,  
    "reset_delta_bound":0.3, 
    "cost_action_scaler":0.001,
})

Env_tran = env_tran_IEEE39_cont(IEEE39_config)

######### initialize policy
policyname = "TD3PD"

if policyname == "TD3PD":
    policy_param = {
        "state_dim": Env_tran.bus_dim + Env_tran.bus_delta_dim,
        "action_dim": Env_tran.bus_dim,
        "max_action": 1.0,
        "discount": 0.98, # 0.98 1.0
        "tau": 0.005,
        "policy_freq":2,
        "hidden_dim": 128, 
        "actor_lr": 1e-4,
        "critic_lr": 5e-3,   
        "mu_eta":10,
    }
    agent = TD3PD.TD3PD(Env_tran, policy_param, device)

utils_param = {
    "num_episodes" : 1000, #1000
    "alpha" : 0.10 ,
    "track_length" : 1000, #1000
    "mini_batch" : 2,
    "epochs" : 10,
}


filename = 'TD3PD_' + str(utils_param["num_episodes"])

return_list, omega_outside_list, lambda_upper_list, lambda_lower_list, eta_upper_list, eta_lower_list = \
    utils.train_on_policy_agent(Env_tran, agent, utils_param)

####### save model
save_dir = os.path.join(current_dir, 'Model', policyname)
torch.save(agent.critic, os.path.join(save_dir, f'{filename}_critic.pth'))
torch.save(agent.actor, os.path.join(save_dir, f'{filename}_actor.pth'))

print("train end")
