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

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action = 3.0 * torch.tanh(self.fc3(x))     
        return action

class ActorSingleLayerNN(torch.nn.Module):
    def __init__(self, hidden_dim, scale_factor=1.0):
        super(ActorSingleLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(1, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, 1)
        self.scale_factor = scale_factor

    def forward(self, x):
        zero_input = torch.zeros_like(x)

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        action_single = self.scale_factor * 3.0 * torch.tanh(self.layer3(x))

        zero_input = torch.relu(self.layer1(zero_input))
        zero_input = torch.relu(self.layer2(zero_input))
        action_single_0 = self.scale_factor * 3.0 * torch.tanh(self.layer3(zero_input))

        return action_single-action_single_0

class ActorParallelNN(torch.nn.Module):
    def __init__(self,action_dim, hidden_dim):
        super(ActorParallelNN, self).__init__()
        self.networks = torch.nn.ModuleList([
            ActorSingleLayerNN(hidden_dim, scale_factor=0.1 if i < 29 else 1.0)
            for i in range(action_dim)
        ])
        self.action_dim = action_dim

    def forward(self, x):
        if x.shape[1] != self.action_dim:
            print(f"Warning: Input dimension mismatch! Expected x.shape[1]={self.action_dim}, but got x.shape[1]={x.shape[1]}")

        outputs = []
        for i in range(self.action_dim):
            output = self.networks[i](x[:, i:i+1])
            outputs.append(output)
        return torch.cat(outputs, dim=1)



class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.fc4 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1

class LagrangeGradient(Function):
    @staticmethod
    def forward(ctx, NN_input, omg_lower, omg_upper, lam_upp, lam_low, eta_upp,eta_low):
        ctx.save_for_backward(NN_input, omg_lower, omg_upper, lam_upp, lam_low, eta_upp,eta_low)

        return NN_input

    @staticmethod
    def backward(ctx,grad_output):
        w_state, omg_lower, omg_upper, lam_upp, lam_low, eta_upp,eta_low =ctx.saved_tensors
        
        #cost lower
        mask_lower=torch.zeros_like(w_state)  
        mask_lower[eta_low + omg_lower - w_state >= 0] = 1
        indic_lower=-lam_low*mask_lower   

        #cost upper
        mask_upper=torch.zeros_like(w_state)
        mask_upper[eta_upp+w_state-omg_upper>=0]=1
        indic_upper=lam_upp*mask_upper  

        derivative = indic_lower + indic_upper 
        grad_output = grad_output*derivative
        return grad_output,None,None,None,None,None,None

class TD3PD(object):
    def __init__(self, env, param, device):

        self.state_dim = param["state_dim"]
        self.action_dim = param["action_dim"]

        self.max_action = param["max_action"] 
        self.discount = param["discount"]
        self.tau = param["tau"]
        self.policy_freq = param["policy_freq"]
        self.device = device

        self.policy_noise = 0.2 * self.max_action
        self.noise_clip = 0.5 * self.max_action

        self.actor = ActorParallelNN(self.action_dim, param["hidden_dim"]).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = param["actor_lr"])
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = Critic(self.state_dim, self.action_dim, param["hidden_dim"]).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = param["critic_lr"])
        self.critic_target = copy.deepcopy(self.critic)

        self.Lagrangelayer=LagrangeGradient.apply
        self.mu_w = 1.0
        self.mu_lambda = 2e2 
        self.mu_eta = param["mu_eta"]

        # env parameter
        self.env = env
        self.delta_t=env.delta_t
        self.state_transfer1 = self._to_tensor_load(env.state_transfer1)
        self.select_delta = self._to_tensor_load(env.select_delta)
        self.D_reactance = self._to_tensor_load(env.D_reactance)
        self.D = self._to_tensor_load(env.D)
        self.M = self._to_tensor_load(env.M)
        self.state_transfer2 = self._to_tensor_load(env.state_transfer2)
        self.state_transfer3 = self._to_tensor_load(env.state_transfer3)
        self.select_w = self._to_tensor_load(env.select_omega)

        # help to update actor
        self.total_it = 0

    def _to_tensor_load(self, x): 
        ###### for env array 
        return torch.tensor(x, dtype=torch.float32, device=self.device)
    
    def _to_tensor_flatten(self, x):
        ###### for dataset load
        return torch.tensor(x, dtype=torch.float32, device=self.device).flatten(0,1)    

    def take_action(self, state):
        original_shape = state.shape   
        if len(original_shape) == 1:
            state_adapt = state[np.newaxis, ...]
            state_adapt = self._to_tensor_load(state_adapt)
            state_fre = state_adapt@self.select_w
            action = self.actor(state_fre)   
            return action[0].cpu().detach().numpy()
        elif len(original_shape) == 2: 
            state_adapt = self._to_tensor_load(state)
            state_fre = state_adapt@self.select_w
            action = self.actor(state_fre)   
            return action.cpu().detach().numpy()

    def update(self, transition_dict, eta_upp, eta_low, lam_upp, lam_low):
        self.total_it += 1

        env_update = self.env

        states = self._to_tensor_flatten(transition_dict['states'])
        actions = self._to_tensor_flatten(transition_dict['actions'])
        rewards = self._to_tensor_flatten(transition_dict['rewards'])
        next_states = self._to_tensor_flatten(transition_dict['next_states'])
        Pm_trains = self._to_tensor_flatten(transition_dict['Pm_trains'])

        eta_upp=self._to_tensor_load(eta_upp) 
        eta_low=self._to_tensor_load(eta_low)  
        lam_upp=self._to_tensor_load(lam_upp)
        lam_low=self._to_tensor_load(lam_low)

        omg_lower= self._to_tensor_load(env_update.omg_low_th)
        omg_upper= self._to_tensor_load(env_update.omg_upp_th)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_states_fre = next_states @ self.select_w
            next_actions = (self.actor_target(next_states_fre) + noise)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss   
            states_fre = states@self.select_w
            actions_update = self.actor(states_fre)

            ##### increase rewards
            actor_loss = -self.critic.Q1(states, actions_update).mean()

            # calculate Lagrange loss           
            next_states_update=states@self.state_transfer1 \
                    -((self.delta_t*((torch.sin(states@self.select_delta))@self.D_reactance)@self.D)*(self.M**(-1)))@self.state_transfer2 \
                        + (Pm_trains+actions_update)@self.state_transfer3

            next_w_update = next_states_update@self.select_w
            next_w_update = self.Lagrangelayer(next_w_update, omg_lower, omg_upper, lam_upp, lam_low, eta_upp,eta_low)  
            Lagrange_loss = torch.mean(next_w_update)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            Lagrange_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
