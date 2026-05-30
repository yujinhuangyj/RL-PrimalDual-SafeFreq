import numpy as np
import copy
from scipy.integrate import solve_ivp

class env_tran_IEEE39_cont():
    def  __init__(self, config):
        self.param_gamma=1
        self.M=config["M"]
        self.E=config["E"]
        self.Pm=config["Pm"]
        self.D=config["D"]
        self.D_reactance=config["D_reactance"]
        self.Power_initial = config["Power_initial"]
        self.L = config["L"]
        self.L_pinv = config["L_pinv"]

        self.delta_t=config["delta_t"]
        self.bus_dim=config["dim_omega"]
        self.bus_delta_dim=config["dim_lambda"]
        self.delta_ref = config["delta_ref"]
        self.equilibrium_init = config["equilibrium_init"]

        self.omega_max=config["omega_max"]
        self.omega_min=config["omega_min"]
        self.Penalty_action=config["Penalty_action"]
        self.cost_action_scaler=config["cost_action_scaler"]

        self.config = config

        ######### 1 can change to batch num
        self.batch_num = 1

        self.state_transfer1=np.vstack([
            np.hstack([np.identity(self.bus_delta_dim), np.zeros((self.bus_delta_dim,self.bus_dim))]),
            np.hstack([self.delta_t*np.transpose(self.D), (np.identity(self.bus_dim)-self.delta_t*np.diag(np.squeeze(self.E/self.M)))]) 
            ]) #(85, 85)
        
        self.state_transfer2=np.hstack([np.zeros((self.bus_dim,self.bus_delta_dim)), np.identity(self.bus_dim)]) #(39, 85)

        self.state_transfer3=np.hstack([np.zeros((self.bus_dim,self.bus_delta_dim)), 
                                        self.delta_t*np.diag(np.squeeze(self.M**(-1)))]) #(39, 85)

        self.select_add_omega=np.vstack([
            np.zeros((self.bus_delta_dim,1)), 
            np.ones((self.bus_dim,1))
            ])  #(85, 1)

        self.select_omega=np.vstack([
            np.zeros((self.bus_delta_dim,self.bus_dim)), 
            np.identity(self.bus_dim)
            ])  #(85, 39)

        self.select_delta=np.vstack([
            np.identity(self.bus_delta_dim), 
            np.zeros((self.bus_dim,self.bus_delta_dim))
            ])  #(85, 46)
        
        ###### for compare model
        self.dim_omega = config["dim_omega"] 
        self.dim_lambda = config["dim_lambda"] 
        self.select_lambda = self.select_delta


        #################  change 
        self.omg_low_th = -config["omega_bound_th"] * np.ones((1, self.bus_dim)) #(1,39) 0.1
        self.omg_upp_th = config["omega_bound_th"] * np.ones((1, self.bus_dim))
        self.omg_low = -config["omega_bound_sf"] * np.ones((1, self.bus_dim)) ##0.2
        self.omg_upp = config["omega_bound_sf"] * np.ones((1, self.bus_dim))  ##0.2

        self.state = np.zeros((self.batch_num, self.bus_delta_dim + self.bus_dim)) 
        self.reset_omega_bound = config["reset_omega_bound"]
        self.reset_delta_bound = config["reset_delta_bound"]

    def set_state(self, tra_state):
        # here tra_state is (85,)
        self.state[0] = tra_state.copy()

        if self.state.shape[0] == 1:
            return self.state[0]
        return self.state

    def reset(self):        
        # initial_state_delta=np.transpose(self.D@np.linalg.pinv(self.L)@(self.Power_initial*np.random.uniform(-0.1,0.1,(self.dim_omega,1))))
        # delta_bound = 0.3  #0.3 0.15
        # power_bound = 0.1
        # power_ratio = 0.1
        # omega_bound = 0.3  #0.3 0.15
        # initial_state_delta1 = np.transpose(self.D@np.random.uniform(-delta_bound,delta_bound,(self.bus_dim,self.batch_num)))
        # initial_state_delta2 = self.D@self.L_pinv@(self.Power_initial * np.random.uniform(-power_ratio,power_ratio,(self.bus_dim,self.batch_num)))
        # initial_state_delta = self.delta_ref + initial_state_delta1 + initial_state_delta2.T

        # initial_state_power = np.random.uniform(-power_bound,power_bound,(self.bus_dim,self.batch_num))\
        #     + self.Power_initial * np.random.uniform(-power_ratio,power_ratio,(self.bus_dim,self.batch_num))
        # initial_state_delta = self.delta_ref + np.transpose(self.D@self.L_pinv@initial_state_power)

        initial_state_delta1 = np.transpose(self.D@np.random.uniform(-self.reset_delta_bound,self.reset_delta_bound,(self.bus_dim,self.batch_num)))
        initial_state_delta = self.delta_ref + initial_state_delta1
        initial_state_omega = np.random.uniform(-self.reset_omega_bound,self.reset_omega_bound,(self.batch_num,self.bus_dim))
        initial_state = np.hstack((initial_state_delta,initial_state_omega))
        self.state = initial_state

        if self.state.shape[0] == 1:
            return self.state[0].copy()
        return self.state.copy()

    def dynamics(self, t, state_vec):
        """
        ODE system: d(state)/dt = f(state, action, Pm)
        state_vec: (n,)
        """
        # Reshape to 2D for matrix operations
        state = state_vec.reshape(1, -1)
        delta = state[:, :self.bus_delta_dim]  # (1, 46)
        omega = state[:, self.bus_delta_dim:self.bus_delta_dim + self.bus_dim]  # (1, 39)
        
        d_delta_dt = omega @ np.transpose(self.D) 

        # power_term = np.sin(self.delta_ref + delta) @ self.D_reactance @ self.D  # (1, 39)
        power_term = np.sin(delta) @ self.D_reactance @ self.D  # (1, 39)
        d_omega_dt = omega * (-(self.E / self.M)) \
                    + self.current_Pm * (1.0 / self.M) \
                    + self.current_action * (1.0 / self.M) \
                    - power_term * (1.0 / self.M)  # (1, 39)
        
        d_state_dt = np.hstack([d_delta_dt, d_omega_dt])  # (1, 85)
        
        return d_state_dt.flatten()  # (85,)

    def step(self, action, Pm):

        initial_action = action
        action = np.atleast_2d(action)
        Pm = np.atleast_2d(Pm) 
        self.current_action = action
        self.current_Pm = Pm

        state0 = self.state[0].copy()
        
        # Solve ODE from t=0 to t=delta_t, y0 should be (n,)
        sol = solve_ivp(
            fun=self.dynamics,
            t_span=[0, self.delta_t],
            y0=state0, #(85，)
            method='RK45',  # MATLAB ode45 DOP853 RK45
            dense_output=False,
            # dense_output=True,
            # max_step=0.00002,
            rtol=1e-6,  
            atol=1e-9   
        )
        
        next_state = sol.y[:, -1].reshape(1, -1)
        self.state = next_state

        cost_state = self.param_gamma*pow(next_state,2)@self.select_add_omega  #(1, 1)        
        cost = cost_state.item() + self.cost_action_scaler * np.sum(pow(action,2))
        
        # reward= -cost
        if len(initial_action.shape) == 1:
            return self.state[0].copy(), -cost
        return self.state.copy(), -cost
    
    def action_projection(self, action, Pm):
        initial_action = action

        action = np.atleast_2d(action)
        Pm = np.atleast_2d(Pm) 

        omega_old = self.state@self.select_omega
        delta_old = self.state@self.select_delta
        omega_lower_term = (self.delta_t**(-1))*self.M*(self.omg_low-omega_old)
        omega_upper_term = (self.delta_t**(-1))*self.M*(self.omg_upp-omega_old)
        delta_term = (np.sin(delta_old)@self.D_reactance)@self.D
        action_lower = omega_lower_term + delta_term + self.E*omega_old - Pm
        action_upper = omega_upper_term + delta_term + self.E*omega_old - Pm
        action_pro = np.clip(action, action_lower, action_upper)

        if len(initial_action.shape) == 1:
            return action_pro[0].copy()
        return action_pro.copy()
    

    
