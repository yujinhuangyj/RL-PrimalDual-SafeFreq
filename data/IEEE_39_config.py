import os
import numpy as np
from mat4py import loadmat

np.random.seed(42)

def load_ieee39_config(**overrides):
    # Simulation data load from IEEE 39-bus transmission system
    current_dir = os.path.dirname(os.path.abspath(__file__))

    dim_omega=39 #dimension of action space  
    dim_lambda=46 #dimension of lambda

    data_path = os.path.join(current_dir, "IEEE-39-power-initial-balanced.mat")
    data = loadmat(data_path)
    Power_initial=data["power_initial"]
    Power_initial=np.asarray(Power_initial) #(39, 1)
    ## data change 1
    Power_initial=0.3*Power_initial

    data_path = os.path.join(current_dir, 'IEEE-39-rotational-inertial-generator.mat')
    data = loadmat(data_path)
    Rotational_inertial_generator=data['rotational_inertial_generator']
    Rotational_inertial_generator=np.asarray(Rotational_inertial_generator)  #(10, 2) name+inertia last10(30-39)

    Rotational_inertial_generator_blc = Rotational_inertial_generator.copy()
    ## data change 2
    Rotational_inertial_generator_blc[-1,1]=np.mean(Rotational_inertial_generator_blc[:-1,1])

    data_path = os.path.join(current_dir, 'IEEE-39-rotational-inertial-load.mat')
    data = loadmat(data_path)
    Rotational_inertial_load=data['rotational_inertial_load']
    Rotational_inertial_load=np.asarray(Rotational_inertial_load)    #(29, 2) first29(1-29)

    M=((np.vstack((Rotational_inertial_load,Rotational_inertial_generator_blc)))[:,1]).reshape(1,dim_omega)
    # 4.86-8.4 
    ####### para test LM Low Inertia
    # M = 0.7 * M
    # M = M * np.random.uniform(0.7, 1.3, size=M.shape)

    data_path = os.path.join(current_dir, "IEEE-39-adjacency-matrix.mat")
    data = loadmat(data_path)
    D=data["D"]
    D=np.asarray(D) #(46, 39) all 0,1,-1

    data_path = os.path.join(current_dir, "IEEE-39-susceptance-matrix.mat")
    data = loadmat(data_path)
    # D_reactance_vector=data["D_reactance"]
    # D_reactance_vector=np.asarray(D_reactance_vector) #(46,1)
    # D_reactance=np.diag(D_reactance_vector[:, 0])#Yb (46, 46) diagonal matrix
    D_reactance=data['D_reactance']
    D_reactance=np.asarray(D_reactance)
    ## data change 3  0.1
    D_reactance=0.3*D_reactance

    ####### para test WD Weak Grid
    # D_reactance = 0.85 * D_reactance
    # D_reactance = D_reactance * np.random.uniform(0.9, 1.1, size=D_reactance.shape)

    # def stable_float32_pinv(matrix, rcond=1e-6):
    #     matrix = np.array(matrix, dtype=np.float32) 
    #     U, s, Vh = np.linalg.svd(matrix, full_matrices=False) 
    #     cutoff = rcond * np.max(s)   
    #     s_inv = np.zeros_like(s)
    #     mask = s > cutoff
    #     s_inv[mask] = 1.0 / s[mask]   
    #     pinv = (Vh.T * s_inv) @ U.T   
    #     return pinv.astype(np.float32)
    
    def stable_pinv(matrix, rcond=1e-6):
        matrix = np.array(matrix)  
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False) 
        cutoff = rcond * np.max(s)   
        s_inv = np.zeros_like(s)
        mask = s > cutoff
        s_inv[mask] = 1.0 / s[mask]   
        pinv = (Vh.T * s_inv) @ U.T   
        return pinv

    E=np.ones((1,dim_omega)) #(1, 39) all 1 damping coefficents D 

    ####### para test HE High Damping
    # E = 1.5 * E
    # E = E * np.random.uniform(0.8, 1.2, size=E.shape)

    ## data change 4  for third
    # E[0, -10:] += 5
    L=np.transpose(D)@(D_reactance)@D  #(39, 39)
    L_pinv = stable_pinv(L)

    syn_frequence=np.sum(Power_initial,axis=0)/np.sum(E,axis=1)  #0.0
    f_initial=D@L_pinv@(Power_initial-syn_frequence*np.transpose(E))
    equilibrium_init=np.hstack((np.transpose(f_initial),syn_frequence*np.ones((1,dim_omega))))
    delta_ref = equilibrium_init[:, :dim_lambda]


    config = {
        "M": M, #(1,39)
        "E": E, #(1, 39) all 1 damping coefficents D 
        "Pm": np.transpose(Power_initial), #(1, 39) <8.3
        "D": D, #(46, 39) all 0,1,-1
        "D_reactance": D_reactance, #(46, 46)
        "Power_initial": Power_initial,    #(39, 1)
        "L": L, #(39, 39)
        "L_pinv": L_pinv, #(39, 39)

        "delta_t": 0.0008,  #0.0008

        "dim_omega": dim_omega, #39
        "dim_lambda": dim_lambda, #46 

        "Penalty_action": 40*200,   
        "omega_max": 0.2,
        "omega_min": -0.2,

        ### 
        "equilibrium_init": equilibrium_init, #(1, 85)
        "delta_ref": delta_ref
    }
    
    config.update(overrides)
    return config