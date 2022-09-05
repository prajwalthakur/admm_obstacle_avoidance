import numpy as np
import jax.numpy as   jnp
import pytorch_kin_test



import Metric_Ik_Analytic_witthOrient
import random



def get_pose():
    #random.seed(int(time.time()/1777))
    q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165,0.0,0.0]) * np.pi / 180 # TODO
    #q_min = q_min.reshape(7,1)
    q_max = np.array([165, 101, 165, 1.0, 165, 214, 165]) * np.pi / 180  #TODO
    init_joint = [random.uniform(s, e) for s, e in zip(q_min, q_max)]
    #init_joint[8] = init_joint[7]
    start_cartesian =  jnp.asarray(Metric_Ik_Analytic_witthOrient.fk_franka(jnp.asarray(init_joint[0:7]))).reshape((1,3))
    #init_joint = jnp.array(init_joint[0:7]).reshape(1,7)
    return start_cartesian

def get_traj_guess(num_sphere , num_timestep):
    
    
    
    
    
    traj_sphere =0
    return traj_sphere


def get_franka_sphere():
    
    
        pos =  0
        return pos





