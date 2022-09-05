import numpy as np
import jax.numpy as jnp
import bernstein_coeff_order10
import admm02
import time
from jax import vmap,random
import pdb
import  obstacle_function
import plot

  
##################
t_fin = 10
num_timesteps =100
num_obs = 10
radius_obs =0.2
radius_franka_sphere = 0.1
maxitr = 100
rho_obs  = 1

weight_smoothness = 0.2
tot_time = np.linspace(0, t_fin, num_timesteps)

tot_time_copy = tot_time.reshape(num_timesteps, 1)
		

P,Pddot = bernstein_coeff_order10.bernstein_coeff_order10(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)		
nvar = np.shape(P)[1]
#pdb.set_trace()
######################### up sampling
"""num_up = 200
tot_time_up = np.linspace(0, t_fin, num_up)
tot_time_copy_up = tot_time_up.reshape(num_up, 1)

P_up = bernstein_coeff_order10.bernstein_coeff_order10(10, tot_time_copy_up[0], tot_time_copy_up[-1], tot_time_copy_up)

P_up_jax = jnp.asarray(P_up)"""
#pdb.set_trace()
ex = 1
to_save = False
to_load = True
X_guess_traj = obstacle_function.get_traj_guess_random(num_timesteps,save = to_save,load =to_load , example =ex,objects=1)
num_sphere  = X_guess_traj.shape[0]
X_guess_init = X_guess_traj[:,0,:]  
#pdb.set_trace()
#pdb.set_trace()
X_goal =  X_guess_traj[:,-1,:]
X_guess_traj_init = X_guess_traj
obs_vector = obstacle_function.get_obs_vector(X_guess_init , X_goal, num_obs,radius_obs,radius_franka_sphere, min_speration = 0.0,save = to_save,load =to_load  , example =ex)
min_distance_obs_init ,index_init,min_dist_Arg_timeStep_init = obstacle_function.get_min_distance(obs_vector,X_guess_traj)
#pdb.set_trace()
traj_opt = admm02.batch_traj_opt(P,Pddot,obs_vector,rho_obs,num_obs,t_fin,num_timesteps,num_sphere,tot_time_copy,maxitr,weight_smoothness)
traj_opt.initial_alpha_beta_d_obs(X_guess_traj)
X_next = X_guess_traj
cost_init = obstacle_function.get_cost(X_guess_init , X_goal)
print("going to solve")
cost_list =np.zeros((0,0))
for  i in range(maxitr):
    X_next,residual = traj_opt.solve(X_next)  #dim num_sphere*num_timesteps*3
    temp_x = jnp.hstack([obstacle_function.get_cost(X_next[:,i,:] , X_goal)  for i in range(num_timesteps)])
    #cost_temp =jnp.linalg.norm(temp_x,axis=1)[:,jnp.newaxis]
    if i ==0:
        cost_list = residual
    else:
        cost_list = jnp.vstack((cost_list,residual)) 
        #pdb.set_trace()
    print("iter=",i)
x_final = X_next[:,-1,:]
cost_final = obstacle_function.get_cost(x_final , X_goal)
cost_final_init = obstacle_function.get_cost(X_next[:,0,:] , X_goal)
min_distance_obs ,index,min_dist_Arg_timeStep = obstacle_function.get_min_distance(obs_vector,X_next)

print("min=" , np.min(min_distance_obs) , "arg min =", index)
plot.do_plot(cost_list.T,plotno=ex)
plot.do_3dplot(X_next,X_guess_init,X_goal,obs_vector,ex,index,min_distance_obs,min_dist_Arg_timeStep,min_distance_obs_init ,index_init,min_dist_Arg_timeStep_init,X_guess_traj_init)
pdb.set_trace()