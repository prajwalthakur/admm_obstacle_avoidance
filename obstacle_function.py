import numpy as np
import jax.numpy as jnp
import random
import pdb

def get_pose():
    q_min = np.array([-0.3,-0.3,-0.3])  # TODO
    q_max = np.array([0.3,0.3,0.3])   #TODO
    start_cartesian =  np.array([random.uniform(s, e) for s, e in zip(q_min, q_max)])
    return start_cartesian






def get_obs_pose(X_franka_sphere_init ,X_goal, num_obs,radius_obs,radius_franka_sphere, min_speration):
    #dim X_franka_sphere = num_sphere*3
    obs_pose_vector =np.zeros((0,0))
    itr_obs =0
    while obs_pose_vector.shape[0] < num_obs:
        
        obs_center_sampled = get_pose()
        #pdb.set_trace()
        obs_center_sampled = jnp.asarray(obs_center_sampled)

        if ( (jnp.linalg.norm((obs_center_sampled-X_franka_sphere_init)) > radius_obs + radius_franka_sphere + min_speration ).all() and (jnp.linalg.norm((obs_center_sampled-X_goal)) > radius_obs + radius_franka_sphere + min_speration ).all()):
            if itr_obs ==0:
                obs_pose_vector = obs_center_sampled
            else :
                obs_pose_vector =jnp.vstack((obs_pose_vector,obs_center_sampled))
            itr_obs+=1
    return obs_pose_vector #dim num_obs*3   obs_vector = np.array([(x,y,z)_obs )



def get_obs_vector(X_franka_sphere ,X_goal, num_obs,radius_obs,radius_franka_sphere, min_speration = 0.1,save = True,load = False , example =np.NAN):
    
    """if load == False:
        #for all sphere obs_vector is same 
        obs_pose_vector = get_obs_pose(X_franka_sphere ,X_goal, num_obs,radius_obs,radius_franka_sphere, min_speration)
        temp_a = jnp.array((radius_obs + radius_franka_sphere)).reshape(1,1)
        temp = jnp.tile(temp_a ,(num_obs,1))
        
        obs_vector = jnp.hstack((obs_pose_vector,temp))
        #pdb.set_trace()"""
    if True:
        obs_vector = np.load(f"examples/obs_vector_{200}.npy")
        #return obs_vector
    if save ==True:
        np.save(f"examples/obs_vector_{example}",obs_vector)
    return obs_vector  #obs_vector = np.array([(x,y,z)_obs , (a_obs,b_obs,c_obs)_init ] )
    
    

def get_traj_guess_random(num_timestep,save = True,load = False , example =np.NAN,objects=1):
    if load == False:
        sphere_traj_list = np.zeros((0,0))
        for sphere_num in range(objects):
            sphere_traj_list = np.load(f"examples/X_guess_traj_{200}.npy")
            #pdb.set_trace()
            start_cartesian = sphere_traj_list[3,0,:] #get_pose()
            goal_cartesian = sphere_traj_list[3,-1,:] #get_pose()
            sphere_traj_temp =  np.linspace(start_cartesian.squeeze(),goal_cartesian.squeeze(),num_timestep)
            if sphere_num == 0:
                sphere_traj_list = sphere_traj_temp[jnp.newaxis,]
                
            else:
                
                sphere_traj_list = jnp.vstack((sphere_traj_list,sphere_traj_temp[jnp.newaxis,]))
               
    else:
        sphere_traj_list = np.load(f"examples/X_guess_traj_{example}.npy")
        return sphere_traj_list
    
    if save ==True:
        np.save(f"examples/X_guess_traj_{example}" ,sphere_traj_list )
    
    return sphere_traj_list #dim = num_sphere*num_timestep*3


def get_cost(X_final , X_goal):
    
        cost = np.linalg.norm((X_final-X_goal),axis =1)
        #pdb.set_trace()
        return cost[:,jnp.newaxis]


def get_min_distance(obs_vector,X_next ):
    #dim x_next num_sphere*num_timesetep*3
    obs_vector_pose = obs_vector[:,0:3]
    X_obs = obs_vector_pose[:,jnp.newaxis,] #new_dim = num_obs*1*3
    x_next = X_next[:,jnp.newaxis,]
    dist = jnp.linalg.norm((X_obs- x_next) , axis = -1)
    
    min_dist = jnp.min(dist,axis=2)
    min_dist_Arg_timeStep  = np.vstack([np.unravel_index(np.argmin(dist[i,:,:], axis=None), dist[i,:,:].shape) for i in range(dist.shape[0])  ])
    #pdb.set_trace()
    #output min_dist = num_sphere*num_obs*1
    
    
    index = np.unravel_index(np.argmin(min_dist, axis=None), min_dist.shape)
    #arg_sort = np.unravel_index(np.argsort(min_dist))
    return min_dist ,index,min_dist_Arg_timeStep

import pdb
if __name__ == "__main__":
    coords_array_reshape= get_traj_guess_random(200)
    pdb.set_trace()