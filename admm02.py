from ctypes import c_bool
import jax.numpy as jnp
from functools import partial
from jax import jit,random
import time
import numpy as np
import pdb
class batch_traj_opt():
    def __init__(self,P,Pddot,obs_vector,rho_obs,num_obs,t_fin,num_timestep,num_sphere,time_steps,maxitr,weight_smoothness):
        #dim P = num_timestep*coff
        #obs_vector = np.array([(x,y,z)_obs , (a_obs,b_obs,c_obs)_init ])
        self.rho_obs = rho_obs
        self.num_obs = num_obs
        self.maxitr = maxitr
        self.weight_smoothness = weight_smoothness
        
        self.t_fin = t_fin
        self.num_timestep = num_timestep
        self.num_sphere = num_sphere
        self.t = self.t_fin/self.num_timestep

        
        self.a_obs = 0.2
        self.b_obs = 0.2
        self.c_obs = 0.2   #dim num_obs*1
        self.X_obs = obs_vector[:,0:3]  #dim num_obs*3
        #self.d_init = 
        
        self.time_steps_vec  = time_steps
        self.P = P
        self.Pddot_jax = jnp.asarray(Pddot)
        self.P_jax = jnp.asarray(self.P) #dim num_timestep*coff
        self.nvar = jnp.shape(self.P_jax)[1] #number of Polynommial cofficients

        self.A_obs = jnp.tile(self.P_jax, (self.num_obs, 1))
        
        
        self.alpha_obs= jnp.zeros((self.num_sphere,self.num_obs*self.num_timestep))
        self.beta_obs= jnp.zeros((self.num_sphere,self.num_obs*self.num_timestep))
        self.d_obs = jnp.zeros((self.num_sphere,self.num_obs*self.num_timestep))
        self.C_xyz  = jnp.zeros((self.num_sphere,self.nvar,3)) #dim num_sphere*coff*1
        self.lambda_xyz =  jnp.zeros((self.num_sphere,self.nvar,3)) # dim num_sphere*num_coff*3
    
        
        self.cost_smoothness = self.weight_smoothness*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
        
        self.mu_xyz = jnp.zeros((self.num_sphere,self.nvar,3))
        
        
        self.A_eq = jnp.vstack(( self.P_jax[0], self.P_jax[-1] ))
        
        
    
    def initial_alpha_beta_d_obs(self,X_guess):
        #dim X_guess = num_spheres*(num_timesetep*3)
        x_guess, y_guess,z_guess =  X_guess[:,:,0] ,X_guess[:,:,1] , X_guess[:,:,2]    #dim num_sphere*num_timesetep
        x_obs, y_obs,z_obs =  self.X_obs[:,0] ,self.X_obs[:,1] ,self.X_obs[:,2]         #dim num_obs
        
        x_obs= jnp.tile(x_obs,self.num_timestep)
        x_obs = x_obs.reshape((self.num_obs ,self.num_timestep)) #dim num_obs*num_timestep
        
        y_obs= jnp.tile(y_obs,self.num_timestep)
        y_obs = y_obs.reshape((self.num_obs ,self.num_timestep))        
        
        z_obs= jnp.tile(z_obs,self.num_timestep)
        z_obs = z_obs.reshape((self.num_obs ,self.num_timestep))     
        
        wc_alpha_temp = x_guess - x_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        ws_alpha_temp = y_guess - y_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        wc_beta_temp =  z_guess - z_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        
        wc_alpha = wc_alpha_temp.transpose(1,0,2)  #dim num_sphere*num_obs*num_timestep
        ws_alpha = ws_alpha_temp.transpose(1,0,2) #dim num_sphere*num_obs*num_timestep
        wc_beta = wc_beta_temp.transpose(1,0,2)  #dim num_sphere*num_obs*num_timestep
        
        wc_alpha = wc_alpha.reshape(self.num_sphere , self.num_timestep*self.num_obs)  #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        ws_alpha = ws_alpha.reshape(self.num_sphere , self.num_timestep*self.num_obs)  #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        wc_beta  = wc_beta.reshape(self.num_sphere , self.num_timestep*self.num_obs)   #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        
        alpha_obs = np.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
        ws_beta = (wc_alpha)/(jnp.cos(alpha_obs))
        beta_obs = jnp.arctan2( ws_beta*(1/self.a_obs), wc_beta*(1/self.c_obs) )
        rho_aug = self.rho_obs
        c1_d = 1.0*rho_aug*(self.a_obs**2*jnp.sin(beta_obs)**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2*jnp.sin(beta_obs)**2 + self.c_obs**2*jnp.cos(beta_obs)**2)
        c2_d = 1.0*rho_aug*(self.a_obs*wc_alpha*jnp.sin(beta_obs)*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + self.c_obs*wc_beta*jnp.cos(beta_obs))
        d_temp = c2_d/c1_d
        
        d_obs_temp = np.maximum(np.ones((self.num_sphere,  self.num_timestep*self.num_obs   )), d_temp   )
        self.d_obs = d_obs_temp
        self.alpha_obs =alpha_obs
        self.beta_obs = beta_obs
        #pdb.set_trace()
        
    def compute_alpha_beta_d_obs(self,X_guess ):
        #dim X_guess = num_spheres*(num_timesetep*3)
        x_guess, y_guess,z_guess =  X_guess[:,:,0] ,X_guess[:,:,1] , X_guess[:,:,2]    #dim num_sphere*num_timesetep
        x_obs, y_obs,z_obs =  self.X_obs[:,0] ,self.X_obs[:,1] ,self.X_obs[:,2]         #dim num_obs
        x_obs= jnp.tile(x_obs,self.num_timestep)
        x_obs = x_obs.reshape((self.num_obs ,self.num_timestep)) #dim num_obs*num_timestep
        
        y_obs= jnp.tile(y_obs,self.num_timestep)
        y_obs = y_obs.reshape((self.num_obs ,self.num_timestep))        
        
        z_obs= jnp.tile(z_obs,self.num_timestep)
        z_obs = z_obs.reshape((self.num_obs ,self.num_timestep))    
        
        
        wc_alpha_temp = x_guess - x_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        ws_alpha_temp = y_guess - y_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        wc_beta_temp =  z_guess - z_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        
        wc_alpha = wc_alpha_temp.transpose(1,0,2)  #dim num_sphere*num_obs*num_timestep
        ws_alpha = ws_alpha_temp.transpose(1,0,2) #dim num_sphere*num_obs*num_timestep
        wc_beta = wc_beta_temp.transpose(1,0,2)  #dim num_sphere*num_obs*num_timestep
        
        wc_alpha = wc_alpha.reshape(self.num_sphere , self.num_timestep*self.num_obs)  #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        ws_alpha = ws_alpha.reshape(self.num_sphere , self.num_timestep*self.num_obs)  #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        wc_beta  = wc_beta.reshape(self.num_sphere , self.num_timestep*self.num_obs)   #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        
        alpha_obs = np.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
        ws_beta = (wc_alpha)/(jnp.cos(alpha_obs))
        beta_obs = jnp.arctan2( ws_beta*(1/self.a_obs), wc_beta*(1/self.c_obs) )
        rho_aug = self.rho_obs
        c1_d = 1.0*rho_aug*(self.a_obs**2*jnp.sin(beta_obs)**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2*jnp.sin(beta_obs)**2 + self.c_obs**2*jnp.cos(beta_obs)**2)
        c2_d = 1.0*rho_aug*(self.a_obs*wc_alpha*jnp.sin(beta_obs)*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)*jnp.sin(beta_obs) + self.c_obs*wc_beta*jnp.cos(beta_obs))
        d_temp = c2_d/c1_d
        
        d_obs_temp = np.maximum(np.ones((self.num_sphere,  self.num_timestep*self.num_obs   )), d_temp   )
        self.d_obs = d_obs_temp
        self.alpha_obs =alpha_obs
        self.beta_obs = beta_obs
    
    def compute_sphere_X(self,X_guess):
        #dim X_guess = num_spheres*(num_timesetep*3)
        lambda_xyz = self.lambda_xyz # dim num_sphere*num_coff*3
        lambda_x = lambda_xyz[:,:,0] #dim num_sphere*num_coff
        lambda_y = lambda_xyz[:,:,1] #dim num_sphere*num_coff
        lambda_z = lambda_xyz[:,:,2] #dim num_sphere*num_coff
        x_guess, y_guess,z_guess =  X_guess[:,:,0] ,X_guess[:,:,1] , X_guess[:,:,2]    #dim num_sphere*num_timesetep
        
        x_obs, y_obs,z_obs =  self.X_obs[:,0] ,self.X_obs[:,1] ,self.X_obs[:,2]         #dim num_obs
        
        x_obs= jnp.tile(x_obs,self.num_timestep)
        x_obs = x_obs.reshape((self.num_obs ,self.num_timestep)) #dim num_obs*num_timestep
        
        y_obs= jnp.tile(y_obs,self.num_timestep)
        y_obs = y_obs.reshape((self.num_obs ,self.num_timestep))        
        
        z_obs= jnp.tile(z_obs,self.num_timestep)
        z_obs = z_obs.reshape((self.num_obs ,self.num_timestep))             
        
        
        
        
        temp_x_obs = self.d_obs*jnp.cos(self.alpha_obs)*jnp.sin(self.beta_obs)*self.a_obs
        
        #pdb.set_trace()
        
        b_obs_x = x_obs.reshape(self.num_timestep*self.num_obs) +temp_x_obs  #dim  num_sphere*(self.num_obs*num_timestep) 2d array 
        
        temp_y_obs = self.d_obs*jnp.sin(self.alpha_obs)*jnp.sin(self.beta_obs)*self.b_obs
        
        b_obs_y = y_obs.reshape(self.num_timestep*self.num_obs) +temp_y_obs  #dim  num_sphere*(self.num_obs*num_timestep) 2d array 

        temp_z_obs = self.d_obs*jnp.cos(self.beta_obs)*self.c_obs

        b_obs_z = z_obs.reshape(self.num_timestep*self.num_obs) +temp_z_obs  #dim  num_sphere*(self.num_obs*num_timestep) 2d array      
        
        
        cost_x = self.cost_smoothness + self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs) + jnp.dot(self.P_jax.T, self.P_jax)
        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq.T )), jnp.hstack(( self.A_eq, jnp.zeros(( jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0] )) )) ))
        
                
        lincost_x = -lambda_x-self.rho_obs*jnp.matmul(self.A_obs.T, b_obs_x.T).T  - jnp.matmul(self.P.T,x_guess.T).T
        lincost_y = -lambda_y-self.rho_obs*jnp.matmul(self.A_obs.T, b_obs_y.T).T - jnp.matmul(self.P.T,y_guess.T).T
        lincost_z = -lambda_z-self.rho_obs*jnp.matmul(self.A_obs.T, b_obs_z.T).T - jnp.matmul(self.P.T,z_guess.T).T
        cost_mat_inv_x = cost_mat_inv_y = cost_mat_inv_z =  jnp.linalg.inv(cost_mat_x)
        
        #pdb.set_trace()
        b_eq_x = np.hstack(( X_guess[:,0,0][:,jnp.newaxis] , X_guess[:,-1,0][:,jnp.newaxis]))
        sol_x = jnp.dot(cost_mat_inv_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        primal_sol_x = sol_x[:,0:self.nvar]
        x = jnp.dot(self.P_jax, primal_sol_x.T).T
    
        b_eq_y = np.hstack(( X_guess[:,0,1][:,jnp.newaxis]  , X_guess[:,-1,1][:,jnp.newaxis] ))
        sol_y = jnp.dot(cost_mat_inv_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T
        primal_sol_y = sol_y[:,0:self.nvar]
        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        
        
        b_eq_z = np.hstack(( X_guess[:,0,2][:,jnp.newaxis]  , X_guess[:,-1,2][:,jnp.newaxis] ))
        sol_z = jnp.dot(cost_mat_inv_z, jnp.hstack(( -lincost_z, b_eq_z )).T).T
        primal_sol_z = sol_z[:,0:self.nvar]
        z = jnp.dot(self.P_jax, primal_sol_z.T).T    
        
        X_sphere = jnp.stack((x,y,z),axis=2)
        #pdb.set_trace()
        return  X_sphere
    
    
    
    
    def cal_residual(self,X_guess):
        x_guess, y_guess,z_guess =  X_guess[:,:,0] ,X_guess[:,:,1] , X_guess[:,:,2]    #dim num_sphere*num_timesetep
        x_obs, y_obs,z_obs =  self.X_obs[:,0] ,self.X_obs[:,1] ,self.X_obs[:,2]         #dim num_obs
        x_obs= jnp.tile(x_obs,self.num_timestep)
        x_obs = x_obs.reshape((self.num_obs ,self.num_timestep)) #dim num_obs*num_timestep
        
        y_obs= jnp.tile(y_obs,self.num_timestep)
        y_obs = y_obs.reshape((self.num_obs ,self.num_timestep))        
        
        z_obs= jnp.tile(z_obs,self.num_timestep)
        z_obs = z_obs.reshape((self.num_obs ,self.num_timestep))   
        
        
        wc_alpha_temp = x_guess - x_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        ws_alpha_temp = y_guess - y_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        wc_beta_temp =  z_guess - z_obs[:,jnp.newaxis]  #dim num_obs*num_sphere*num_timestep
        
        wc_alpha = wc_alpha_temp.transpose(1,0,2)  #dim num_sphere*num_obs*num_timestep
        ws_alpha = ws_alpha_temp.transpose(1,0,2) #dim num_sphere*num_obs*num_timestep
        wc_beta = wc_beta_temp.transpose(1,0,2)  #dim num_sphere*num_obs*num_timestep
        
        wc_alpha = wc_alpha.reshape(self.num_sphere , self.num_timestep*self.num_obs)  #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        ws_alpha = ws_alpha.reshape(self.num_sphere , self.num_timestep*self.num_obs)  #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        wc_beta  = wc_beta.reshape(self.num_sphere , self.num_timestep*self.num_obs)   #dim num_sphere*(self.num_obs *self.num_timestep ) #2d array
        
        
        res_x_obs_vec = wc_alpha - self.a_obs*self.d_obs*jnp.sin(self.beta_obs)*jnp.cos(self.alpha_obs)
        res_y_obs_vec = ws_alpha - self.b_obs*self.d_obs*jnp.sin(self.beta_obs)*jnp.sin(self.alpha_obs)
        res_z_obs_vec = wc_beta - self.c_obs*self.d_obs*jnp.cos(self.beta_obs)
        v =jnp.hstack((res_x_obs_vec,res_y_obs_vec,res_z_obs_vec))
        #pdb.set_trace()
        residual = jnp.linalg.norm(v,axis=1).reshape((1,len(v))) #shape no_sphere*(self.num_obs *self.num_timestep) 2d array
        return res_x_obs_vec,res_y_obs_vec,res_z_obs_vec,residual
        
    def compute_lambda(self,X_guess):
        lambda_xyz = self.lambda_xyz # dim num_sphere*num_coff*3
        lambda_x = lambda_xyz[:,:,0] #dim num_sphere*num_coff
        lambda_y = lambda_xyz[:,:,1] #dim num_sphere*num_coff
        lambda_z = lambda_xyz[:,:,2] #dim num_sphere*num_coff
        
        res_x_obs_vec,res_y_obs_vec,res_z_obs_vec,residual = self.cal_residual(X_guess)
        lambda_x = lambda_x -self.rho_obs*np.dot(self.A_obs.T, res_x_obs_vec.T).T
        lambda_y = lambda_y -self.rho_obs*np.dot(self.A_obs.T, res_y_obs_vec.T).T
        lambda_z = lambda_z -self.rho_obs*np.dot(self.A_obs.T, res_z_obs_vec.T).T
        #pdb.set_trace()
        self.lambda_xyz = jnp.stack((lambda_x,lambda_y,lambda_z),axis=2) 
        #pdb.set_trace()
        return residual
    
    
    def solve(self,X_guess_init):
        X_sphere = self.compute_sphere_X(X_guess_init)  #dim = num_sphere*num_timesetep*3
        self.compute_alpha_beta_d_obs(X_sphere) #num_spheres*num_obs*num_timesteps*1
        residual = self.compute_lambda(X_sphere)  #dim num_sphere*coff
        X_sphere_next =np.asarray(X_sphere)
        #pdb.set_trace()
        return X_sphere_next,residual  #dim = num_sphere*num_timesetep*3