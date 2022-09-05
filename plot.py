import matplotlib.pyplot as plt 
import numpy as np

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pdb
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, x2,y2,z2, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (x2-x, y2-y, z2-z)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)
def do_plot(cost_list,plotno):
    #dim cost_list = num_sphere*maxitr
    maxitr= cost_list.shape[1]
    maxitr = np.arange(maxitr)
    for itr in  range(cost_list.shape[0] ) :
        sphere_no = itr
        cost_sphere = cost_list[sphere_no,:]
        fig = plt.figure(itr, figsize=(12, 12), dpi=100)

        ax = fig.add_subplot(111)
        leg = ax.legend()
        ax.plot(maxitr, cost_sphere, '--or', linewidth=2.0, markersize=6.0,
            label=('init_residual=', np.asarray(cost_sphere)[0] , 'final-res=',np.asarray(cost_sphere)[-1]))
        ax.set_xlabel('ITR')
        ax.set_ylabel('COST-RESIDUALS')
        ax.legend(loc='upper left', frameon=False)
        plt.title(f"cost residual vs itr for sphere={sphere_no}")
        plt.savefig(f"plots/example_{plotno}/cost_residual/sphere_{sphere_no}")
        plt.close()
    return

def do_3dplot(X_next,X_guess_init,X_goal,obs_vector,plotno,index,min_distance_obs,min_dist_Arg_timeStep,min_distance_obs_init ,index_init,min_dist_Arg_timeStep_init,X_guess_traj_init):
    #dim cost_list = num_sphere*maxitr
    timestep= X_next.shape[1]
    maxitr = np.arange(timestep)
    for itr in  range(X_next.shape[0] ) :
        sphere_no = itr
        sphere_obs_distance = min_distance_obs[sphere_no,:]
        sphere_obs_distance_init = min_distance_obs_init[sphere_no,:]
        min_dist_Arg_timeStep_sphere = min_dist_Arg_timeStep[sphere_no,:]
        min_dist_Arg_timeStep_sphere_init = min_dist_Arg_timeStep_init[sphere_no,:]
        X_init = X_guess_init[sphere_no,:]
        X_final = X_goal[sphere_no,:]
        X_next_sphere = X_next[sphere_no,:,:]
        X_next_sphere_init = X_guess_traj_init[sphere_no,:,:]
        fig = plt.figure(itr, figsize=(12, 12), dpi=100)

        ax = fig.add_subplot(111,projection='3d')
        leg = ax.legend()
        ax.plot(X_next_sphere[:,0], X_next_sphere[:,1],X_next_sphere[:,2], '--or', linewidth=2.0, markersize=6.0,
            label=('final calculated traj' , 'final-res=',np.linalg.norm((X_next_sphere[-1,:]-X_final ))))
        
        ax.plot(X_next_sphere_init[:,0], X_next_sphere_init[:,1],X_next_sphere_init[:,2], '--og', linewidth=2.0, markersize=6.0,label=('initial guess traj'))
        
        ax.plot(X_init[0] * np.ones(1), X_init[1] * np.ones(1), X_init[2] * np.ones(1), 'om', markersize=15,label=('start_position'))
        ax.plot(X_final[0]* np.ones(1),X_final[1]* np.ones(1), X_final[2] * np.ones(1), 'og', markersize=10,label=('goal_position'))
        
        
        for obs_itr in range(obs_vector.shape[0]):
            if obs_itr == min_dist_Arg_timeStep_sphere[0]:
                ax.plot(obs_vector[obs_itr,0] * np.ones(1), obs_vector[obs_itr,1] * np.ones(1), obs_vector[obs_itr,2] * np.ones(1), 'ob', markersize=15)
            elif obs_itr == min_dist_Arg_timeStep_sphere_init[0]:
                ax.plot(obs_vector[obs_itr,0] * np.ones(1), obs_vector[obs_itr,1] * np.ones(1), obs_vector[obs_itr,2] * np.ones(1), 'or', markersize=15)
            else:
                if obs_itr == obs_vector.shape[0]-1:
                    ax.plot(obs_vector[obs_itr,0] * np.ones(1), obs_vector[obs_itr,1] * np.ones(1), obs_vector[obs_itr,2] * np.ones(1), 'ok', markersize=15,label =('obstacles'))
                else:
                    ax.plot(obs_vector[obs_itr,0] * np.ones(1), obs_vector[obs_itr,1] * np.ones(1), obs_vector[obs_itr,2] * np.ones(1), 'ok', markersize=15)
        
        ax.plot(X_next_sphere[min_dist_Arg_timeStep_sphere[1],0]* np.ones(1), X_next_sphere[min_dist_Arg_timeStep_sphere[1],1]* np.ones(1),X_next_sphere[min_dist_Arg_timeStep_sphere[1],2],'+k', markersize=20)
        ax.plot(X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],0]* np.ones(1), X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],1]* np.ones(1),X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],2],'+k', markersize=20)
        
        ax.arrow3D(obs_vector[min_dist_Arg_timeStep_sphere_init[0],0],obs_vector[min_dist_Arg_timeStep_sphere_init[0],1],obs_vector[min_dist_Arg_timeStep_sphere_init[0],2],
           X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],0],X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],1],X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],2],
           mutation_scale=20,
           arrowstyle="-|>",
           linestyle='dashed',fc='green',ec ='red',label=('min-distance b/w obs and traj=', np.min(np.asarray(sphere_obs_distance))))
        ax.arrow3D(obs_vector[min_dist_Arg_timeStep_sphere[0],0],obs_vector[min_dist_Arg_timeStep_sphere[0],1],obs_vector[min_dist_Arg_timeStep_sphere[0],2],
           X_next_sphere[min_dist_Arg_timeStep_sphere[1],0],X_next_sphere[min_dist_Arg_timeStep_sphere[1],1],X_next_sphere[min_dist_Arg_timeStep_sphere[1],2],
           mutation_scale=20,
           arrowstyle="-|>",
           linestyle='dashed',fc='red',label=('min-distance b/w obs and traj=',np.min(np.asarray(sphere_obs_distance_init))))
        #ax.arrow(obs_vector[min_dist_Arg_timeStep_sphere_init[0],0], obs_vector[min_dist_Arg_timeStep_sphere_init[0],1], X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],0] - obs_vector[min_dist_Arg_timeStep_sphere_init[0],0], X_next_sphere_init[min_dist_Arg_timeStep_sphere_init[1],1] - obs_vector[min_dist_Arg_timeStep_sphere_init[0],1],head_width=3, length_includes_head=True)
        ax.set_xlabel('x-coord')
        ax.set_ylabel('y-coord')
        ax.set_zlabel('z-coord')
        ax.legend(loc='upper left', frameon=False)
        plt.title(f"3d traj plot with avoiding obstalces={sphere_no}")
        #pdb.set_trace()
        plt.show()
        plt.savefig(f"plots/example_{plotno}/3d_plot/sphere_{sphere_no}")
        #if sphere_no ==index[0]:
        #plt.show()
        plt.close()
    return