from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from distribution_lqr_controller import DistributionLQRController

class DistributionTest():
    def __init__(self):      
        
        self.P = 0.1 # cycle periode
        self.nb_rounds = 10000000
        self.counter_max = 100
        self.counter = 0
        
        self.R = 0.5
        alpha_max = np.pi
        alpha_min = -np.pi
        vel_max = 0.2
        vel_min = -0.2
        dalpha_max = 0.1
        dalpha_min = -0.1

        self.border = np.max([vel_max,vel_min])*self.P
        pos_max = self.R + self.border
        pos_min = -self.R - self.border

        self.grid = np.array([0.05, 0.05, 0.2, 0.02, 0.02, 0.02]) # x, y, alpha, dx, dy, omega
        self.state_max = np.array([pos_max, pos_max, alpha_max, vel_max, vel_max, dalpha_max])
        self.state_min = np.array([pos_min, pos_min, alpha_min, vel_min, vel_min, dalpha_min])

        self.idx_max = []
        self.idx_max = self._state2idx(self.state_max) + 1

        self.field = np.zeros((self.idx_max))


    def runExperiment(self):
        # init. state and field
        state = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]) # x, y, alpha, dx, dy, omega

        for _ in range(self.nb_rounds):
            state = self.updateState(state)
            self.updateField(state)

        self.calcJointProb()
        self.plotField()

    def updateState(self, state):
        state = np.copy(state)

        # update position
        new_pos = state[0:3] + self.P*state[3:6]
        new_pos[2] = self._normalizeAngle(new_pos[2])

        # update velocity
        new_vel = state[3:6]
        self.counter -= 1
        if self.counter <= 0:
            new_vel[0] = np.random.rand(1)*(self.state_max[3]-self.state_min[3]) + self.state_min[3]
            new_vel[1] = np.random.rand(1)*(self.state_max[4]-self.state_min[4]) + self.state_min[4]
            new_vel[2] = np.random.rand(1)*(self.state_max[5]-self.state_min[5]) + self.state_min[5]
            self.counter = np.random.rand(1)*self.counter_max

        if (new_pos[0]>self.R and new_vel[0]>=0) or (new_pos[0]<-self.R and new_vel[0]<=0):
            new_vel[0] = -new_vel[0]
        if (new_pos[1]>self.R and new_vel[1]>=0) or (new_pos[1]<-self.R and new_vel[1]<=0):
            new_vel[1] = -new_vel[1]            

        return np.concatenate((new_pos, new_vel), axis=0)

    def updateField(self, state):
        state = np.copy(state)

        idxs = self._state2idx(state)

        self.field[tuple(idxs)] += 1

    def calcJointProb(self):
        
        dim_name = ['x','y','a','dx','dy','da']
        axes = np.arange(self.field.ndim)
        for i in range(0,self.field.ndim):
            for j in range(i+1,self.field.ndim):
                axes_ij = tuple(np.delete(axes, (i,j)))
                field_ij = np.sum(self.field, axis=axes_ij)

                prob_i = np.sum(field_ij, axis=1) / self.nb_rounds
                prob_j = np.sum(field_ij, axis=0) / self.nb_rounds
                prob_ij = field_ij / self.nb_rounds

                prob_error =  prob_ij - np.einsum('i,j->ij', prob_i, prob_j)
                prob_error = np.mean(np.abs(prob_error))
                print(f"Avg. error p({dim_name[i]},{dim_name[j]}) - p({dim_name[i]})*p({dim_name[j]}) = {prob_error}")


    def plotField(self):

        axis = self._axisState()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize =(12, 6))

        X, Y = np.meshgrid(axis[0], axis[1])
        Z = np.sum(self.field, axis=(2,3,4,5)) # collapse all dimensions except of x and y
        axs[0].set_title(f"Spatial density")
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_aspect('equal', 'box')
        c = axs[0].pcolormesh(X, Y, Z, cmap ='Blues', vmin=0, vmax=np.max(Z))
        fig.colorbar(c, ax=axs[0])

        X, Y = np.meshgrid(axis[3], axis[4])
        Z = np.sum(self.field, axis=(0,1,2,5)) # collapse all dimensions except of dx and dy
        axs[1].set_title(f"Velocity density")
        axs[1].set_xlabel('dx')
        axs[1].set_ylabel('dy')
        axs[1].set_aspect('equal', 'box')
        c = axs[1].pcolormesh(X, Y, Z, cmap ='Reds', vmin=0, vmax=np.max(Z))
        fig.colorbar(c, ax=axs[1])

        # x_axis = np.linspace(self.state_min[2], self.state_max[2], num=len(self.alpha_field))
        # x_axis = x_axis*180/np.pi
        # axs[1,0].set_title(f"Angular density")
        # axs[1,0].set_xlabel('alpha [°]')
        # axs[1,0].set_ylabel('Nb. events')  
        # axs[1,0].plot(x_axis, self.alpha_field, color="b") 

        # x_axis = np.linspace(self.state_min[5], self.state_max[5], num=len(self.dalpha_field))
        # x_axis = x_axis*180/np.pi
        # axs[1,1].set_title(f"Angular velocity density")
        # axs[1,1].set_xlabel('omega [°/s]')
        # axs[1,1].set_ylabel('Nb. events')  
        # axs[1,1].plot(x_axis, self.dalpha_field, color="r")      

        plt.show()

    def _normalizeAngle(self, angle):
        while angle > np.pi:
            angle -= 2*np.pi
        while angle <= -np.pi:
            angle += 2*np.pi
        return angle

    def _state2idx(self, state):
        state = np.copy(state)

        idxs = (state - self.state_min)/self.grid
        idxs = np.round(idxs).astype(int)

        # if len(self.idx_max) > 0:
        #     idxs = np.clip(idxs, a_min=0, a_max=self.idx_max-1, dtype=int)

        return idxs

    def _idx2state(self, idx):
        idx = np.copy(idx)

        return idx*self.grid + self.state_min

    def _axisState(self):
        x_axis = []
        for i in range(len(self.state_min)):
            x_axis.append(np.linspace(self.state_min[i], self.state_max[i], num=self.idx_max[i]))

            if i==2 or i==5:
                x_axis[i] = x_axis[i]*180/np.pi

        return x_axis



def main():
    dis = DistributionTest()
    dis.runExperiment()


if __name__ == "__main__":
    main()