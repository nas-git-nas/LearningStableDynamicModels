from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

class DistributionTest():
    def __init__(self):
        self.field_type = "squared"
        # self.field_type = "circle"

        self.R = 0.5
        
        self.P = 0.1 # cycle periode
        self.nb_rounds = 1000000
        

        
        alpha_max = np.pi
        alpha_min = -np.pi
        self.vel_max = 0.2
        self.vel_min = -0.2
        self.dalpha_max = 0.2
        self.dalpha_min = -0.2

        self.border = np.ceil(np.max([self.vel_max,self.vel_min])*self.P)
        pos_max = self.R + self.border
        pos_min = -self.R - self.border

        self.grid = np.array([0.1, 0.1, 0.6, 0.04, 0.04, 0.04]) # x, y, alpha, dx, dy, omega
        self.state_max = np.array([pos_max, pos_max, alpha_max, self.vel_max, self.vel_max, self.dalpha_max])
        self.state_min = np.array([pos_min, pos_min, alpha_min, self.vel_min, self.vel_min, self.dalpha_min])

        self.idx_max = self._state2idx(self.state_max) + 1
        self.spacial_field = np.zeros((self.idx_max[0],self.idx_max[1]))
        self.vel_field = np.zeros((self.idx_max[3],self.idx_max[4]))
        self.alpha_field = np.zeros((self.idx_max[2]))
        self.dalpha_field = np.zeros((self.idx_max[5]))
        
        # self.grid = 0.1
        # self.nb_boarder_grids = np.ceil(4*np.max([self.vel_max,self.vel_min])*self.P/self.grid).astype(int)
        # self.nb_field_grids = np.ceil(2*self.R/self.grid).astype(int)
        # self.nb_pos_grids = int(self.nb_boarder_grids + self.nb_field_grids + 1)
        # self.spacial_field = np.zeros((self.nb_pos_grids,self.nb_pos_grids))

        # self.vel_grid = 0.04
        # self.nb_vel_grids = np.ceil((self.vel_max-self.vel_min)/self.vel_grid + 1).astype(int)
        # self.vel_field = np.zeros((self.nb_vel_grids,self.nb_vel_grids))

        # self.angle_grid = 0.6
        # self.nb_angle_grids = np.ceil((2*np.pi)/self.angle_grid + 1).astype(int)
        # self.angle_field = np.zeros((self.nb_angle_grids))

        # self.omega_grid = 0.04
        # self.nb_omega_grids = np.ceil((self.omega_max-self.omega_min)/self.omega_grid + 1).astype(int)
        # self.omega_field = np.zeros((self.nb_omega_grids))

    def runExperiment(self):
        # init. state and field
        state = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]) # x, y, alpha, dx, dy, omega

        for _ in range(self.nb_rounds):
            state = self.updateState(state)
            self.updateField(state)

        self.plotField()

    def updateState(self, state):
        state = np.copy(state)
        if self.field_type == "squared":
            return self.updateSquared(state)

        if self.field_type == "circle":
            return self.updateCircle(state)


    def updateSquared(self, state):
        state = np.copy(state)

        # update position
        new_pos = state[0:3] + self.P*state[3:6]
        new_pos[2] = self.normalizeAngle(new_pos[2])

        # update velocity
        new_vel = state[3:6]
        if new_pos[0]>self.R:
            new_vel[0] = np.random.rand(1)*self.vel_min
            new_vel[1] = np.random.rand(1)*(self.vel_max-self.vel_min) + self.vel_min
            new_vel[2] = np.random.rand(1)*(self.dalpha_max-self.dalpha_min) + self.dalpha_min
        if new_pos[0]<-self.R:
            new_vel[0] = np.random.rand(1)*self.vel_max
            new_vel[1] = np.random.rand(1)*(self.vel_max-self.vel_min) + self.vel_min
            new_vel[2] = np.random.rand(1)*(self.dalpha_max-self.dalpha_min) + self.dalpha_min
        if new_pos[1]>self.R:
            new_vel[0] = np.random.rand(1)*(self.vel_max-self.vel_min) + self.vel_min
            new_vel[1] =  np.random.rand(1)*self.vel_min
            new_vel[2] = np.random.rand(1)*(self.dalpha_max-self.dalpha_min) + self.dalpha_min
        if new_pos[1]<-self.R:
            new_vel[0] = np.random.rand(1)*(self.vel_max-self.vel_min) + self.vel_min
            new_vel[1] = np.random.rand(1)*self.vel_max
            new_vel[2] = np.random.rand(1)*(self.dalpha_max-self.dalpha_min) + self.dalpha_min        

        return np.concatenate((new_pos, new_vel), axis=0)

    def updateCircle(self, state):
        pass

    def updateField(self, state):
        state = np.copy(state)

        idxs = self._state2idx(self.state_max)

        self.spacial_field[idxs[0], idxs[1]] += 1
        self.vel_field[idxs[3], idxs[4]] += 1
        self.alpha_field[idxs[2]] += 1
        self.dalpha_field[idxs[5]] += 1

    def plotField(self):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(12, 12))

        axs[0,0].set_title(f"Spatial density")
        axs[0,0].set_xlabel('x')
        axs[0,0].set_ylabel('y')
        axs[0,0].set_aspect('equal', 'box')
        c = axs[0,0].pcolormesh(self.spacial_field, cmap ='Blues', vmin=0, vmax=np.max(self.spacial_field))
        fig.colorbar(c, ax=axs[0,0])

        axs[0,1].set_title(f"Velocity density")
        axs[0,1].set_xlabel('dx')
        axs[0,1].set_ylabel('dy')
        axs[0,1].set_aspect('equal', 'box')
        c = axs[0,1].pcolormesh(self.vel_field, cmap ='Reds', vmin=0, vmax=np.max(self.vel_field))
        fig.colorbar(c, ax=axs[0,1])

        axs[1,0].set_title(f"Angular density")
        axs[1,0].set_xlabel('alpha')
        axs[1,0].set_ylabel('Nb. events')  
        axs[1,0].hist(self.alpha_field, bins=self.nb_angle_grids, color="b") 

        axs[1,1].set_title(f"Angular velocity density")
        axs[1,1].set_xlabel('omega')
        axs[1,1].set_ylabel('Nb. events')  
        axs[1,1].hist(self.dalpha_field, bins=self.nb_omega_grids, color="r")      

        plt.show()

    def normalizeAngle(self, angle):
        while angle > np.pi:
            angle -= 2*np.pi
        while angle <= -np.pi:
            angle += 2*np.pi
        return angle

    def _state2idx(self, state):
        state = np.copy(state)

        idx = (state - self.state_min)/self.grid
        return np.round(idx).astype(int)

    def _idx2state(self, idx):
        idx = np.copy(idx)

        return idx*self.grid + self.state_min



def main():
    dis = DistributionTest()
    dis.runExperiment()


if __name__ == "__main__":
    main()