import numpy as np
from control import lqr
from scipy import linalg


class DistributionLQRController():
    def __init__(self): 
        # linear system
        self.D = 6
        self.M = 3
        self.A = np.array( [[0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]] )
        self.B = np.array( [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]] )

        # LQR controller
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0] )
        self.R = np.diag( [1.0, 1.0, 1.0] )

        # controll limits
        self.u_max = np.array([0.3,0.3,0.1])
        self.u_min = np.array([-0.3,-0.3,-0.1])

        self.pos_max = np.array([0.5,0.5,np.pi])
        self.pos_min = np.array([-0.5,-0.5,-np.pi])

    def calcStateLimits(self):
        K = self.calcGain()
        K_pos = K[:,0:3]
        K_vel = K[:,3:6]
        K_pos_inv = np.diag([1,1,1])
        K_vel_inv = np.diag([0.5773672,0.5773672,0.5773672])

        print(f"K = {K}")
        print(f"K_pos = {K_pos_inv}")
        print(f"K_vel = {K_vel_inv}")

        p_u = 1 / (self.u_max-self.u_min)
        p_pos = 1 / (self.pos_max-self.pos_min)
        p_vel = np.linalg.solve(K_vel_inv, (p_u - K_pos_inv@p_pos) )
        p_state = np.concatenate((p_pos,p_vel), axis=0)

        print(f"pu = {p_u}")
        print(f"p_state: {p_state}")
        print(f"p_state check: {p_u-K@p_state}")       

        state_max = (1 / p_state)
        state_min = -(1 / p_state)
        
        print(f"state_max = {state_max}")
        print(f"state_min = {state_min}")

        x = state_max
        x[0:3] = [0,0,0]
        u = K@x
        print(f"u_max = {u}")

    

    def calcGain(self):
        K, S, E = lqr(self.A, self.B, self.Q, self.R)
        return K


def main():
    dis = DistributionLQRController()
    dis.calcStateLimits()


if __name__ == "__main__":
    main()