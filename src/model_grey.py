from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn

from src.args import Args

class ModelGrey(nn.Module):
    def __init__(self, dev):
        super(ModelGrey, self).__init__()

        self.device = dev 
 
        # activation fcts.
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sp = nn.Softplus() #softplus (smooth approx. of ReLU)
        self.tanh = nn.Tanh()

    @abstractmethod
    def signal2acc(self, U, X):
        pass

    def forward(self, X, U):
        """
        Description: forward pass through main model
        In X: state input batch (N, D)
        In U: controll input batch (N, M)
        Out dX_X: state derivative (N, D)
        """
        # ThrustNN: map signal to thrust
        acc = self.signal2acc(U, X)        

        # state = [x, y, theta, dx, dy, dtheta], dynamics = [dx, dy, dtheta, ddx, ddy, ddtheta], 
        # acc = [ddx, ddy, ddtheta] -> dynamics[0:3] = state[3:6] and dynamics[3:6] = acc
        dX_X = torch.concat((X[:,3:6], acc), axis=1)       

        return dX_X


class HolohoverModelGrey(ModelGrey):
    def __init__(self, args, dev):
        ModelGrey.__init__(self, dev)

        # system parameters
        self.D = 6 # dim. of state x
        self.M = 6 # dim. of controll input u   
        self.S = 3 # dim. of space

        # holohover params
        self.mass = 0.0983 # mass of holohover
        self.motor_distance = 0.046532 # distance form center (0,0,0) in robot frame to motors
        self.motor_angle_offset = 0.0 # offset angle of first motor pair (angle between motor 1 and motor 2)
        self.motor_angel_delta = 0.328220 # angle between center of motor pair and motors to the left and right
        self.motors_vec, self.motors_pos = self.initMotorPosVec() # unit vectors of thrust from motors, motor positions
        self.init_center_of_mass = torch.zeros(self.S) # initial center of mass
        self.init_inertia = torch.tensor([0.0003599]) # intitial inertia
        self.init_signal2thrust = torch.tensor([[0.19089428, 0.73228747, -0.27675822], # initial signal2thrust coeff.
                                                [0.21030774, 0.71018331, -0.26599642], # tensor (M, poly_expand_U)
                                                [0.08407954, 1.01239162, -0.47014436], # for each motor [a1, a2, a3]
                                                [0.23272980, 0.63304683, -0.19689546], # where thrust = a1*u + a2*u^2 + a3*u^3
                                                [0.16751077, 0.85337578, -0.34435633],
                                                [0.19129567, 0.82140377, -0.33533427] ])

        # we measured the signal2thrust coeff. of degree=3, 
        # if a different polynomial expansion is desired the shape of init_signal2thrust must be adapted                           
        if args.poly_expand_U < 3: # desired polynomial expansion has smaller degree than measured coeff. -> ignore higher coeff.
            self.init_signal2thrust = self.init_signal2thrust[:,:args.poly_expand_U]
        if args.poly_expand_U > 3: # desired polynomial expansion has larger degree than measured coeff. -> add coeff. = zero
            padding = torch.zeros(self.M, args.poly_expand_U-3)
            self.init_signal2thrust = torch.concat((self.init_signal2thrust, padding), axis=1)
        

        # signal2thrust mapping
        tnn_input_size = args.poly_expand_U
        tnn_output_size = 1
        self.tnn_sig2thr_fcts = [nn.Linear(tnn_input_size, tnn_output_size, bias=False, dtype=torch.float32) for _ in range(self.M)]

        for i, lin_fct in enumerate(self.tnn_sig2thr_fcts):
            lin_fct.weight = torch.nn.parameter.Parameter(self.init_signal2thrust[i,:].detach().clone().reshape(1, args.poly_expand_U))
            if args.learn_signal2thrust:
                lin_fct.weight.requires_grad = True
            else:
                lin_fct.weight.requires_grad = False        

        # Center of mass
        self.center_of_mass = torch.nn.parameter.Parameter(self.init_center_of_mass.detach().clone())
        if args.learn_center_of_mass:
            self.center_of_mass.requires_grad = True
        else:
            self.center_of_mass.requires_grad = False

        # Inertia around z axis
        self.inertia = torch.nn.parameter.Parameter(self.init_inertia.detach().clone())
        if args.learn_inertia:
            self.inertia.requires_grad = True
        else:
            self.inertia.requires_grad = False

    def initMotorPosVec(self):
        """
        Initiate motors vector and motors position
        Returns:
            motors_vec: unit vector pointing in direction of thrust from each motor, tensor (M,S)
            motors_pos: position of motors in robot frame, tensor (M,S)
        """
        
        motors_vec = torch.zeros(self.M, self.S)
        motors_pos = torch.zeros(self.M, self.S)

        for j in np.arange(0, self.M, step=2):
            angle_motor_pair = self.motor_angle_offset  + j*np.pi/3
            motors_vec[j,:] = torch.tensor([-np.sin(angle_motor_pair), np.cos(angle_motor_pair), 0.0])
            motors_vec[j+1,:] = torch.tensor([np.sin(angle_motor_pair), -np.cos(angle_motor_pair), 0.0])

            angle_first_motor = angle_motor_pair - self.motor_angel_delta
            angle_second_motor = angle_motor_pair + self.motor_angel_delta
            motors_pos[j,:] = self.motor_distance * torch.tensor([np.cos(angle_first_motor), np.sin(angle_first_motor), 0.0])
            motors_pos[j+1,:] = self.motor_distance * torch.tensor([np.cos(angle_second_motor), np.sin(angle_second_motor), 0.0])

        return motors_vec.detach().clone(), motors_pos.detach().clone()


    def signal2acc(self, U, X):
        thrust = self.signal2thrust(U) # (N, M)
        acc = self.thrust2acc(thrust, X) # (N, S)



        return acc

    def signal2thrust(self, U):
        """
        Description: motor signal to motor thrust mapping
        In U: motor signals batch, tensor (N, M*poly_expand_U)
        Out thrust: motor thrust, tensor (N, M)
        """
        assert U.shape[1]%self.M == 0
        deg = int(U.shape[1] / self.M) # degree of polynomial expansion

        thrust = torch.zeros((U.shape[0], self.M), dtype=torch.float32)
        for i, lin_fct in enumerate(self.tnn_sig2thr_fcts):
            thrust[:,i] = lin_fct(U[:,int(i*deg):int((i+1)*deg)]).flatten()

        return thrust

    def thrust2acc(self, thrust, X):
        """       
        Args: 
            thrust: norm of thrust from each motor, tensor (N,M)
            X: current state [x, y, theta, dx, dy, dtheta], tensor (N,D)
        Returns:
            acc: acceleration of holohover [ddx, ddy, ddtheta], tensor (N, S)
        """

        # calc. thrust vector for each motor
        thrust_vec = torch.einsum('nm,ms->nms', thrust, self.motors_vec) # (N, M, S)

        # calc. sum of forces in body and world frame
        Fb_sum = torch.sum(thrust_vec, axis=1) # (N, S)
        rotation_b2w = self.rotMatrix(X[:,2]) # (N,S,S)
        Fw_sum = torch.einsum('ns,nps->np', Fb_sum, rotation_b2w) # (N, S)

        # calc. sum of moments in body frame
        com2motor_vec = self.motors_pos - self.center_of_mass # (M, S)
        com2motor_vec = com2motor_vec.reshape(1,self.M,self.S).tile(thrust.shape[0],1,1) # (N,M,S)
        Mb = torch.linalg.cross(com2motor_vec, thrust_vec) # (N, M, S)
        Mb_sum = torch.sum(Mb, axis=1) # (N, S)

        # calc. acceleration, Fw_sum[0,:] = [Fx, Fy, Fz] and Mb[0,:] = [Mx, My, Mz]
        # holohover moves in a plane -> Fz = Mx = My = 0, also Mz_body = Mz_world
        acc = Fw_sum/self.mass #+ Mb_sum/self.inertia

        return acc

    def rotMatrix(self, theta):
        """
        Calc. 3D rotational matrix for batch
        Args:
            theta: rotation aroung z-axis in world frame, tensor (N)
        Returns:
            rot_mat: rotational matrix, tensor (N,S,S)
        """
        rot_mat = torch.zeros(theta.shape[0], self.S, self.S)
        cos = torch.cos(theta) # (N)
        sin = torch.sin(theta) # (N)
        rot_mat[:,0,0] = cos
        rot_mat[:,1,1] = cos
        rot_mat[:,0,1] = -sin
        rot_mat[:,1,0] = sin
        rot_mat[:,2,2] = torch.ones(theta.shape[0])
        return rot_mat




def main():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev) 

    args = Args(model_type="HolohoverGrey")
    model = HolohoverModelGrey(args=args, dev=device)

    # U = torch.tensor(  [[1.0, 1.0, 1.0, 0.0 ,0.0 ,0.0, 1.0, 1.0, 1.0, 0.0 ,0.0 ,0.0, 0.0, 0.0, 0.0, 0.0 ,0.0 ,0.0],
    #                     [1.0, 1.0, 1.0, 0.0 ,0.0 ,0.0, 1.0, 1.0, 1.0, 0.0 ,0.0 ,0.0, 0.0, 0.0, 0.0, 0.0 ,0.0 ,0.0]])

    # # U = torch.tensor(  [[1.0, 1.0, 1.0, 0.0 ,0.0 ,0.0],
    # #                     [1.0, 1.0, 1.0, 0.0 ,0.0 ,0.0]])

    # thrust = model.signal2thrust(U)

    

    X = torch.tensor(  [[0,0,0,0,0,0],
                        [0,0,1.0472,0,0,0]])
    thrust = torch.tensor( [[0.0, 0, 1, 0, 1, 0],
                            [1.0, 0, 0, 0, 0, 1]])
    
    acc = model.thrust2acc(thrust, X)
    print(thrust)
    print(acc)


if __name__ == "__main__":
    main()  
