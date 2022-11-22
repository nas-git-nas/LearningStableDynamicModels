from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn

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

        # TNN: model parameters
        tnn_input_size = 1
        tnn_hidden1_size = 80
        tnn_hidden2_size = 120
        tnn_hidden3_size = 40
        tnn_output_size = 1

        # TCNN: layers
        self.tnn1_fc1 = nn.Linear(tnn_input_size, tnn_hidden1_size, bias=True)
        self.tnn1_fc2 = nn.Linear(tnn_hidden1_size, tnn_hidden2_size, bias=True)
        self.tnn1_fc3 = nn.Linear(tnn_hidden2_size, tnn_hidden3_size, bias=True)
        self.tnn1_fc4 = nn.Linear(tnn_hidden3_size, tnn_output_size, bias=True)
        self.tnn2_fc1 = nn.Linear(tnn_input_size, tnn_hidden1_size, bias=True)
        self.tnn2_fc2 = nn.Linear(tnn_hidden1_size, tnn_hidden2_size, bias=True)
        self.tnn2_fc3 = nn.Linear(tnn_hidden2_size, tnn_hidden3_size, bias=True)
        self.tnn2_fc4 = nn.Linear(tnn_hidden3_size, tnn_output_size, bias=True)
        self.tnn3_fc1 = nn.Linear(tnn_input_size, tnn_hidden1_size, bias=True)
        self.tnn3_fc2 = nn.Linear(tnn_hidden1_size, tnn_hidden2_size, bias=True)
        self.tnn3_fc3 = nn.Linear(tnn_hidden2_size, tnn_hidden3_size, bias=True)
        self.tnn3_fc4 = nn.Linear(tnn_hidden3_size, tnn_output_size, bias=True)
        self.tnn4_fc1 = nn.Linear(tnn_input_size, tnn_hidden1_size, bias=True)
        self.tnn4_fc2 = nn.Linear(tnn_hidden1_size, tnn_hidden2_size, bias=True)
        self.tnn4_fc3 = nn.Linear(tnn_hidden2_size, tnn_hidden3_size, bias=True)
        self.tnn4_fc4 = nn.Linear(tnn_hidden3_size, tnn_output_size, bias=True)
        self.tnn5_fc1 = nn.Linear(tnn_input_size, tnn_hidden1_size, bias=True)
        self.tnn5_fc2 = nn.Linear(tnn_hidden1_size, tnn_hidden2_size, bias=True)
        self.tnn5_fc3 = nn.Linear(tnn_hidden2_size, tnn_hidden3_size, bias=True)
        self.tnn5_fc4 = nn.Linear(tnn_hidden3_size, tnn_output_size, bias=True)
        self.tnn6_fc1 = nn.Linear(tnn_input_size, tnn_hidden1_size, bias=True)
        self.tnn6_fc2 = nn.Linear(tnn_hidden1_size, tnn_hidden2_size, bias=True)
        self.tnn6_fc3 = nn.Linear(tnn_hidden2_size, tnn_hidden3_size, bias=True)
        self.tnn6_fc4 = nn.Linear(tnn_hidden3_size, tnn_output_size, bias=True)
        

        # Center of mass
        if args.learn_center_of_mass:
            self.center_of_mass = torch.nn.parameter.Parameter(self.init_center_of_mass.detach().clone(), requires_grad=True)
        else:
            self.center_of_mass = self.init_center_of_mass.detach().clone()

        # Inertia around z axis
        if args.learn_inertia:
            self.inertia = torch.nn.parameter.Parameter(self.init_inertia.detach().clone(), requires_grad=True)
        else:
            self.inertia = self.init_inertia.detach().clone()

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
        In U: motor signals batch, tensor (N x M)
        Out thrust: motor thrust, tensor (N x M)
        """
        x_tnn1_fc1 = self.tnn1_fc1(U[:,0].reshape(U.shape[0],1))
        x_tnn1_tanh1 = self.tanh(x_tnn1_fc1)
        x_tnn1_fc2 = self.tnn1_fc2(x_tnn1_tanh1)
        x_tnn1_tanh2 = self.tanh(x_tnn1_fc2)
        x_tnn1_fc3 = self.tnn1_fc3(x_tnn1_tanh2)
        x_tnn1_tanh3 = self.tanh(x_tnn1_fc3)
        t1 = self.tnn1_fc4(x_tnn1_tanh3)

        x_tnn2_fc1 = self.tnn2_fc1(U[:,1].reshape(U.shape[0],1))
        x_tnn2_tanh1 = self.tanh(x_tnn2_fc1)
        x_tnn2_fc2 = self.tnn2_fc2(x_tnn2_tanh1)
        x_tnn2_tanh2 = self.tanh(x_tnn2_fc2)
        x_tnn2_fc3 = self.tnn2_fc3(x_tnn2_tanh2)
        x_tnn2_tanh3 = self.tanh(x_tnn2_fc3)
        t2 = self.tnn2_fc4(x_tnn2_tanh3)

        x_tnn3_fc1 = self.tnn3_fc1(U[:,2].reshape(U.shape[0],1))
        x_tnn3_tanh1 = self.tanh(x_tnn3_fc1)
        x_tnn3_fc2 = self.tnn3_fc2(x_tnn3_tanh1)
        x_tnn3_tanh2 = self.tanh(x_tnn3_fc2)
        x_tnn3_fc3 = self.tnn3_fc3(x_tnn3_tanh2)
        x_tnn3_tanh3 = self.tanh(x_tnn3_fc3)
        t3 = self.tnn3_fc4(x_tnn3_tanh3)

        x_tnn4_fc1 = self.tnn4_fc1(U[:,3].reshape(U.shape[0],1))
        x_tnn4_tanh1 = self.tanh(x_tnn4_fc1)
        x_tnn4_fc2 = self.tnn4_fc2(x_tnn4_tanh1)
        x_tnn4_tanh2 = self.tanh(x_tnn4_fc2)
        x_tnn4_fc3 = self.tnn4_fc3(x_tnn4_tanh2)
        x_tnn4_tanh3 = self.tanh(x_tnn4_fc3)
        t4 = self.tnn4_fc4(x_tnn4_tanh3)

        x_tnn5_fc1 = self.tnn5_fc1(U[:,4].reshape(U.shape[0],1))
        x_tnn5_tanh1 = self.tanh(x_tnn5_fc1)
        x_tnn5_fc2 = self.tnn5_fc2(x_tnn5_tanh1)
        x_tnn5_tanh2 = self.tanh(x_tnn5_fc2)
        x_tnn5_fc3 = self.tnn5_fc3(x_tnn5_tanh2)
        x_tnn5_tanh3 = self.tanh(x_tnn5_fc3)
        t5 = self.tnn5_fc4(x_tnn5_tanh3)

        x_tnn6_fc1 = self.tnn6_fc1(U[:,5].reshape(U.shape[0],1))
        x_tnn6_tanh1 = self.tanh(x_tnn6_fc1)
        x_tnn6_fc2 = self.tnn6_fc2(x_tnn6_tanh1)
        x_tnn6_tanh2 = self.tanh(x_tnn6_fc2)
        x_tnn6_fc3 = self.tnn6_fc3(x_tnn6_tanh2)
        x_tnn6_tanh3 = self.tanh(x_tnn6_fc3)
        t6 = self.tnn6_fc4(x_tnn6_tanh3)

        return torch.concat((t1,t2,t3,t4,t5,t6), axis=1)

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
        acc = Fw_sum/self.mass + Mb_sum/self.inertia

        return acc

    def rotMatrix(self, theta):
        """
        Calc. 3D rotational matrix for batch
        Args:
            theta: rotation aroung z-axis, tensor (N)
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

    model = HolohoverModelGrey(dev=device)

    thrust = torch.tensor( [[1,0,1,0,0,0],
                            [0,0,2,0,2,0]])
    X = torch.tensor(  [[0,0,0.7854,0,0,0],
                        [0,0,3.142,0,0,0]])

    acc = model.thrust2acc(thrust, X)

if __name__ == "__main__":
    main()  
