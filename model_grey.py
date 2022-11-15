from abc import abstractmethod
import torch
import torch.nn as nn

class ModelBlack(nn.Module):
    def __init__(self, controlled_system, lyapunov_correction, generator, dev, xref):
        super(ModelBlack, self).__init__()

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
        

        # state = [x, y, theta, dx, dy, dtheta], dynamics = [dx, dy, dtheta, ddx, ddy, ddtheta], acc = [ddx, ddy, ddtheta]
        # therefore dynamics[0:3] = state[3:6] and dynamics[3:6] = acc
        dX_X = torch.concat((X[:,3:6], acc), axis=1)       

        return dX_X





class HolohoverModelBlack(ModelBlack):
    def __init__(self, controlled_system, lyapunov_correction, generator, dev, xref):
        ModelBlack.__init__(self, controlled_system, lyapunov_correction, generator, dev, xref)

        # system parameters
        self.D = 6 # dim. of state x
        self.M = 6 # dim. of controll input u   
        self.S = 3 # dim. of space
        self.mass = 0 # mass of holohover
        # position of motors [x, y]   
        self.motors_pos = torch.tensor([[0,0,0], 
                                        [0,0,0],
                                        [0,0,0],
                                        [0,0,0],
                                        [0,0,0],
                                        [0,0,0],])
        # unit vectors pointing in direction of motors
        self.motors_vec = torch.tensor([[0,0,0], 
                                        [0,0,0],
                                        [0,0,0],
                                        [0,0,0],
                                        [0,0,0],
                                        [0,0,0],])

        # TNN: model parameters
        tnn_input_size = self.D
        tnn_hidden1_size = 80
        tnn_hidden2_size = 200
        tnn_output_size = self.D

        # TCNN: layers
        self.tnn_fc1 = nn.Linear(tnn_input_size, tnn_hidden1_size, bias=True)
        self.tnn_fc2 = nn.Linear(tnn_hidden1_size, tnn_hidden2_size, bias=True)
        self.tnn_fc3 = nn.Linear(tnn_hidden2_size, tnn_output_size, bias=True)

        # Center of mass
        self.com = torch.nn.parameterParameter(torch.zeros(self.S), requires_grad=True)

        # Inertia around z axis
        self.inertia = torch.nn.parameterParameter(torch.zeros(1), requires_grad=True)

    def signal2acc(self, U, X):
        thrust = self.forwardTNN(U) # (N, M)
        force = self.thrust2force(thrust) # (N, S)
        acc = self.force2acc(force, X) # (N, S)

        return acc

    def forwardTNN(self, U):
        """
        Description: motor signal to motor thrust mapping
        In U: motor signals batch, tensor (N x D)
        Out thrust: motor thrust, tensor (N x D)
        """
        x_fnn_fc1 = self.tnn_fc1(X)
        x_fnn_tanh1 = self.tanh(x_fnn_fc1)

        x_fnn_fc2 = self.tnn_fc2(x_fnn_tanh1)
        x_fnn_tanh2 = self.tanh(x_fnn_fc2)

        thrust = self.tnn_fc3(x_fnn_tanh2)
        return thrust

    def thrust2acc(self, thrust, X):
        """
        
        Args: norm of thrust from each motor, tensor (N,M)"""

        # calc. thrust vector for each motor
        thrust_vec = torch.einsum('nm,ms->nms', thrust, self.motors_vec) # (N, M, S)

        # calc. sum of forces in body and world frame
        Fb_sum = torch.sum(thrust_vec, axis=1) # (N, S)
        rotation_b2w = torch.tensor([   [torch.cos(X[:,2]), -torch.sin(X[:,2]), torch.zeros(X.shape[0])], # (S, S, N)
                                        [torch.sin(X[:,2]), torch.cos(X[:,2]), torch.zeros(X.shape[0])],
                                        [torch.zeros(X.shape[0]), torch.zeros(X.shape[0]), torch.zeros(X.shape[0])]])
        Fw_sum = torch.einsum('ns,psn->np', Fb_sum, rotation_b2w) # (N, S)

        # calc. sum of moments in body frame
        com2motor_vec = self.motors_pos - self.com # (M, S)
        Mb = torch.linalg.cross(com2motor_vec, thrust_vec, dim=2) # (N, M, S)
        Mb_sum = torch.sum(Mb, axis=1) # (N, S)

        # calc. acceleration, Fw_sum[0,:] = [Fx, Fy, Fz] and Mb[0,:] = [Mx, My, Mz]
        # holohover moves in a plane -> Fz = Mx = My = 0, also Mz_body = Mz_world
        acc = Fw_sum/self.mass + Mb_sum/self.inertia

        return acc
