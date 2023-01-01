import numpy as np
import torch
import torch.nn as nn

class ModelGrey(nn.Module):
    def __init__(self, dev):
        """
        Args:
            dev: pytorch device
        """
        super(ModelGrey, self).__init__()
        self.device = dev 

        # system parameters
        self.D = 6 # dim. of state x
        self.M = 6 # dim. of controll input u   
        self.S = 3 # dim. of space
 
        # activation fcts.
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sp = nn.Softplus() #softplus (smooth approx. of ReLU)
        self.tanh = nn.Tanh()

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


class HolohoverModelGrey(ModelGrey):
    def __init__(self, args, params, dev):
        """
        Args:
            args: argument class instance
            params: parameter class instance
            dev: pytorch device
        """
        ModelGrey.__init__(self, dev)

        # Center of mass
        self.center_of_mass = torch.nn.parameter.Parameter(torch.tensor(params.center_of_mass))
        if args.learn_center_of_mass:
            self.center_of_mass.requires_grad = True
        else:
            self.center_of_mass.requires_grad = False

        # mass
        self.mass = torch.nn.parameter.Parameter(torch.tensor(params.mass))
        if args.learn_mass:
            self.mass.requires_grad = True
        else:
            self.mass.requires_grad = False

        # Inertia around z axis
        self.inertia = torch.nn.parameter.Parameter(torch.tensor(params.inertia))
        if args.learn_inertia:
            self.inertia.requires_grad = True
        else:
            self.inertia.requires_grad = False

        
        # unit vectors of thrust from motors
        self.motors_vec = torch.nn.parameter.Parameter(self.initMotorVec(params).detach().clone())
        if args.learn_motors_vec:
            self.motors_vec.requires_grad = True
        else:
            self.motors_vec.requires_grad = False

        # motor positions
        self.init_motors_pos = self.initMotorPos(params)
        self.motors_pos = torch.nn.parameter.Parameter(self.init_motors_pos.detach().clone())
        if args.learn_motors_pos:
            self.motors_pos.requires_grad = True
        else:
            self.motors_pos.requires_grad = False

        # if poly_expand_U < 3 or poly_expand_U > 3, then signal2thrust and thrust2signal must be cropped or extended resp. 
        init_signal2thrust = torch.tensor(params.signal2thrust)
        init_thrust2signal = torch.tensor(params.thrust2signal)                         
        if args.poly_expand_U < 3: # desired polynomial expansion has smaller degree than measured coeff. -> ignore higher coeff.
            init_signal2thrust = init_signal2thrust[:,:args.poly_expand_U]
            init_thrust2signal = init_thrust2signal[:,:args.poly_expand_U]
        elif args.poly_expand_U > 3: # desired polynomial expansion has larger degree than measured coeff. -> add coeff. = zero
            padding = torch.zeros(self.M, args.poly_expand_U-3)
            init_signal2thrust = torch.concat((init_signal2thrust, padding), axis=1)
            init_thrust2signal = torch.concat((init_thrust2signal, padding), axis=1)

        # signal2thrust mapping
        input_size = args.poly_expand_U
        output_size = 1
        self.sig2thr_fcts = nn.ModuleList([nn.Linear(input_size, output_size, bias=False, dtype=torch.float32) for _ in range(self.M)])
        for i, lin_fct in enumerate(self.sig2thr_fcts):
            lin_fct.weight = torch.nn.parameter.Parameter(init_signal2thrust[i,:].detach().clone().reshape(1, args.poly_expand_U))
            if args.learn_signal2thrust:
                lin_fct.weight.requires_grad = True
            else:
                lin_fct.weight.requires_grad = False    

        # thrust2signal mapping
        input_size = args.poly_expand_U
        output_size = 1
        self.thr2sig_fcts = nn.ModuleList([nn.Linear(input_size, output_size, bias=False, dtype=torch.float32) for _ in range(self.M)])
        for i, lin_fct in enumerate(self.thr2sig_fcts):
            lin_fct.weight = torch.nn.parameter.Parameter(init_thrust2signal[i,:].detach().clone().reshape(1, args.poly_expand_U))
            lin_fct.weight.requires_grad = False   

    def initMotorPos(self, params):
        """
        Initiate motors position
        Args:
            params: parameters, Params class
        Returns:
            motors_pos: position of motors in robot frame, tensor (M,S)
        """ 
        motors_pos = torch.zeros(self.M, self.S)

        for j in np.arange(0, self.M, step=2):
            angle_motor_pair = params.motor_angle_offset  + j*np.pi/3
            angle_first_motor = angle_motor_pair - params.motor_angel_delta
            angle_second_motor = angle_motor_pair + params.motor_angel_delta

            motors_pos[j,:] = params.motor_distance * torch.tensor([np.cos(angle_first_motor), np.sin(angle_first_motor), 0.0])
            motors_pos[j+1,:] = params.motor_distance * torch.tensor([np.cos(angle_second_motor), np.sin(angle_second_motor), 0.0])

        return motors_pos.detach().clone()

    def initMotorVec(self, params):
        """
        Initiate motors vector
        Args:
            params: parameters, Params class
        Returns:
            motors_vec: unit vector pointing in direction of thrust from each motor, tensor (M,S)
        """ 
        motors_vec = torch.zeros(self.M, self.S)

        for j in np.arange(0, self.M, step=2):
            angle_motor_pair = params.motor_angle_offset  + j*np.pi/3
            motors_vec[j,:] = torch.tensor([-np.sin(angle_motor_pair), np.cos(angle_motor_pair), 0.0])
            motors_vec[j+1,:] = torch.tensor([np.sin(angle_motor_pair), -np.cos(angle_motor_pair), 0.0])

        return motors_vec.detach().clone()

    def forward(self, X, U):
        """
        Forward pass through main model
        Args:
            X: state input batch (N, D)
            U: controll input batch (N, M)
        Returns:
            dX_X: state derivative (N, D)
        """
        acc = self.signal2acc(U, X)      
        dX_X = torch.concat((X[:,3:6], acc), axis=1)       
        return dX_X


    def signal2acc(self, U, X):
        """
        Calc acceleration with current state and control input
        Args:
            X: state input batch (N, D)
            U: controll input batch (N, M)
        Returns:
            acc: acceleration (N, S)
        """
        thrust = self.signal2thrust(U) # (N, M)
        acc = self.thrust2acc(thrust, X) # (N, S)
        return acc

    def signal2thrust(self, U):
        """
        Motor signal to motor thrust mapping
        Args:
            U: motor signals batch, tensor (N, M*poly_expand_U)
        Returns:
            thrust: motor thrust, tensor (N, M)
        """
        assert U.shape[1]%self.M == 0
        deg = int(U.shape[1] / self.M) # degree of polynomial expansion

        thrust = torch.zeros((U.shape[0], self.M), dtype=torch.float32)
        for i, lin_fct in enumerate(self.sig2thr_fcts):
            thrust[:,i] = lin_fct(U[:,int(i*deg):int((i+1)*deg)]).flatten()

        return thrust

    def thrust2signal(self, thrust):
        """
        Motor thrust to motor signal mapping
        Args:
            thrust: motor thrust, tensor (N, M*poly_expand_U)
        Returns:
            U: motor signals batch, tensor (N, M)
        """
        assert thrust.shape[1]%self.M == 0
        deg = int(thrust.shape[1] / self.M) # degree of polynomial expansion

        U = torch.zeros((thrust.shape[0], self.M), dtype=torch.float32)
        for i, lin_fct in enumerate(self.thr2sig_fcts):
            U[:,i] = lin_fct(thrust[:,int(i*deg):int((i+1)*deg)]).flatten()

        return U

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


class CorrectModelGrey(ModelGrey):
    def __init__(self, args, dev):
        """
        Args:
            args: argument class instance
            dev: pytorch device
        """
        ModelGrey.__init__(self, dev)

        hidden_size = 64
        nb_hidden_layers = 4

        nnx_input_size = [self.M*args.poly_expand_U + 1]
        nny_input_size = [self.M*args.poly_expand_U + 1]
        nnt_input_size = [self.M*args.poly_expand_U]
        nnx_input_size.extend([hidden_size]*nb_hidden_layers)
        nny_input_size.extend([hidden_size]*nb_hidden_layers) 
        nnt_input_size.extend([hidden_size]*nb_hidden_layers)

        output_size = [hidden_size]*nb_hidden_layers
        output_size.append(1)
        

        self.nnx_lin_fcts = nn.ModuleList([nn.Linear(input, output, bias=True, dtype=torch.float32) 
                                            for (input, output) in zip(nnx_input_size,output_size)])
        self.nny_lin_fcts = nn.ModuleList([nn.Linear(input, output, bias=True, dtype=torch.float32) 
                                            for (input, output) in zip(nny_input_size,output_size)])
        self.nnt_lin_fcts = nn.ModuleList([nn.Linear(input, output, bias=True, dtype=torch.float32) 
                                            for (input, output) in zip(nnt_input_size,output_size)])

    def forward(self, X, U):
        """
        Forward pass through main model
        Args:
            X: state input batch (N, D)
            U: controll input batch (N, M)
        Returns:
            dX_X: state derivative (N, D)
        """
        acc = torch.zeros(X.shape[0],self.S)

        acc[:,0] = self.forwardCorr(Y=torch.concat([U,X[:,2,np.newaxis]], axis=1), lin_fcts=self.nnx_lin_fcts)
        acc[:,1] = self.forwardCorr(Y=torch.concat([U,X[:,2,np.newaxis]], axis=1), lin_fcts=self.nny_lin_fcts)
        acc[:,2] = self.forwardCorr(Y=U, lin_fcts=self.nnt_lin_fcts)
        
        return acc

    def forwardCorr(self, Y, lin_fcts):
        """
        Correct acceleration of grey box model
        Args:
            Y: control input concatenated with theta, tensor (N,M+1)
            lin_fcts: list of linear functions to apply, nn.ModuleList
        """
        for i, lin in enumerate(lin_fcts[0:-1]):
            Y = lin(Y)
            Y = self.tanh(Y)
        
        return lin_fcts[-1](Y).flatten()
