from abc import abstractmethod
import torch
import torch.nn as nn

class ModelBlack(nn.Module):
    def __init__(self, args, dev, system, xref):
        """
        Args:
            args: argument class instance
            dev: pytorch device
            system: system class instance
            model: model class instance
            xref: reference or equilibrium position, numpy array (D)
        """
        super(ModelBlack, self).__init__()

        self.device = dev
        self.sys = system

        # system parameters
        self.controlled_system = args.controlled_system
        self.lyapunov_correction = args.lyapunov_correction   

        # reference point
        self.Xref = xref.clone().detach().reshape(1,len(xref)).float().to(self.device) # (1,D)  
 
        # activation fcts.
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sp = nn.Softplus() #softplus (smooth approx. of ReLU)
        self.tanh = nn.Tanh()

        self.h_X = None
        self.h_zero = None

    @abstractmethod
    def forwardFNN(self, X):
        raise Exception(f"Function not implemented")

    @abstractmethod
    def forwardGNN(self, X):
        raise Exception(f"Function not implemented")

    @abstractmethod
    def forwardICNN(self, X):
        raise Exception(f"Function not implemented")

    def forward(self, X, U):
        """
        Forward pass through main model
        Args:
            X: state input batch (N x D)
            U: controll input batch (N X M)
        Returns:
            dX_opt: optimal approx. of state derivative
        """
        # FNN
        f_X = self.forwardFNN(X) # (N x D)

        # GNN
        g_X = None
        if self.controlled_system:
            g_X = self.forwardGNN(X) # (N x D x M)

        # f_opt is the best approx. of f_X that ensures lyapunov stability (N x D)
        f_opt = f_X
        if self.lyapunov_correction:
            # if np.count_nonzero(np.isnan(X.detach().numpy())) > 0:
            #     raise Exception("ERROR: at least one element of X is nan")

            V = self.forwardLyapunov(X) # (N)
            dV = self.gradient_lyapunov(X) # (N x D)
            f_opt = f_opt + self.fCorrection(f_X, g_X, V, dV)

        # dX_opt is the derivative of the state including control input u
        dX_opt = f_opt
        if self.controlled_system:
            dX_opt += torch.einsum('ndm,nm->nd', g_X, U)

        return dX_opt

    def forwardLyapunov(self, X):
        """
        Calc. lyapunov fct. used to correct f_X and ensure stability
        Args:
            X: state input batch (N x D)
        Returns:
            V: lyapunov fct. (N)
        """
        self.h_X = self.forwardICNN(X) # (N x 1)
        h_zero = self.forwardICNN(self.Xref) # (1 x 1) 
        self.h_zero = h_zero.tile(X.shape[0],1) # (N x 1)

        deltaX = X - self.Xref.tile(X.shape[0],1) # (N,D)
        V = self.activationLyapunov(self.h_X, self.h_zero) + self.epsilon*torch.einsum('nd,nd->n', deltaX, deltaX) # (N)
        return V

    def activationLyapunov(self, h_X, h_zero):
        """
        Calc. activation fct. of h(X)-h(0) st. V(x=0)=0 (enforce positive definitness)
        Args:
            h_X: output of ICNN with input X (N x 1)
            h_zero: output of ICNN with input 0 (N x 1)
        Returns:
            sigma_lyap: h(X)-h(0) after activation fct. (N)
        """
        h = torch.flatten(h_X) - torch.flatten(h_zero) # (N)
        
        return (h>=1)*(h-0.5) + (h>0)*(h<1)*(0.5*h*h)

    def gradient_lyapunov(self, X):
        """
        Calc. gradient of lyapunov fct. V
        Args:
            X: input batch (N x D)
        Returns:
            dV: gradient of lyapunov fct. V (N x D)
        """
        # The fct. jacobian from torch.autograde returns a squeezed 4 dimensional matrix where every output dimension is derived
        # by every input dimension [first ouput dimension, second ouput dimension, first input dimension, second input dimension].
        # We know that the ouput of each sample depends uniquly on its corresponding input and the derivative with respect to each
        # other sample will be equal to zero. Therefore, we can sum up all samples before calculating the jacobian. This prevents
        # to calculate a jacobian that is one dimension larger and diagonalizing it afterwards.
        dV = torch.autograd.functional.jacobian(lambda X: torch.sum(self.forwardLyapunov(X), axis=0), X, create_graph=True).squeeze(0)
        return dV

    def fCorrection(self, f_X, g_X, V, dV):
        """
        Calc. correction of f_X used to ensure stability
        Args:
            f_X: output of FCNN (N x D)
            g_X: output of GCNN (N x D x M)
            V: lyapunov fct. (N)
            dV: gradient of lyapunov fct. V (N x D)
        Returns:
            f_cor: forrection of f_X (N x D)
        """
        stability_conditions = torch.einsum('nd,nd->n', dV, f_X) + self.alpha*V # (N)
        if self.controlled_system:
            stability_conditions = stability_conditions - torch.sum(torch.abs(torch.einsum('nd,ndm->nm', dV, g_X)), dim=1)

        dV_norm = torch.einsum('nd,n->nd', dV, (1/torch.einsum('nd,nd->n', dV, dV))) # (N x D), normalize dV with squared L2-norm

        return -torch.einsum('nd,n->nd', dV_norm, self.relu(stability_conditions))

class HolohoverModelBlack(ModelBlack):
    def __init__(self, args, dev, system, xref):
        """
        Args:
            args: argument class instance
            dev: pytorch device
            system: system class instance
            model: model class instance
            xref: reference or equilibrium position, numpy array (D)
        """
        ModelBlack.__init__(self, args, dev, system, xref)

        # system parameters
        self.epsilon = 0.00001
        self.alpha = 0.05
        self.D = 6 # dim. of state x
        self.M = 6 # dim. of controll input u      

        # FNN: model parameters
        fnn_input_size = self.D
        fnn_output_size = self.D

        # GNN: model parameters
        gnn_input_size = self.D
        gnn_hidden1_size = 20
        gnn_hidden2_size = 80
        gnn_hidden3_size = 200
        gnn_hidden4_size = 200
        gnn_hidden5_size = 100
        gnn_hidden6_size = 50
        gnn_output_size = self.D*self.M # D*M (dim. of X times dim. of U)


        # FCNN: layers
        self.fnn_fc1 = nn.Linear(fnn_input_size, fnn_output_size, bias=False)

        # GCNN: layers
        self.gnn_fc1 = nn.Linear(gnn_input_size, gnn_hidden1_size, bias=True)
        self.gnn_fc2 = nn.Linear(gnn_hidden1_size, gnn_hidden2_size, bias=True)
        self.gnn_fc3 = nn.Linear(gnn_hidden2_size, gnn_hidden3_size, bias=True)
        self.gnn_fc4 = nn.Linear(gnn_hidden3_size, gnn_hidden4_size, bias=True)
        self.gnn_fc5 = nn.Linear(gnn_hidden4_size, gnn_hidden5_size, bias=True)
        self.gnn_fc6 = nn.Linear(gnn_hidden5_size, gnn_hidden6_size, bias=True)
        self.gnn_fc7 = nn.Linear(gnn_hidden6_size, gnn_output_size, bias=True)


    def forwardFNN(self, X):
        """
        Forward pass through FNN
        Args:
            X: state input batch (N x D)
        Returns:
            f_X: output of FCNN (N x D)
        """
        return self.fnn_fc1(X)

    def forwardGNN(self, X):
        """
        Forward pass through GNN
        Args:
            X: state input batch (N x D)
        Returns:
            g_X: output of GCNN (N x D x M)
        """
        g_X1 = self.relu( self.gnn_fc1(X) )
        g_X2 = self.relu( self.gnn_fc2(g_X1) )
        g_X3 = self.relu( self.gnn_fc3(g_X2) )
        g_X4 = self.relu( self.gnn_fc4(g_X3) )
        g_X5 = self.relu( self.gnn_fc5(g_X4) )
        g_X6 = self.relu( self.gnn_fc6(g_X5) )
        g_X7 = self.relu( self.gnn_fc7(g_X6) )

        return g_X7.reshape([X.shape[0], self.D, self.M])


class CSTRModelBlack(ModelBlack):
    def __init__(self, args, dev, system, xref):
        """
        Args:
            args: argument class instance
            dev: pytorch device
            system: system class instance
            model: model class instance
            xref: reference or equilibrium position, numpy array (D)
        """
        ModelBlack.__init__(self, args, dev, system, xref)

        # system parameters
        self.epsilon = 0.00001
        self.alpha = 0.05
        self.D = 2 # dim. of state x
        self.M = 1 # dim. of controll input u      

        # FNN: model parameters
        fnn_input_size = self.D
        fnn_hidden1_size = 80
        fnn_hidden2_size = 200
        fnn_output_size = self.D

        # GNN: model parameters
        gnn_input_size = self.D
        gnn_output_size = self.D*self.M # D*M (dim. of X times dim. of U)

        # ICNN: model parameters
        icnn_input_size = self.D
        icnn_hidden1_size = 60
        icnn_hidden2_size = 60
        icnn_hidden3_size = 30
        icnnn_output_size = 1

        # FCNN: layers
        self.fnn_fc1 = nn.Linear(fnn_input_size, fnn_hidden1_size, bias=True)
        self.fnn_fc2 = nn.Linear(fnn_hidden1_size, fnn_hidden2_size, bias=True)
        self.fnn_fc3 = nn.Linear(fnn_hidden2_size, fnn_output_size, bias=True)

        # GCNN: layers
        self.gnn_fc1 = nn.Linear(gnn_input_size, gnn_output_size, bias=True)

        # ICNN: fully connected layers and input mapping
        self.icnn_fc1 = nn.Linear(icnn_input_size, icnn_hidden1_size, bias=True)
        self.icnn_fc2 = nn.Linear(icnn_hidden1_size, icnn_hidden2_size, bias=True)
        self.icnn_fc3 = nn.Linear(icnn_hidden2_size, icnn_hidden3_size, bias=True)
        self.icnn_fc4 = nn.Linear(icnn_hidden3_size, icnnn_output_size, bias=True)
        self.icnn_im2 = nn.Linear(icnn_input_size, icnn_hidden2_size, bias=False)
        self.icnn_im3 = nn.Linear(icnn_input_size, icnn_hidden3_size, bias=False)
        self.icnn_im4 = nn.Linear(icnn_input_size, icnnn_output_size, bias=False)

    def forwardFNN(self, X):
        """
        Forward pass through FNN
        Args:
            X: state input batch (N x D)
        Returns:
            f_X: output of FCNN (N x D)
        """
        x_fnn_fc1 = self.fnn_fc1(X)
        x_fnn_tanh1 = self.tanh(x_fnn_fc1)

        x_fnn_fc2 = self.fnn_fc2(x_fnn_tanh1)
        x_fnn_tanh2 = self.tanh(x_fnn_fc2)

        f_X = self.fnn_fc3(x_fnn_tanh2)
        return f_X

    def forwardGNN(self, X):
        """
        Forward pass through GNN
        Args:
            X: state input batch (N x D)
        Returns:
            g_X: output of GCNN (N x D x M)
        """
        g_X = self.gnn_fc1(X)
        return g_X.reshape([X.shape[0], self.D, self.M])

    def forwardICNN(self, X):
        """
        Pass through ICNN (input convex neural network)
        Args:
            X: state input batch (N x D)
        Returns:
            h_X: output of ICNN (N x 1)
        """
        x_icnn_fc1 = self.icnn_fc1(X)
        x_icnn_sp1 = self.sp(x_icnn_fc1)

        x_icnn_fc2 = self.icnn_fc2(x_icnn_sp1)
        x_icnn_im2 = self.icnn_im2(X)
        x_icnn_sp2 = self.sp(x_icnn_fc2 + x_icnn_im2)

        x_icnn_fc3 = self.icnn_fc3(x_icnn_sp2)
        x_icnn_im3 = self.icnn_im3(X)
        x_icnn_sp3 = self.sp(x_icnn_fc3 + x_icnn_im3)

        x_icnn_fc4 = self.icnn_fc4(x_icnn_sp3)
        x_icnn_im4 = self.icnn_im4(X)
        h_X = self.sp(x_icnn_fc4 + x_icnn_im4)
        return h_X


class DHOModelBlack(ModelBlack):
    def __init__(self, args, dev, system, xref):
        """
        Args:
            args: argument class instance
            dev: pytorch device
            system: system class instance
            model: model class instance
            xref: reference or equilibrium position, numpy array (D)
        """
        ModelBlack.__init__(self, args, dev, system, xref)
        # system parameters
        self.epsilon = 0.01
        self.alpha = 0.1
        self.D = 2 # dim. of state x
        self.M = 1 # dim. of controll input u

        # FNN: model parameters
        fnn_input_size = self.D
        fnn_output_size = self.D

        # GNN: model parameters
        gnn_input_size = self.D
        gnn_output_size = self.D*self.M # D*M (dim. of x times dim. of u)

        # ICNN: model parameters
        icnn_input_size = self.D
        icnn_hidden1_size = 60
        icnn_hidden2_size = 60
        icnnn_output_size = 1

        # FCNN: layers
        self.fnn_fc1 = nn.Linear(fnn_input_size, fnn_output_size, bias=False)

        # GCNN: layers
        self.gnn_fc1 = nn.Linear(gnn_input_size, gnn_output_size, bias=True)

        # ICNN: fully connected layers
        self.icnn_fc1 = nn.Linear(icnn_input_size, icnn_hidden1_size, bias=True)
        self.icnn_fc2 = nn.Linear(icnn_hidden1_size, icnn_hidden2_size, bias=True)
        self.icnn_fc3 = nn.Linear(icnn_hidden2_size, icnnn_output_size, bias=True)

        # ICNN: input mapping
        self.icnn_im2 = nn.Linear(icnn_input_size, icnn_hidden2_size, bias=False)
        self.icnn_im3 = nn.Linear(icnn_input_size, icnnn_output_size, bias=False)

    def forwardFNN(self, X):
        """
        Forward pass through FNN
        Args:
            X: state input batch (N x D)
        Returns:
            f_X: output of FCNN (N x D)
        """
        f_X = self.fnn_fc1(X)
        return f_X

    def forwardGNN(self, X):
        """
        Forward pass through GNN
        Args:
            X: state input batch (N x D)
        Returns:
            g_X: output of GCNN (N x D x M)
        """
        g_X = self.gnn_fc1(X) 
        return g_X.reshape([X.shape[0], self.D, self.M])

    def forwardICNN(self, X):
        """
        Pass through ICNN (input convex neural network)
        Args:
            X: state input batch (N x D)
        Returns:
            h_X: output of ICNN (N x 1)
        """
        x_icnn_fc1 = self.icnn_fc1(X)
        x_icnn_sp1 = self.sp(x_icnn_fc1)

        x_icnn_fc2 = self.icnn_fc2(x_icnn_sp1)
        x_icnn_im2 = self.icnn_im2(X)
        x_icnn_sp2 = self.sp(x_icnn_fc2 + x_icnn_im2)

        x_icnn_fc3 = self.icnn_fc3(x_icnn_sp2)
        x_icnn_im3 = self.icnn_im3(X)
        h_X = self.sp(x_icnn_fc3 + x_icnn_im3)
        return h_X