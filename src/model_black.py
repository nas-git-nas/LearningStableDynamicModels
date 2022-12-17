from abc import abstractmethod
import torch
import torch.nn as nn

class ModelBlack(nn.Module):
    def __init__(self, args, dev, generator, xref):
        super(ModelBlack, self).__init__()

        self.device = dev
        self.sys = generator

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
        pass

    @abstractmethod
    def forwardGNN(self, X):
        pass

    @abstractmethod
    def forwardICNN(self, X):
        pass

    def forward(self, X, U):
        """
        Description: forward pass through main model
        In X: state input batch (N x D)
        In U: controll input batch (N X M)
        Out dX_opt: optimal approx. of state derivative
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
        Description: calc. lyapunov fct. used to correct f_X and ensure stability
        In X: state input batch (N x D)
        Out V: lyapunov fct. (N)
        """
        self.h_X = self.forwardICNN(X) # (N x 1)
        h_zero = self.forwardICNN(self.Xref) # (1 x 1) 
        self.h_zero = h_zero.tile(X.shape[0],1) # (N x 1)

        deltaX = X - self.Xref.tile(X.shape[0],1) # (N,D)
        V = self.activationLyapunov(self.h_X, self.h_zero) + self.epsilon*torch.einsum('nd,nd->n', deltaX, deltaX) # (N)
        return V

    def activationLyapunov(self, h_X, h_zero):
        """
        Description: calc. activation fct. of h(X)-h(0) st. V(x=0)=0 (enforce positive definitness)
        In h_X: output of ICNN with input X (N x 1)
        In h_zero: output of ICNN with input 0 (N x 1)
        Out sigma_lyap: h(X)-h(0) after activation fct. (N)
        """
        h = torch.flatten(h_X) - torch.flatten(h_zero) # (N)
        
        return (h>=1)*(h-0.5) + (h>0)*(h<1)*(0.5*h*h)

    def gradient_lyapunov(self, X):
        """
        Description: calc. gradient of lyapunov fct. V
        In X: input batch (N x D)
        Out dV: gradient of lyapunov fct. V (N x D)
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
        Description: calc. correction of f_X used to ensure stability
        In f_X: output of FCNN (N x D)
        In g_X: output of GCNN (N x D x M)
        In V: lyapunov fct. (N)
        In dV: gradient of lyapunov fct. V (N x D)
        Out f_cor: forrection of f_X (N x D)
        """
        stability_conditions = torch.einsum('nd,nd->n', dV, f_X) + self.alpha*V # (N)
        if self.controlled_system:
            stability_conditions = stability_conditions - torch.sum(torch.abs(torch.einsum('nd,ndm->nm', dV, g_X)), dim=1)

        dV_norm = torch.einsum('nd,n->nd', dV, (1/torch.einsum('nd,nd->n', dV, dV))) # (N x D), normalize dV with squared L2-norm

        return -torch.einsum('nd,n->nd', dV_norm, self.relu(stability_conditions))

class HolohoverModelBlack(ModelBlack):
    def __init__(self, args, dev, generator, xref):
        ModelBlack.__init__(self, args, dev, generator, xref)

        # system parameters
        self.epsilon = 0.00001
        self.alpha = 0.05
        self.D = 6 # dim. of state x
        self.M = 6 # dim. of controll input u      

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


class CSTRModelBlack(ModelBlack):
    def __init__(self, args, dev, generator, xref):
        ModelBlack.__init__(self, args, dev, generator, xref)

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
        Description: forward pass through FNN
        In X: state input batch (N x D)
        Out f_X: output of FCNN (N x D)
        """
        x_fnn_fc1 = self.fnn_fc1(X)
        x_fnn_tanh1 = self.tanh(x_fnn_fc1)

        x_fnn_fc2 = self.fnn_fc2(x_fnn_tanh1)
        x_fnn_tanh2 = self.tanh(x_fnn_fc2)

        f_X = self.fnn_fc3(x_fnn_tanh2)
        return f_X

    def forwardGNN(self, X):
        """
        Description: forward pass through GNN
        In X: state input batch (N x D)
        Out g_X: output of GCNN (N x D x M)
        """
        g_X = self.gnn_fc1(X)
        return g_X.reshape([X.shape[0], self.D, self.M])

    def forwardICNN(self, X):
        """
        Description: pass through ICNN (input convex neural network)
        In X: state input batch (N x D)
        Out h_X: output of ICNN (N x 1)
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
    def __init__(self, args, dev, generator, xref):
        ModelBlack.__init__(self, args, dev, generator, xref)
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
        Description: forward pass through FNN
        In X: state input batch (N x D)
        Out f_X: output of FCNN (N x D)
        """
        f_X = self.fnn_fc1(X)
        return f_X

    def forwardGNN(self, X):
        """
        Description: forward pass through GNN
        In X: state input batch (N x D)
        Out g_X: output of GCNN (N x D x M)
        """
        g_X = self.gnn_fc1(X) 
        return g_X.reshape([X.shape[0], self.D, self.M])

    def forwardICNN(self, X):
        """
        Description: pass through ICNN (input convex neural network)
        In X: state input batch (N x D)
        Out h_X: output of ICNN (N x 1)
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