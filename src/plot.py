import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches

from src.model_grey import HolohoverModelGrey, CorrectModelGrey

class Plot():
    def __init__(self, args, params, dev, model, cor_model, system, learn, learn_cor) -> None:
        self.args = args
        self.params = params
        self.model = model
        self.cor_model = cor_model
        self.sys = system
        self.learn = learn
        self.learn_cor = learn_cor
        self.device = dev

        if args.model_type == "DHO":
            self.quiver_scale = 50.0
        elif args.model_type == "CSTR":
            self.quiver_scale = 0.8
        elif args.model_type == "HolohoverGrey":
            self.quiver_scale = 1

        self.frequency = 240
        self.plot_range = [int(5*self.frequency), int(8*self.frequency)]

    def greyModel(self, u_eq):
        # define control input, u_hat is bounded by [-1,1]
        Ueq = u_eq.reshape((1,self.sys.M))
        Umax = torch.ones((1,self.sys.M)) 

        xmin = self.sys.x_min - (self.sys.x_max-self.sys.x_min)/4
        xmax = self.sys.x_max + (self.sys.x_max-self.sys.x_min)/4

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(10, 10))

        self.modelLoss(axs[0,0], losses_tr=self.learn.metrics["losses_tr"], losses_te=self.learn.metrics["losses_te"])
        self.modelError((axs[1,0], axs[2,0]), abs_error=self.learn.metrics["abs_error"], rms_error=self.learn.metrics["rms_error"])
        self.modelData((axs[0,1],axs[1,1],axs[2,1]), plot_white=True)

        plt.savefig(os.path.join(self.args.dir_path, "learn_model.pdf")) 

    def corModel(self, u_eq=False):

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(10, 10))

        self.modelLoss(axs[0,0], losses_tr=self.learn_cor.metrics["losses_tr"], losses_te=self.learn_cor.metrics["losses_te"])
        self.modelError((axs[1,0], axs[2,0]), abs_error=self.learn_cor.metrics["abs_error"], rms_error=self.learn.metrics["rms_error"])
        self.modelData((axs[0,1],axs[1,1],axs[2,1]), plot_cor=True)

        plt.savefig(os.path.join(self.args.dir_path, "learn_correction.pdf"))   


    def modelData(self, axs, plot_cor=False, plot_white=False):
        X_data, U_data, dX_data = self.sys.getData()
        time = np.arange(X_data.shape[0]) / self.frequency

        X = X_data[self.plot_range[0]:self.plot_range[1],:]
        U = U_data[self.plot_range[0]:self.plot_range[1],:]
        dX_real = dX_data[self.plot_range[0]:self.plot_range[1],:]
        time = time[self.plot_range[0]:self.plot_range[1]]

        with torch.no_grad():
            dX_model = self.learn.forward(X, U)

            if plot_cor:
               dX_cor = self.learn_cor.forward(X, U)

            if plot_white:
                white_model = HolohoverModelGrey(args=self.args, params=self.params, dev="cpu")
                dX_white = white_model.forward(X=X, U=U)

        axs[0].set_title(f"Finale dd(x)")
        axs[0].set_ylabel('[m/s^2]')
        axs[0].plot(time, dX_real[:,3], label="real", color="black")
        if plot_white:
            axs[0].plot(time, dX_white[:,3], label="white box", color="blue")
            axs[0].plot(time, dX_model[:,3], "--", label="grey box", color="cyan")
        if plot_cor:
            axs[0].plot(time, dX_model[:,3], label="grey box", color="cyan")
            axs[0].plot(time, dX_cor[:,3], "--", label="grey box corr.", color="orange")
        axs[0].legend()

        axs[1].set_title(f"Finale dd(y)")
        axs[1].set_ylabel('[m/s^2]')
        axs[1].plot(time, dX_real[:,4], label="real", color="black")
        if plot_white:
            axs[1].plot(time, dX_white[:,4], label="white box", color="blue")
            axs[1].plot(time, dX_model[:,4], "--", label="grey box", color="cyan")
        if plot_cor:
            axs[1].plot(time, dX_model[:,4], label="grey box", color="cyan")
            axs[1].plot(time, dX_cor[:,4], "--", label="grey box corr.", color="orange")
        axs[1].legend()

        axs[2].set_title(f"Finale dd(theta)")
        axs[2].set_xlabel('time [s]')
        axs[2].set_ylabel('[rad/s^2]')
        axs[2].plot(time, dX_real[:,5], label="real", color="black")
        if plot_white:
            axs[2].plot(time, dX_white[:,5], label="white box", color="blue")
            axs[2].plot(time, dX_model[:,5], "--", label="grey box", color="cyan")
        if plot_cor:
            axs[2].plot(time, dX_model[:,5], label="grey box", color="cyan")
            axs[2].plot(time, dX_cor[:,5], "--", label="grey box corr.", color="orange")
        axs[2].legend()



    def modelError(self, axs, abs_error, rms_error):
        abs_error = np.array(abs_error)
        rms_error = np.array(rms_error)

        axs[0].set_title(f"Error")
        axs[0].set_ylabel('[m/s^2]')
        axs[0].plot(abs_error[:,3], label="abs dd(x)", color="red")
        axs[0].plot(rms_error[:,3], "--", label="rms dd(x)", color="red")
        axs[0].plot(abs_error[:,4], label="abs dd(y)", color="orange")
        axs[0].plot(rms_error[:,4], "--", label="rms dd(y)", color="orange")
        axs[0].legend()

        axs[1].set_title(f"Error")
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('[rad/s^2]')
        axs[1].plot(abs_error[:,5], label=" abs dd(theta)", color="gold")
        axs[1].plot(rms_error[:,5], "--", label="rms dd(theta)", color="gold")
        axs[1].legend()


           

    def fakeModel(self, u_eq):

        # define control input, u_hat is bounded by [-1,1]
        Ueq = u_eq.reshape((1,self.sys.M))
        Umax = torch.ones((1,self.sys.M)) 

        xmin = self.sys.x_min - (self.sys.x_max-self.sys.x_min)/4
        xmax = self.sys.x_max + (self.sys.x_max-self.sys.x_min)/4

        fig, axs = plt.subplots(nrows=4, ncols=2, figsize =(10, 18))

        self.modelLoss(axs[0,0])
        self.modelLoss(axs[0,1], log_scale=True)
        if self.model.lyapunov_correction:
            self.modelLyap(axs[1,0], xmin, xmax, add_title=True)
            self.modelCorr(axs[1,1], xmin, xmax)

        self.modelRealDyn(axs[2,0], xmin, xmax, Ueq)
        self.modelLearnedDyn(axs[2,1], xmin, xmax, Ueq)
        if self.model.controlled_system:
            self.modelRealDyn(axs[3,0], xmin, xmax, Umax)
            self.modelLearnedDyn(axs[3,1], xmin, xmax, Umax)

        plt.savefig(os.path.join(self.learn.model_dir, self.learn.model_name + "_figure"))

    def modelLoss(self, axis, losses_tr, losses_te, log_scale=False):     
        axis.set_title(f"Loss")
        axis.set_ylabel('loss')
        axis.plot(losses_te, label="testing", color="purple")
        axis.plot(losses_tr, "--", label="training", color="purple")
        axis.legend()

        if log_scale:
            axis.set_yscale("log")

    # def modelCorrection(self):
    #     X, U, dX_real = self.sys.getData()

    #     with torch.no_grad():
    #         acc_cor = self.learn_cor.model.forward(X=X, U=U)

    #     nb_bins = 15
    #     X_bins = np.zeros((X.shape[0],3))
    #     for i in range(3,6):
    #         X_bins[:,i] = np.digitize(x=X[:,i], bins=np.linspace(start=np.min(X[:,i]), stop=np.max(X[:,i]), num=nb_bins))

    #     X_bin_means = np.zeros((nb_bins,3))
    #     for bin in range(nb_bins):


    def paramsSig2Thrust(self):

        U = torch.linspace(0, 1, steps=100).reshape(100,1).repeat(1,6)
        U = self.sys.polyExpandU(U)
        with torch.no_grad():
            thrust_learned = self.model.signal2thrust(U=U.detach().clone())
        thrust_learned = thrust_learned.detach().numpy()

        white_model = HolohoverModelGrey(args=self.args, params=self.params, dev="cpu")
        with torch.no_grad():
            thrust_init = white_model.signal2thrust(U=U.detach().clone())
        thrust_init = thrust_init.detach().numpy()

        U = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize =(6, 5))
        colors = ["royalblue", "orange", "green", "red", "darkorchid", "brown"]
        for i in range(thrust_learned.shape[1]):           
            ax.plot(U, thrust_learned[:,i], color=colors[i], label=f"Motor {i+1}")
            ax.plot(U, thrust_init[:,i], "--", color=colors[i])
        ax.set_ylabel("Thrust [N]")
        ax.set_xlabel("Signal")
        ax.legend()

        plt.savefig(os.path.join(self.args.dir_path, "signal2thrust_params.pdf")) 

    def paramsVec(self):
        pos_learned = self.model.motors_pos.detach().numpy()
        vec_learned = self.model.motors_vec.detach().numpy()

        white_model = HolohoverModelGrey(args=self.args, params=self.params, dev="cpu")
        pos_init = white_model.motors_pos.detach().numpy()
        vec_init = white_model.motors_vec.detach().numpy()

        com_learned = self.model.center_of_mass.detach().numpy()
        com_init = white_model.center_of_mass.detach().numpy()
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize =(6, 5))
        circle = plt.Circle((0, 0), 0.06, color='black', alpha=0.2, label="Holohover")
        ax.add_patch(circle)

        scaling_vec = 0.008
        vec_learned = vec_learned*scaling_vec
        vec_init = vec_init*scaling_vec
        for i in range(pos_init.shape[0]): 
            arrow =  mpatches.FancyArrow(pos_init[i,0], pos_init[i,1], vec_init[i,0], vec_init[i,1], 
                                        color="blue", length_includes_head=True)
            ax.add_patch(arrow)
            arrow =  mpatches.FancyArrow(pos_learned[i,0], pos_learned[i,1], vec_learned[i,0], vec_learned[i,1], 
                                        color="green", length_includes_head=True)
            ax.add_patch(arrow)   
        
        ax.scatter(com_init[0], com_init[1], color="blue", label=f"Before learning")
        ax.scatter(com_learned[0], com_learned[1], color="green", label=f"After learning")
        ax.legend()
        ax.set_aspect('equal', 'box')
        ax.set_ylabel("position [m]")
        ax.set_xlabel("position [m]")
        ax.set_xlim([-0.1,0.12])
        ax.set_ylim([-0.1,0.12])
        ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
        ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
        plt.savefig(os.path.join(self.args.dir_path, "pos_vec_params.pdf"))

    def dataHistogram(self):
        X_data, U_data, dX_data = self.sys.getData()
        X = X_data.detach().numpy()
        U = U_data.detach().numpy()
        dX = dX_data.detach().numpy()

        U = np.concatenate([U[:,0]+U[:,1], U[:,2]+U[:,3], U[:,4]+U[:,5]])

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(8, 7))
        fig.suptitle(f"Total nb. samples: {X.shape[0]}")
        axs[0,0].hist(U, bins=100, label="signal")
        axs[1,0].hist(dX[:,3], bins=100, label="dd(x)")
        axs[1,1].hist(dX[:,4], bins=100, label="dd(y)")
        axs[0,1].hist(dX[:,5], bins=100, label="dd(theta)")
        axs[0,0].set_xlabel("signal")
        axs[1,0].set_xlabel("[m/s^2]")
        axs[1,1].set_xlabel("[m/s^2]")
        axs[0,1].set_xlabel("[rad/s^2]")
        axs[0,0].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        axs[0,1].legend()
        plt.savefig(os.path.join(self.args.dir_path, "histogram.pdf"))

    def blackDHO(self):
        xmin = (-1, -0.75)
        xmax = (2, 1.0)

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(8, 8))
        self.modelLoss(axs[0,0], losses_tr=self.learn.metrics["losses_tr"], losses_te=self.learn.metrics["losses_te"])
        self.modelLyap(axs[0,1], xmin=xmin, xmax=xmax, add_title=True, add_xlabel=False, add_ylabel=False)
        self.modelRealDyn(axs[1,0], xmin=xmin, xmax=xmax, U=0, U_hat=False, add_xlabel=False, add_ylabel=True)
        self.modelRealDyn(axs[2,0], xmin=xmin, xmax=xmax, U=-0.5, U_hat=False, add_xlabel=True, add_ylabel=True)
        self.modelLearnedDyn(axs[1,1], xmin=xmin, xmax=xmax, U=0, dim=0, add_xlabel=False, add_ylabel=False)
        self.modelLearnedDyn(axs[2,1], xmin=xmin, xmax=xmax, U=-0.5, dim=0, add_xlabel=True, add_ylabel=False)
        plt.savefig(os.path.join(self.args.dir_path, "learned_dynamics.pdf"))

    def blackCSTR(self):
        xmin = (0.5, 0.0)
        xmax = (3.5, 2.5)
        U0 = self.sys.uMap(torch.tensor([35]).reshape(1,1))[0,0]
        Ueq = self.sys.uMap(torch.tensor([14.19]).reshape(1,1))[0,0]

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(8, 10))
        self.modelLoss(axs[0,0], losses_tr=self.learn.metrics["losses_tr"], losses_te=self.learn.metrics["losses_te"])
        self.modelLyap(axs[0,1], xmin=xmin, xmax=xmax, add_title=True, add_xlabel=False, add_ylabel=False)
        self.modelRealDyn(axs[1,0], xmin=xmin, xmax=xmax, U=Ueq, U_hat=True, add_xlabel=False, add_ylabel=True)
        self.modelRealDyn(axs[2,0], xmin=xmin, xmax=xmax, U=U0, U_hat=True, add_xlabel=True, add_ylabel=True)
        self.modelLearnedDyn(axs[1,1], xmin=xmin, xmax=xmax, U=Ueq, dim=0, add_xlabel=False, add_ylabel=False)
        self.modelLearnedDyn(axs[2,1], xmin=xmin, xmax=xmax, U=U0, dim=0, add_xlabel=True, add_ylabel=False)
        plt.savefig(os.path.join(self.args.dir_path, "learned_dynamics.pdf"))


    def modelGenX(self, xmin, xmax):
        # define range of plot
        x0_range = torch.linspace(xmin[0], xmax[0], 20).to(self.device)
        x1_range = torch.linspace(xmin[1], xmax[1], 20).to(self.device)

        # create equal distributed state vectors
        x0_vector = x0_range.tile((x1_range.size(0),))
        x1_vector = x1_range.repeat_interleave(x0_range.size(0))
        X = torch.zeros(x0_vector.size(0),self.sys.D).to(self.device)
        X[:,0] = x0_vector
        X[:,1] = x1_vector

        return X.detach().clone(), x0_range, x1_range

    def modelLyap(self, axis, xmin, xmax, add_title=True, add_xlabel=True, add_ylabel=True):

        X, x0_range, x1_range = self.modelGenX(xmin, xmax)

        # calc. Lyapunov fct. and lyapunov correction f_cor
        V = self.model.forwardLyapunov(X) # (N)
        V = V.detach().numpy()        

        X_contour, Y_contour = np.meshgrid(x0_range, x1_range)
        Z_contour = np.array(V.reshape(x1_range.size(0),x0_range.size(0)))

        contours = axis.contour(X_contour, Y_contour, Z_contour)
        axis.clabel(contours, inline=1, fontsize=10)
        axis.plot(self.model.Xref[0,0], self.model.Xref[0,1], marker=(5, 1), markeredgecolor="red", markerfacecolor="red")   
        axis.set_aspect('equal')
        axis.legend()
        if add_xlabel:
            axis.set_xlabel('x[0]')
        if add_ylabel:
            axis.set_ylabel('x[1]')
        if add_title:
            axis.set_title('Lyapunov fct. (V)')

    def modelCorr(self, axis, xmin, xmax, add_xlabel=True, add_ylabel=True):

        X, _, _ = self.modelGenX(xmin, xmax)

        # calc. Lyapunov fct. and lyapunov correction f_cor
        f_X = self.model.forwardFNN(X)
        g_X = self.model.forwardGNN(X) # (N x D x M)
        V = self.model.forwardLyapunov(X) # (N)
        dV = self.model.gradient_lyapunov(X) # (N x D)
        f_cor = self.model.fCorrection(f_X, g_X, V, dV)

        # convert tensors to numpy arrays
        f_X = f_X.detach().numpy()
        f_cor = f_cor.detach().numpy()
        V = V.detach().numpy()

        axis.set_title('Dynamics correction by Lyapunov fct.')
        axis.set_xlabel('x[0]')
        axis.set_ylabel('x[1]')
        axis.quiver(X[:,0], X[:,1], f_X[:,0], f_X[:,1], color="b", scale=self.quiver_scale)
        axis.quiver(X[:,0], X[:,1], f_cor[:,0], f_cor[:,1], color="r", scale=self.quiver_scale)
        axis.set_aspect('equal')
        if add_xlabel:
            axis.set_xlabel('x[0]')
        if add_ylabel:
            axis.set_ylabel('x[1]')

    def modelRealDyn(self, axis, xmin, xmax, U, U_hat, add_xlabel=True, add_ylabel=True):

        X, _, _ = self.modelGenX(xmin, xmax)
        U = torch.tensor([U]).repeat(X.shape[0],1)

        # calc. real system dynamics
        dX_real = self.sys.calcDX(X=X.cpu(), U=U.cpu(), U_hat=U_hat)

        # calc. equilibrium point for U
        x_eq = self.sys.equPoint(U[0], U_hat=U_hat)
        x_eq = x_eq.reshape(self.sys.D)

        axis.set_title('Real dynamics (U='+str(self.sys.uMapInv(U[0,:]).numpy())+')')
        axis.quiver(X[:,0], X[:,1], dX_real[:,0], dX_real[:,1], scale=self.quiver_scale)
        axis.set_aspect('equal')
        if (x_eq[0]>X[0,0] and x_eq[0]<X[-1,0]) and (x_eq[1]>X[0,1] and x_eq[1]<X[-1,1]):
            axis.plot(x_eq[0], x_eq[1], marker=(5, 1), color="white", markeredgecolor="red", markerfacecolor="red", label="Equilibrium point")
        rect_training = patches.Rectangle((self.sys.x_min[0],self.sys.x_min[1]), width=(self.sys.x_max[0]-self.sys.x_min[0]), \
                                            height=(self.sys.x_max[1]-self.sys.x_min[1]), facecolor='none', edgecolor="g")     
        axis.add_patch(rect_training)
        axis.legend()
        if add_xlabel:
            axis.set_xlabel('x[0]')
        if add_ylabel:
            axis.set_ylabel('x[1]')

    def modelLearnedDyn(self, axis, xmin, xmax, U, dim=0, add_xlabel=True, add_ylabel=True):

        X, _, _ = self.modelGenX(xmin, xmax)
        U = torch.tensor([U]).repeat(X.shape[0],1)

        # calc. learned system dynamics
        with torch.no_grad():
            dX = self.model.forward(X, U)      

        axis.set_title('Learned dynamics (U='+str(self.sys.uMapInv(U[0,:]).numpy())+')')
        axis.quiver(X[:,dim], X[:,dim+1], dX[:,dim], dX[:,dim+1], scale=self.quiver_scale)
        axis.set_aspect('equal')
        rect_training = patches.Rectangle((self.sys.x_min[dim],self.sys.x_min[dim+1]), width=(self.sys.x_max[dim]-self.sys.x_min[dim]), \
                                            height=(self.sys.x_max[dim+1]-self.sys.x_min[dim+1]), facecolor='none', edgecolor="g", \
                                            label="Region of learning data")        
        axis.add_patch(rect_training)
        axis.legend()
        if add_xlabel:
            axis.set_xlabel('x[0]')
        if add_ylabel:
            axis.set_ylabel('x[1]')

    def modelApproxDyn(self, axis, U, dim=0):
        # approximate dynamics by sampling data in a certain region
        dX = self.sys.sampleX(Udes=U, U_hat=True)

        axis.set_title('Real dynamics (U='+str(self.sys.uMapInv(U[0,:]).numpy())+')')
        axis.set_xlabel('x[0]')
        axis.set_ylabel('x[1]')
        axis.plot(dX[:,dim], dX[:,dim+self.sys.S])


        

    def sim(self, X_seq_on, X_seq_off, Udes_seq, Usafe_seq_on, slack_seq_on, V_seq_on):
        """
        Plot simulation
        Args:
            X_seq_on: sequence of states when safety filter is on, numpy array (nb_steps+1,D)
            X_seq_off: sequence of states when safety filter is off, numpy array (nb_steps+1,D)
            Udes_seq: desired control input sequence (nb_steps,M)
            Usafe_seq: resulting control input sequence (nb_steps,M)
            slack_seq_on: resulting slacks used in optimization when safety filter is on (nb_steps)
        """
        nb_steps = Udes_seq.shape[0]

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(9, 9))

        xmin = np.minimum(np.min(X_seq_off, axis=0), np.min(X_seq_on, axis=0))
        xmax = np.maximum(np.max(X_seq_off, axis=0), np.max(X_seq_on, axis=0))
        xmin -= (xmax-xmin)/6
        xmax += (xmax-xmin)/6
        
        self.simControlInput(axs[0,0], Udes_seq, Usafe_seq_on)
        self.simSlack(axs[0,1], slack_seq_on)        
        self.simTrajectory(axs[1,0], X_seq_off, nb_steps, xmin, xmax, filter_on=False)
        self.simTrajectory(axs[1,1], X_seq_on, nb_steps, xmin, xmax, filter_on=True)
        self.modelLyap(axs[1,1], xmin, xmax, add_title=False)

        axs[1,2].set_title(f"V(x) on trajectory")
        axs[1,2].set_xlabel('V(x)')
        axs[1,2].set_ylabel('timestep')
        axs[1,2].plot(V_seq_on)


        plt.show()

    def simGrey(self, Xreal_seq, Xreal_integ_seq, Xlearn_seq):
        """
        Plot simulation
        Args:
            
        """
        nb_steps = Xreal_seq.shape[0]
        xmin = np.minimum(np.min(Xreal_seq, axis=0), np.min(Xlearn_seq, axis=0), np.min(Xreal_integ_seq, axis=0))
        xmax = np.maximum(np.max(Xreal_seq, axis=0), np.max(Xlearn_seq, axis=0), np.max(Xreal_integ_seq, axis=0))
        xmin -= (xmax-xmin)/6
        xmax += (xmax-xmin)/6

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(9, 9))

        self.modelLoss(axs[0,0])               
        self.simTrajectory(axs[0,1], Xreal_seq, nb_steps-1, xmin, xmax, title="Real trajectory")
        self.simTrajectory(axs[1,0], Xreal_integ_seq, nb_steps-1, xmin, xmax, title="Real integrated trajectory")
        self.simTrajectory(axs[1,1], Xlearn_seq, nb_steps-1, xmin, xmax, title="Learned trajectory")

        plt.savefig(os.path.join(self.learn.model_dir, self.learn.model_name + "_figure_sim"))

    def simTrajectory(self, axis, X_seq, nb_steps, xmin, xmax, title=False):
        """
        Plot state sequence
        Args:
            axis: matplotlib axis to create plot
            nb_steps: number of simulation steps
            X_seq: sequence of states, numpy array (nb_steps,D)
            xmin: min. value for range, numpy array (D)
            xmax: max. value for range, numpy array (D)
            filter_on: if True then the safety fiter was on
        """
        color = plt.cm.rainbow(np.linspace(0, 1, nb_steps))
        for i in range(nb_steps):
            if not i%(nb_steps/5):
                axis.plot(X_seq[i+1,0], X_seq[i+1,1], 'o', color=color[i], label="t = "+str(i))
            else:
                axis.plot(X_seq[i+1,0], X_seq[i+1,1], 'o', color=color[i])  
          
        axis.legend()
        axis.set_xlim([xmin[0], xmax[0]])
        axis.set_ylim([xmin[1], xmax[1]])
        axis.set_xlabel('x[0]')
        axis.set_ylabel('x[1]')
        if title:
            axis.set_title(title)
        else:
            axis.set_title(f"Trajectory")

    def simControlInput(self, axis, Udes_seq, Usafe_seq):
        """
        Plot sequence of control inputs
        Returns:
            axis: matplotlib axis to create plot
            Udes_seq: desired control input sequence (nb_steps,M)
            Usafe_seq: resulting control input sequence (nb_steps,M)
        """
        x_axis = np.arange(Udes_seq.shape[0])
        axis.plot(x_axis, Udes_seq, color="b", label="U desired")
        axis.plot(x_axis, Usafe_seq, color="g", label="U safe")
        axis.legend()
        axis.set_xlabel('Time step')
        axis.set_ylabel('control input')
        axis.set_title(f"Control sequence")
        
    def simSlack(self, axis, slack_seq):
        """
        Plot sequence of control inputs
        Returns:
            axis: matplotlib axis to create plot
            slack_seq: resulting slacks used in optimization (nb_steps)
        """
        x_axis = np.arange(len(slack_seq))
        axis.plot(x_axis, slack_seq, linestyle='dashed', color="r", label="Sum of slack vector")
        axis.legend()
        axis.set_xlabel('Time step')
        axis.set_title(f"Optimization slack")
        
    