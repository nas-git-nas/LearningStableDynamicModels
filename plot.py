import numpy as np
import torch
import matplotlib.pyplot as plt

class Plot():
    def __init__(self, model, system, dev) -> None:
        self.model = model
        self.sys = system
        self.device = dev


    def modelLyap(self, axis, xmin, xmax, add_title=True):

        # define range of plot
        x0_range = torch.linspace(xmin[0], xmax[0], 100).to(self.device)
        x1_range = torch.linspace(xmin[1], xmax[1], 100).to(self.device)

        # create equal distributed state vectors
        x0_vector = x0_range.tile((x1_range.size(0),))
        x1_vector = x1_range.repeat_interleave(x0_range.size(0))
        X = torch.zeros(x0_vector.size(0),self.sys.D).to(self.device)
        X[:,0] = x0_vector
        X[:,1] = x1_vector

        # calc. Lyapunov fct. and lyapunov correction f_cor
        V = self.model.forwardLyapunov(X) # (N)
        V = V.detach().numpy()        

        X_contour, Y_contour = np.meshgrid(x0_range, x1_range)
        Z_contour = np.array(V.reshape(x1_range.size(0),x0_range.size(0)))

        contours = axis.contour(X_contour, Y_contour, Z_contour, colors='black', label="Lyapunov fct.")
        axis.clabel(contours, inline=1, fontsize=10)
        axis.plot(self.model.Xref[0,0], self.model.Xref[0,1], marker="o", markeredgecolor="red", markerfacecolor="red")
        
        if add_title:
            axis.set_title('Lyapunov fct. (V)')
            axis.set_xlabel('x0')
            axis.set_ylabel('x1')
            axis.set_aspect('equal')
        

    def sim(self, X_seq_on, X_seq_off, Udes_seq, Usafe_seq_on, slack_seq_on):
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

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(10, 10))

        xmin = np.min(X_seq_off, axis=0)
        xmax = np.max(X_seq_off, axis=0)
        xmin -= (xmax-xmin)/6
        xmax += (xmax-xmin)/6
        
        self.simControlInput(axs[0,0], Udes_seq, Usafe_seq_on)
        self.simSlack(axs[0,1], slack_seq_on)
        self.simTrajectory(axs[1,0], X_seq_on, nb_steps, xmin, xmax, filter_on=True)
        self.modelLyap(axs[1,0], xmin, xmax, add_title=False)
        self.simTrajectory(axs[1,1], X_seq_off, nb_steps, xmin, xmax, filter_on=False)

        plt.show()

    def simTrajectory(self, axis, X_seq, nb_steps, xmin, xmax, filter_on=True):
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
                axis.plot(X_seq[i+1,0], X_seq[i+1,1], 'o', color=color[i], label="Iter. "+str(i))
            else:
                axis.plot(X_seq[i+1,0], X_seq[i+1,1], 'o', color=color[i])           
        axis.legend()
        axis.set_xlim([xmin[0], xmax[0]])
        axis.set_ylim([xmin[1], xmax[1]])
        axis.set_xlabel('x[0]')
        axis.set_ylabel('x[1]')
        if filter_on:
            axis.set_title(f"State sequence (safety filter on)")
        else:
            axis.set_title(f"State sequence (safety filter off)")

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
        axis.set_xlabel('Iteration')
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
        axis.set_xlabel('Iteration')
        axis.set_title(f"Optimization slack")
        
    