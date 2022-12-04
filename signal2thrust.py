import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize, interpolate, signal
import os
import copy

from preprocess import Preprocess

def polyFct2(x, a1, a2):
    return a1*x + a2*x*x

def polyFct3(x, a1, a2, a3):
    return a1*x + a2*(x**2) + a3*(x**3)

def polyFct4(x, a1, a2, a3, a4):
    return a1*x + a2*(x**2) + a3*(x**3) + a4*(x**4)

def polyFct5(x, a1, a2, a3, a4, a5):
    return a1*x + a2*(x**2) + a3*(x**3) + a4*(x**4) + a5*(x**5)

class Signal2Thrust():
    def __init__(self, series) -> None:
        self.series = series

        # load data and convert time stamps to seconds
        pp = Preprocess(series=series)
        pp.loadData()
        pp.stamp2seconds()

        # data
        self.u = pp.u
        self.tu = pp.tu
        self.force = pp.force
        self.tforce = pp.tforce

    def saveData(self):
        """
        Save state and control input vectors in csv file
            thrust: [Fx, Fy, Fz], (N, D)
            u: [u1, ..., u6], (N, M)
        """
        force = np.empty((0,3))
        U = np.empty((0,6))
        for exp in self.tforce:
            force = np.concatenate((force, self.force[exp]), axis=0)
            U = np.concatenate((U, self.u[exp]), axis=0)

        np.savetxt(os.path.join("experiment", self.series, "data_force.csv"), force, delimiter=",")
        np.savetxt(os.path.join("experiment", self.series, "data_input.csv"), U, delimiter=",")

    def intermolateForce(self, plot=False):
        """
        Polynomial interpolation of u to match with force
        """
        for exp in self.u:
            print(f"tu min: {self.tu[exp][0]}, tforce min: {self.tforce[exp][0]}")
            print(f"tu max: {self.tu[exp][-1]}, tforce max: {self.tforce[exp][-1]}")


            while self.tforce[exp][0] < self.tu[exp][0]:
                self.tforce[exp] = self.tforce[exp][1:]
                self.force[exp] = self.force[exp][1:]

            while self.tforce[exp][-1] > self.tu[exp][-1]:
                l = self.tforce[exp].shape[0]
                self.tforce[exp] = self.tforce[exp][0:l-1]
                self.force[exp] = self.force[exp][0:l-1]

            inter_fct = interpolate.interp1d(self.tu[exp], self.u[exp], axis=0)
            u_inter = inter_fct(self.tforce[exp])

            if plot:
                fig, axs = plt.subplots(nrows=self.u[exp].shape[1], figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.tu[exp], self.u[exp][:,i], color="b", label="u")
                    ax.plot(self.tforce[exp], u_inter[:,i], '--', color="r", label="u inter.")
                    ax.legend()
                    ax.set_title(f"Control input {i}")
                plt.show()

            self.u[exp] = u_inter
            self.tu[exp] = [] # not used anymore

    def getThrust(self, trigger_delay=0.5, plot=False):
        """
        Get thrust info
        Args:
            trigger_delay: defines from which moment signal is measured
                            if trigger_delay=0 then as soon the control input rises, the signal start (idx_start) is marked
                            if trigger_delay=0.5 then the signal start is delayed by 0.5s
            plot: if True there are some plots created
        Returns:
            thrusts: dict of dict thrusts[mot][sig] where mot indicates the motor number and sig the discrete signal of the series
                     keys:  "idx_start":signal start index, "idx_stop":signal stop index, "time":time (N), "force":force (N,3), 
                            "bg":background when no force is applied (N,3), "norm":norm of force (N), "mean":mean of signal (1), 
                            "std":std of signal (1)
        """
        force = copy.deepcopy(self.force)
        tforce = copy.deepcopy(self.tforce)
        u = copy.deepcopy(self.u)
        tu = copy.deepcopy(self.tu)

        # dict for signals and motors
        thrusts = {}
        for i in range(6):
            thrusts[i] = {}


        # loop through all experiments        
        for exp in force:

            signal_state_prev = True
            signal_time_start = 0 
            mot = 0
            sig = 0
            for i, (tui, ui) in enumerate(zip(tu[exp], u[exp])):
                
                # check if currently a signal is send
                if np.sum(ui) > 0:
                    signal_state = True
                else: 
                    signal_state = False

                # check if signal state toggled
                if signal_state is not signal_state_prev or i==len(tu[exp])-1:
                    # evaluate start and stop indices of signal
                    assert signal_time_start < tui
                    idx_start =  np.argmax(tforce[exp]>signal_time_start)
                    idx_stop = np.argmax(tforce[exp]>=tui)

                    # calc. signal mean and standard deviation
                    if signal_state_prev:
                        force_sig = force[exp][idx_start:idx_stop,:]
                        time = tforce[exp][idx_start:idx_stop]
                        mot = np.argmax(u[exp][i-1])
                        sig = np.max(u[exp][i-1])
                        thrusts[mot][sig] = {   "idx_start":idx_start, "idx_stop":idx_stop, "time":time, "force":force_sig, 
                                                "bg":None, "bg_time":None, "norm":None, "mean":None, "std":None }
                    else: 
                        thrusts[mot][sig]["bg"] = force[exp][idx_start:idx_stop,:]
                        thrusts[mot][sig]["bg_time"] = tforce[exp][idx_start:idx_stop]
                    
                    # set starting time of signal while removing 0.5s
                    signal_time_start = tui + trigger_delay
                
                    # update previous state
                    signal_state_prev = signal_state

        for exp in force:
            if plot:
                fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(16, 8)) 
                axs[0,0].plot(tforce[exp], force[exp][:,0], color="b", label="Thrust x")
                axs[0,1].plot(tforce[exp], force[exp][:,1], color="g", label="Thrust y")
                axs[0,0].set_title(f"Thrust x")
                axs[0,1].set_title(f"Thrust y")
                axs[1,0].set_title(f"Thrust x (offset removed)")   
                axs[1,1].set_title(f"Thrust y (offset removed)")
                axs[1,2].set_title(f"Thrust norm (offset removed)")

                offset_plot_x = []
                offset_plot_y = []
                offset_plot_t = []


            for mot in thrusts:
                for sig in thrusts[mot]:
                    bg_x = np.mean(thrusts[mot][sig]["bg"][:,0])
                    bg_y = np.mean(thrusts[mot][sig]["bg"][:,1])
                    bg_idx = thrusts[mot][sig]["idx_stop"]
                    

                    force_x = thrusts[mot][sig]["force"][:,0] - bg_x
                    force_y = thrusts[mot][sig]["force"][:,1] - bg_y
                    thrusts[mot][sig]["norm"] = np.sqrt(np.power(force_x, 2) + np.power(force_y, 2))
                    thrusts[mot][sig]["mean"] = np.mean(thrusts[mot][sig]["norm"])
                    thrusts[mot][sig]["std"] = np.std(thrusts[mot][sig]["norm"])
            
                    if plot:
                        offset_plot_x.append(bg_x)
                        offset_plot_y.append(bg_y)
                        offset_plot_t.append(tforce[exp][bg_idx])
                        axs[1,0].plot(thrusts[mot][sig]["time"], force_x, color="b")                    
                        axs[1,1].plot(thrusts[mot][sig]["time"], force_y, color="g")
                        axs[1,2].plot(thrusts[mot][sig]["time"], thrusts[mot][sig]["norm"], color="m")
            if plot:
                axs[0,0].plot(offset_plot_t, offset_plot_x, color="r", label="Offset")
                axs[0,1].plot(offset_plot_t, offset_plot_y, color="r", label="Offset")
                axs[1,0].hlines(0, offset_plot_t[0], offset_plot_t[-1], color="r", label="Thrust x")
                axs[1,1].hlines(0, offset_plot_t[0], offset_plot_t[-1], color="r", label="Thrust y")
                axs[1,0].plot([], [], color="b", label="Thrust x")
                axs[1,1].plot([], [], color="g", label="Thrust y")
                axs[1,2].plot([], [], color="m", label="Thrust norm")
                axs[0,0].set_xlabel("time [s]")
                axs[0,0].set_ylabel("force [N]")
                axs[0,1].set_xlabel("time [s]")
                axs[0,1].set_ylabel("force [N]")
                axs[1,0].set_xlabel("time [s]")
                axs[1,0].set_ylabel("force [N]")
                axs[1,1].set_xlabel("time [s]")
                axs[1,1].set_ylabel("force [N]")
                axs[1,2].set_xlabel("time [s]")
                axs[1,2].set_ylabel("force [N]")

                axs[0,0].legend()
                axs[0,1].legend()
                axs[1,0].legend()
                axs[1,1].legend()
                axs[1,2].legend()
                fig.delaxes(axs[0,2])
                plt.show()
    
        if plot:
            for exp in force:
                fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(12, 8))             
                for i, mot in enumerate(thrusts):
                    k = int(i/len(axs[0]))
                    l = i % len(axs[0])

                    for sig in thrusts[mot]:
                        axs[k,l].set_title(f"Motor {mot+1}")
                        axs[k,l].scatter(sig, thrusts[mot][sig]["mean"], color="b")
                        axs[k,l].errorbar(sig, thrusts[mot][sig]["mean"], thrusts[mot][sig]["std"], color="r", fmt='.k')
                    axs[k,l].scatter([], [], color="b", label="Mean")
                    axs[k,l].errorbar([], [], [], color="r", fmt='.k', label="Std.")
                    axs[k,l].set_xlabel("signal")
                    axs[k,l].set_ylabel("thrust [N]")
                    axs[k,l].legend()
                plt.show()

        return thrusts

    def approxSignal2Thrust(self, thrusts, plot=False, print_coeff=False):
        
        for mot in thrusts:
            signal = []
            thrust_mean = []
            thrust_std = []
            for sig in thrusts[mot]:
                signal.append(sig)
                thrust_mean.append(thrusts[mot][sig]["mean"])
                thrust_std.append(thrusts[mot][sig]["std"])

            if plot:
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(12, 8))
                fig.suptitle(f"Motor {mot+1}")

            poly_fcts = [polyFct2, polyFct3, polyFct4, polyFct5]
            for j, fct in enumerate(poly_fcts):
                #coeff = np.polyfit(signal, thrust_mean, deg=deg)
                degree = j + 2
                coeff, _ = optimize.curve_fit(fct, signal, thrust_mean)

                lin_x = np.linspace(-0.2, 1.2, 100)
                lin_X = np.zeros((100,degree))
                for i in range(degree):
                    lin_X[:,i] = np.power(lin_x, i+1)
                thrust_approx = lin_X @ coeff

                stat_x = np.array(signal, dtype=float)
                stat_X = np.zeros((len(signal),degree))
                for i in range(degree):
                    stat_X[:,i] = np.power(stat_x, i+1)
                stat_approx = stat_X @ coeff
                _, _, rvalue, _, _ = stats.linregress(stat_approx, np.array(thrust_mean, dtype=float))

                if plot:
                    k = int(j/len(axs[0]))
                    l = j % len(axs[0])
                    axs[k,l].scatter(signal, thrust_mean, color="b", label="Meas.")
                    axs[k,l].errorbar(signal, thrust_mean, thrust_std, color="r", fmt='.k', label="Std.")
                    axs[k,l].plot(lin_x, thrust_approx, color="g", label="Approx.")
                    axs[k,l].set_title(f"Deg={degree} (R^2={np.round(rvalue**2, 4)})")
                    axs[k,l].set_xlabel("Signal")
                    axs[k,l].set_ylabel("Thrust")
                    axs[k,l].legend()

            if plot:
                plt.show()
    
        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize =(8, 8))

        for mot in thrusts:
            signal = []
            thrust_mean = []
            thrust_std = []
            for sig in thrusts[mot]:
                signal.append(sig)
                thrust_mean.append(thrusts[mot][sig]["mean"])
                thrust_std.append(thrusts[mot][sig]["std"])

            #coeff = np.polyfit(signal, thrust_mean, deg=3)
            coeff, _ = optimize.curve_fit(polyFct3, signal, thrust_mean)
            
            if print_coeff:
                print(f"\nMotor: signal->thrust {mot+1} (poly.: a1*x + a2*x^2 + ... an*x^n)")
                print(coeff)
                print(f"Motor: signal->thrust {mot+1} (poly.: an*x^n + a(n-1)*x^(n-1) + ... + a1*x)")
                print(np.flip(coeff))

            if plot:
                lin_x = np.linspace(0, 1, 100)
                lin_X = np.zeros((100,3))
                for i in range(lin_X.shape[1]):
                    lin_X[:,i] = np.power(lin_x, i+1)
                thrust_approx = lin_X @ coeff

                axs.plot(lin_x, thrust_approx, label=f"Motor {mot}")
                axs.set_title(f"Thrust approx. (poly. degree 3)")
                axs.set_xlabel("Signal")
                axs.set_ylabel("Thrust")
                axs.legend()

        if plot:
            plt.show()

    def approxThrust2Signal(self, thrusts, plot=False, print_coeff=False):

        for mot in thrusts:
            signal = []
            thrust_mean = []
            thrust_std = []
            for sig in thrusts[mot]:
                signal.append(sig)
                thrust_mean.append(thrusts[mot][sig]["mean"])
                thrust_std.append(thrusts[mot][sig]["std"])

            if plot:
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(12, 8))
                fig.suptitle(f"Motor {mot+1}")

            poly_fcts = [polyFct2, polyFct3, polyFct4, polyFct5]
            for j, fct in enumerate(poly_fcts):
                #coeff = np.polyfit(signal, thrust_mean, deg=deg)
                degree = j + 2
                coeff, _ = optimize.curve_fit(fct, thrust_mean, signal)

                lin_x = np.linspace(-0.2, 1.2, 100)
                lin_X = np.zeros((100,degree))
                for i in range(degree):
                    lin_X[:,i] = np.power(lin_x, i+1)
                signal_approx = lin_X @ coeff

                stat_x = np.array(signal, dtype=float)
                stat_X = np.zeros((len(signal),degree))
                for i in range(degree):
                    stat_X[:,i] = np.power(stat_x, i+1)
                stat_approx = stat_X @ coeff
                _, _, rvalue, _, _ = stats.linregress(stat_approx, np.array(thrust_mean, dtype=float))

                if plot:
                    k = int(j/len(axs[0]))
                    l = j % len(axs[0])
                    axs[k,l].scatter(thrust_mean, signal, color="b", label="Meas.")
                    axs[k,l].plot(lin_x, signal_approx, color="g", label="Approx.")
                    axs[k,l].set_title(f"Deg={degree} (R^2={np.round(rvalue**2, 4)})")
                    axs[k,l].set_ylabel("Signal")
                    axs[k,l].set_xlabel("Thrust")
                    axs[k,l].legend()

            if plot:
                plt.show()
    
        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize =(8, 8))

        for mot in thrusts:
            signal = []
            thrust_mean = []
            thrust_std = []
            for sig in thrusts[mot]:
                signal.append(sig)
                thrust_mean.append(thrusts[mot][sig]["mean"])
                thrust_std.append(thrusts[mot][sig]["std"])

            #coeff = np.polyfit(signal, thrust_mean, deg=3)
            coeff, _ = optimize.curve_fit(polyFct3, thrust_mean, signal)
            
            if print_coeff:
                print(f"\nMotor: thrust->signal {mot+1} (poly.: a1*x + a2*x^2 + ... an*x^n)")
                print(coeff)
                print(f"Motor: thrust->signal {mot+1} (poly.: an*x^n + a(n-1)*x^(n-1) + ... + a1*x)")
                print(np.flip(coeff))

            if plot:
                lin_x = np.linspace(0, 1, 100)
                lin_X = np.zeros((100,3))
                for i in range(lin_X.shape[1]):
                    lin_X[:,i] = np.power(lin_x, i+1)
                signal_approx = lin_X @ coeff

                axs.plot(lin_x, signal_approx, label=f"Motor {mot}")
                axs.set_title(f"Thrust approx. (poly. degree 3)")
                axs.set_ylabel("Signal")
                axs.set_xlabel("Thrust")
                axs.legend()

        if plot:
            plt.show()

    def motorTransition(self, plot=False):
        thrusts_del = self.getThrust(trigger_delay=0.5, plot=False)
        thrusts = self.getThrust(trigger_delay=0.0, plot=False)

        trans_up = {}
        trans_dw = {}
        for mot in thrusts:
            trans_up[mot] = {}
            trans_dw[mot] = {}

            if plot:
                fig, axs = plt.subplots(nrows=4, ncols=4, figsize =(12, 8))
                fig.suptitle(f"Motor: {mot + 1}")

            for i, sig in enumerate(thrusts[mot]):
                
                # de-bias force from signal and background
                bg_x_del = np.mean(thrusts_del[mot][sig]["bg"][:,0])
                bg_y_del = np.mean(thrusts_del[mot][sig]["bg"][:,1])
                force_x_del = thrusts_del[mot][sig]["force"][:,0] - bg_x_del
                force_y_del = thrusts_del[mot][sig]["force"][:,1] - bg_y_del
                force_x = thrusts[mot][sig]["force"][:,0] - bg_x_del
                force_y = thrusts[mot][sig]["force"][:,1] - bg_y_del
                bg_x = thrusts[mot][sig]["bg"][:,0] - bg_x_del
                bg_y = thrusts[mot][sig]["bg"][:,1] - bg_y_del

                # calc. norm of force vector
                force_norm = np.sqrt(np.power(force_x, 2) + np.power(force_y, 2))
                force_norm_del = np.sqrt(np.power(force_x_del, 2) + np.power(force_y_del, 2))
                bg_norm = np.sqrt(np.power(bg_x, 2) + np.power(bg_y, 2))
                bg_norm_del = np.sqrt(  np.power(thrusts_del[mot][sig]["bg"][:,0]-bg_x_del, 2) 
                                        + np.power(thrusts_del[mot][sig]["bg"][:,1]-bg_y_del, 2))

                # de-bias norm of force from signal and background
                force_norm = force_norm - np.mean(bg_norm_del)
                force_norm_del = force_norm_del - np.mean(bg_norm_del)
                bg_norm = bg_norm - np.mean(bg_norm_del)
                bg_norm_del = bg_norm_del - np.mean(bg_norm_del)

                # calc. smooth approx of force norm
                force_smooth = signal.savgol_filter(force_norm, window_length=33, polyorder=2, axis=0)
                bg_smooth = signal.savgol_filter(bg_norm, window_length=33, polyorder=2, axis=0)

                # calc. transition zone
                noise_thr = 2*(np.max(bg_norm_del) - np.min(bg_norm_del))
                thrusts_del[mot][sig]["mean"] = np.mean(force_norm_del)
                if noise_thr < thrusts_del[mot][sig]["mean"]:
                    # data of the transition zones
                    trans_up[mot][sig] = {}
                    trans_dw[mot][sig] = {}

                    # start/stop indices of transition zones
                    trans_up[mot][sig]["idx_start"] = np.argmax(force_smooth-np.max([0,force_smooth[0]]) > 0.1*thrusts_del[mot][sig]["mean"])
                    trans_up[mot][sig]["idx_stop"] = np.argmax(force_smooth > 0.9*thrusts_del[mot][sig]["mean"])
                    trans_dw[mot][sig]["idx_start"] = np.argmax(bg_smooth < 0.9*thrusts_del[mot][sig]["mean"])
                    trans_dw[mot][sig]["idx_stop"] = np.argmax(bg_smooth < 0.1*thrusts_del[mot][sig]["mean"])

                    # calc. delay and transition times
                    trans_up[mot][sig]["delay"] = thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_start"]] - thrusts[mot][sig]["time"][0]
                    trans_dw[mot][sig]["delay"] = thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_start"]] - thrusts[mot][sig]["bg_time"][0]
                    trans_up[mot][sig]["trans"] = thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_stop"]] \
                                                    - thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_start"]]
                    trans_dw[mot][sig]["trans"] = thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_stop"]] \
                                                    - thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_start"]]

                    # calc. transition slope
                    trans_up[mot][sig]["slope"] = np.mean(force_norm_del) / trans_up[mot][sig]["trans"]
                    trans_dw[mot][sig]["slope"] = - np.mean(force_norm_del) / trans_dw[mot][sig]["trans"]

                if plot:
                    k = int(i/len(axs[0]))
                    l = i % len(axs[0])

                    axs[k,l].plot(thrusts[mot][sig]["time"], force_norm, color="m")                    
                    axs[k,l].plot(thrusts[mot][sig]["bg_time"], bg_norm, color="b")
                    axs[k,l].plot(thrusts[mot][sig]["time"], force_smooth, color="g")
                    axs[k,l].plot(thrusts[mot][sig]["bg_time"], bg_smooth, color="g")
                    if noise_thr < thrusts_del[mot][sig]["mean"]:
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["time"][0], 
                                    xmax=thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_start"]], color="tab:gray", alpha=0.25)
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_start"]], 
                                    xmax=thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_stop"]], color="tab:brown", alpha=0.5)
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["bg_time"][0], 
                                    xmax=thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_start"]], color="tab:gray", alpha=0.25)
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_start"]], 
                                    xmax=thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_stop"]], color="tab:brown", alpha=0.5)
                        x_up = np.linspace(thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_start"]], 
                                            thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_stop"]], 100)
                        y_up = trans_up[mot][sig]["slope"]*(x_up-thrusts[mot][sig]["time"][trans_up[mot][sig]["idx_start"]])
                        axs[k,l].plot(x_up, y_up, color="r")
                        x_dw = np.linspace(thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_start"]], 
                                            thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_stop"]], 100)
                        y_dw = trans_dw[mot][sig]["slope"]*(x_dw-thrusts[mot][sig]["bg_time"][trans_dw[mot][sig]["idx_start"]]) + np.mean(force_norm_del)
                        axs[k,l].plot(x_dw, y_dw, color="r")
                    axs[k,l].set_title(f"Signal: {sig}")
                    # axs[k,l].set_xticks([])
                    if l == 0: axs[k,l].set_ylabel("thrust (N)")
                    if k == 3: axs[k,l].set_xlabel("seconds (s)")
                
            if plot:
                plt.show()


        self.motelTransition(trans_up, trans_dw, plot=True)

        return trans_up, trans_dw

    def motelTransition(self, trans_up, trans_dw, plot=False):

        if plot:
            fig, axs = plt.subplots(nrows=6, ncols=2, figsize =(12, 8))

        for i, mot in enumerate(trans_up):

            for j, sig in enumerate(trans_up[mot]):

                if plot:
                    k = int(i/len(axs[0]))
                    l = i % len(axs[0])

                    x_up = [trans_up[mot][sig]["delay"], trans_up[mot][sig]["delay"] + trans_up[mot][sig]["trans"]]
                    y_up = [0, trans_up[mot][sig]["slope"]*trans_up[mot][sig]["trans"]]

                    axs[i,0].plot(x_up, y_up, marker = 'o')
                    

                    x_dw = [trans_dw[mot][sig]["delay"], trans_dw[mot][sig]["delay"] + trans_dw[mot][sig]["trans"]]
                    y_dw = [0, trans_dw[mot][sig]["slope"]*trans_dw[mot][sig]["trans"]]
                    axs[i,1].plot(x_dw, y_dw, marker = 'o')
            
            axs[i,0].set_ylabel("motor {i} thrust (N)")
            if i == 5: axs[i,0].set_xlabel("time (s)")
            if i == 5: axs[i,1].set_xlabel("time (s)")
            
        if plot:
            plt.show()





def main():
    s2t = Signal2Thrust(series="signal_20221121")
    # thrusts = s2t.getThrust(plot=False)
    # s2t.approxSignal2Thrust(thrusts, plot=False, print_coeff=True)
    # s2t.approxThrust2Signal(thrusts, plot=True, print_coeff=True)

    # s2t.intermolateForce(plot=True)
    # s2t.saveData()

    s2t.motorTransition(plot=False)

    # thrusts_del = s2t.getThrust(trigger_delay=0.5, plot=False)
    # thrusts = s2t.getThrust(trigger_delay=0.0, plot=False)


    # for mot in thrusts:
    #     for sig in thrusts[mot]:
    #         # mean_del = thrusts_del[mot][sig]["mean"] 
    #         # mean = thrusts[mot][sig]["mean"]
    #         # error = abs(mean-mean_del)
    #         # print(f"Mean delayed: {mean_del}, \tmean: {mean},\t error: {error}")

    #         plt.plot(thrusts[mot][sig]["time"], thrusts[mot][sig]["norm"], color="m")
    #         plt.show()

if __name__ == "__main__":
    main()