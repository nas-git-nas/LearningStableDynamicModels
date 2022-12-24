import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize, interpolate, signal
from copy import deepcopy
import torch


from src_preprocess.functions import polyFct2, polyFct3, polyFct4, polyFct5, expRise, expFall


class Loadcell():
    def __init__(self, data, plot, sys, model) -> None:
        self.M = 6
        self.idle_signal = 0.03
        self.nb_sig_per_mot = 16

        self.data = data
        self.plot = plot
        self.sys = sys
        self.model = model

        # data
        self.sgs = None # contains [start,stop] of the thrust for every signal and motor, array (nb. motors, nb. signals, 2)
        self.bgs = None # contains [start,stop] of the back-ground for every signal and motor, array (nb. motors, nb. signals, 2)
        self.ids = None # contains [motor_id,signal] for every signal and motor, array (nb. motors, nb. signals, 2)
        self.means = None # contains means of thrust for every signal and motor, array (nb. motors, nb. signals)
        self.std = None # contains stds of thrust for every signal and motor, array (nb. motors, nb. signals)

    def cropData(self):
        """
        Crop force data such that min(tu)<min(tf) and max(tf)<max(tu)
        This is necessary to interpolate the force afterwards
        """
        tu, f, tf = self.data.get(names=["tu", "f", "tf"])

        for exp in self.data.exps:

            while tf[exp][0] < tu[exp][0]:
                tf[exp] = tf[exp][1:]
                f[exp] = f[exp][1:]

            while tf[exp][-1] > tu[exp][-1]:
                tf[exp] = tf[exp][0:-1]
                f[exp] = f[exp][0:-1]

        self.data.set(names=["f", "tf"], datas=[f, tf])  

    def interpolateU(self, plot=False):
        """
        Polynomial interpolation of u to match with force
        Args:
            plot: if True plot results
        """
        u, tu, tf = self.data.get(names=["u", "tu", "tf"])

        u_inter = {}
        for exp in self.data.exps:

            inter_fct = interpolate.interp1d(tu[exp], u[exp], axis=0)
            u_inter[exp] = inter_fct(tf[exp])

        if plot:
            self.plot.interpolateU(u_inter)

        self.data.set(names=["u"], datas=[u_inter])
        self.data.delete(names=["tu"])

    def locSig(self, trigger_delay=0.5, plot=False):
        """
        Determines the start and stop indices for every signal (sgs) and the corresponding back-ground (bgs)
        ids contains the motor id and signal for every entry in sgs/bgs
        Args:
            trigger_delay: delay start of sgs/bgs by this amount of seconds to avoid transition zones
            plot: if True then plot the results
        """
        u, f, tf = self.data.get(names=["u", "f", "tf"])
        u, f, tf = list(u.values())[0], list(f.values())[0], list(tf.values())[0]

        # init. previous state
        prev_state = False
        bg_in_front = True
        if np.max(u[0,:]) > self.idle_signal:
            prev_state = True
            bg_in_front = False

        tsig_start = tf[0] + trigger_delay
        sgs = np.zeros((self.M,self.nb_sig_per_mot,2), dtype=int)
        bgs = np.zeros((self.M,self.nb_sig_per_mot,2), dtype=int)
        ids = np.zeros((self.M,self.nb_sig_per_mot,2), dtype=float)
        m = 0
        s = 0
        for i in range(len(tf)):
            # determine state
            state = False
            if np.max(u[i,:]) > self.idle_signal:
                state = True

            # if state toggled or last element is reached, then calc. indices
            if state is not prev_state or i==len(tf)-1:
                assert tsig_start < tf[i]

                if prev_state: # signal
                    sgs[m,s,0] = np.argmax(tf>tsig_start)
                    sgs[m,s,1] = np.argmax(tf>=tf[i])
                    ids[m,s,0] = np.argmax(u[i-10]) + 1 # go 10 indices back to avoid transition zone
                    ids[m,s,1] = np.max(u[i-10,:])
                    if bg_in_front:
                        if s < self.nb_sig_per_mot-1:
                            s += 1
                        else:
                            s = 0
                            m += 1
                else: # background
                    bgs[m,s,0] = np.argmax(tf>tsig_start)
                    bgs[m,s,1] = np.argmax(tf>=tf[i])
                    if not bg_in_front:
                        if s < self.nb_sig_per_mot-1:
                            s += 1
                        else:
                            s = 0
                            m += 1

                tsig_start = tf[i] + trigger_delay
                prev_state = state
                if m == self.M:
                    break
        
        if plot:
            self.plot.locSig(sgs=sgs, bgs=bgs, ids=ids)

        self.sgs = sgs
        self.bgs = bgs
        self.ids = ids

    def calcNorm(self, plot):
        f, tf = self.data.get(names=["f", "tf"])
        exp = list(f.keys())[0]
        f, tf = list(f.values())[0], list(tf.values())[0]
        sgs, bgs = self.sgs.copy(), self.bgs.copy()

        # remove unused back gorund
        if bgs[0,0,0] < sgs[0,0,0]: # back ground before signal
            f = f[0:sgs[self.M-1,self.nb_sig_per_mot-1,1],:]
            tf = tf[0:sgs[self.M-1,self.nb_sig_per_mot-1,1]]
        else: # back ground after signal
            f = f[0:bgs[self.M-1,self.nb_sig_per_mot-1,1],:]
            tf = tf[0:bgs[self.M-1,self.nb_sig_per_mot-1,1]]

        # subtract mean back ground for every region and dimension
        f = self._subtractBG(f=f, sgs=sgs, bgs=bgs)

        # calc. norm of force and center it again
        fn = np.sqrt(np.power(f[:,0], 2) + np.power(f[:,1], 2))
        fn = self._subtractBG(f=fn, sgs=sgs, bgs=bgs)
        f[:,2] = fn

        self.data.set(names=["f", "tf"], datas=[{exp:f}, {exp:tf}])

        if plot:
            self.plot.calcNorm()
        
    
    def _subtractBG(self, f, sgs, bgs):
        """
        Subtract back ground mean of measurement with any number of dimensions
        Args:
            f: force to subtact mean, array (N,dim)
            sgs: contains [start,stop] of the thrust for every signal and motor, array (nb. motors, nb. signals, 2)
            bgs: contains [start,stop] of the back-ground for every signal and motor, array (nb. motors, nb. signals, 2)
        """
        prev_idx = 0
        for i in range(sgs.shape[0]):
            for j in range(sgs.shape[1]):

                bg_mean = np.mean(f[bgs[i,j,0]:bgs[i,j,1]], axis=0)

                if bgs[i,j,0] < sgs[i,j,0]: # back ground before signal
                    f[prev_idx:sgs[i,j,1]] = f[prev_idx:sgs[i,j,1]] - bg_mean
                    prev_idx = sgs[i,j,1]
                else: # back ground after signal
                    f[prev_idx:bgs[i,j,1]] = f[prev_idx:bgs[i,j,1]] - bg_mean
                    prev_idx = bgs[i,j,1]

        return f

    def calcMeanNorm(self, plot):
        f, tf = self.data.get(names=["f", "tf"])
        f, tf = list(f.values())[0], list(tf.values())[0]
        sgs, ids = self.sgs.copy(), self.ids.copy()

        means = np.zeros((self.M,self.nb_sig_per_mot))
        stds = np.zeros((self.M,self.nb_sig_per_mot))
        for i in range(sgs.shape[0]):
            for j in range(sgs.shape[1]):

                means[i,j] = np.mean(f[sgs[i,j,0]:sgs[i,j,1],2], axis=0)
                stds[i,j] = np.std(f[sgs[i,j,0]:sgs[i,j,1],2], axis=0)

        if plot:
            self.plot.calcMeanNorm(means=means, stds=stds, ids=ids)

        self.means = means
        self.stds = stds

    def signal2thrust(self, plot=False, verb=True):
        ids, means, stds = self.ids.copy(), self.means.copy(), self.stds.copy()

        poly_fcts = [polyFct2, polyFct3, polyFct4, polyFct5]
        coeffs = np.zeros((means.shape[0],len(poly_fcts),5))
        for i in range(means.shape[0]):
           
            for j, fct in enumerate(poly_fcts):
                coeff, _ = optimize.curve_fit(fct, ids[i,:,1], means[i,:])
                coeffs[i,j,:j+2] = coeff
              
                if verb and j==1:
                    print(f"\nMotor: signal->thrust {i+1} (poly.: a1*x + a2*x^2 + ... an*x^n)")
                    print(f"[{coeff[0]}, {coeff[1]}, {coeff[2]}]")
                    print(f"Motor: signal->thrust {i+1} (poly.: an*x^n + a(n-1)*x^(n-1) + ... + a1*x)")
                    print(f"[{np.flip(coeff)[0]}, {np.flip(coeff)[1]}, {np.flip(coeff)[2]}]")
        
        if plot:
            self.plot.signal2thrust(means=means, stds=stds, ids=ids, coeffs=coeffs)

  
                

    


    

    

    def approxThrust2Signal(self, thrusts, plot=False, print_coeff=False):
        """
        Args:
            plot: if True plot results
        """

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
                print(f"[{coeff[0]}, {coeff[1]}, {coeff[2]}]")
                print(f"Motor: thrust->signal {mot+1} (poly.: an*x^n + a(n-1)*x^(n-1) + ... + a1*x)")
                print(f"[{np.flip(coeff)[0]}, {np.flip(coeff)[1]}, {np.flip(coeff)[2]}]")

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

    def motorTransition(self, plot=False, signal_space=False):
        """
        Args:
            plot: if True plot results
        """
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

                if signal_space:
                    force_norm = self.thrust2signal(force_norm, mot)
                    bg_norm = self.thrust2signal(bg_norm, mot)

                # calc. transition zone
                noise_thr = 2*(np.max(bg_norm_del) - np.min(bg_norm_del))
                thrusts_del[mot][sig]["mean"] = np.mean(force_norm_del)
                if noise_thr < thrusts_del[mot][sig]["mean"]:
                    # data of the transition zones
                    trans_up[mot][sig] = {}
                    trans_dw[mot][sig] = {}

                    # start/stop indices of transition zones
                    if signal_space:
                        steady_state = sig
                    else:
                        steady_state = thrusts_del[mot][sig]["mean"]

                    # calc. first order approx.
                    fit_up_X = (thrusts[mot][sig]["time"]-thrusts[mot][sig]["time"][0], 
                                np.ones((len(thrusts[mot][sig]["time"])))*steady_state)
                    [tau_up, delay_up], _ = optimize.curve_fit(expRise, fit_up_X, force_norm, [0.05, 0.04])

                    fit_dw_X = (thrusts[mot][sig]["bg_time"]-thrusts[mot][sig]["bg_time"][0], 
                                np.ones((len(thrusts[mot][sig]["bg_time"])))*steady_state)
                    [tau_dw, delay_dw], _ = optimize.curve_fit(expFall, fit_dw_X, bg_norm, [0.05, 0.04])

                    # calc. time of exponential to reach certain procentage of final value
                    thr_y_final = 0.95
                    trans_up[mot][sig]["trans"] = - tau_up * np.log(1-thr_y_final)
                    trans_dw[mot][sig]["trans"] = - tau_dw * np.log(1-thr_y_final)
                    trans_up[mot][sig]["delay"] = delay_up
                    trans_dw[mot][sig]["delay"] = delay_dw
                    trans_up[mot][sig]["tau"] = tau_up
                    trans_dw[mot][sig]["tau"] = tau_dw

                if plot:
                    k = int(i/len(axs[0]))
                    l = i % len(axs[0])

                    axs[k,l].plot(thrusts[mot][sig]["time"], force_norm, color="m")                    
                    axs[k,l].plot(thrusts[mot][sig]["bg_time"], bg_norm, color="b")
                    
                    if noise_thr < thrusts_del[mot][sig]["mean"]:
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["time"][0], 
                                            xmax=thrusts[mot][sig]["time"][0]+delay_up, color="tab:gray", alpha=0.25)
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["time"][0]+delay_up, 
                                            xmax=thrusts[mot][sig]["time"][0]+delay_up+trans_up[mot][sig]["trans"], 
                                            color="tab:brown", alpha=0.5)
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["bg_time"][0], 
                                            xmax=thrusts[mot][sig]["bg_time"][0]+delay_dw, color="tab:gray", alpha=0.25)
                        axs[k,l].axvspan(xmin=thrusts[mot][sig]["bg_time"][0]+delay_dw, 
                                            xmax=thrusts[mot][sig]["bg_time"][0]+delay_dw+trans_dw[mot][sig]["trans"], 
                                            color="tab:brown", alpha=0.5)

                        # if not signal_space:
                        axs[k,l].plot(thrusts[mot][sig]["time"], expRise(X=fit_up_X, tau=tau_up, delay=delay_up), color="g")
                        axs[k,l].plot(thrusts[mot][sig]["bg_time"], expFall(X=fit_dw_X, tau=tau_dw, delay=delay_dw), color="g")
                    axs[k,l].set_title(f"Signal: {sig}")
                    # axs[k,l].set_xticks([])
                    if signal_space:
                        if l == 0: axs[k,l].set_ylabel("signal")
                    else:
                        if l == 0: axs[k,l].set_ylabel("thrust (N)")
                    if k == 3: axs[k,l].set_xlabel("seconds (s)")
                
            if plot:
                plt.show()

        return trans_up, trans_dw

    def thrust2signal(self, thrust_mot, mot):

        thrust = torch.zeros((len(thrust_mot), 6), dtype=torch.float32)
        thrust[:,mot-1] = torch.tensor(thrust_mot, dtype=torch.float32)

        thrust_poly = self.sys.polyExpandU(thrust)
        sig = self.model.thrust2signal(thrust_poly)

        return sig[:,mot-1].detach().numpy()

    def plotTransTime(self, trans_up, trans_dw):
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(8, 12))

        up_delay_all = []
        dw_delay_all = []
        up_trans_all = []       
        dw_trans_all = []
        up_tau_all = []
        dw_tau_all = []
        
        up_delay = []
        dw_delay = []
        up_trans = []          
        dw_trans = []
        up_tau = []
        dw_tau = []
        for i, mot in enumerate(trans_up):
            sigs = []
            up_delay = []
            dw_delay = []
            up_trans = []          
            dw_trans = []
            up_tau = []
            dw_tau = []
            for j, sig in enumerate(trans_up[mot]):

                sigs.append(sig)
                up_delay.append(trans_up[mot][sig]["delay"])
                dw_delay.append(trans_dw[mot][sig]["delay"])
                up_trans.append(trans_up[mot][sig]["trans"])               
                dw_trans.append(trans_dw[mot][sig]["trans"])
                up_tau.append(trans_up[mot][sig]["tau"])               
                dw_tau.append(trans_dw[mot][sig]["tau"])

            axs[0,0].plot(sigs, up_delay, marker = 'o', label=f"motor {mot}")
            axs[0,1].plot(sigs, dw_delay, marker = 'o', label=f"motor {mot}")
            axs[1,0].plot(sigs, up_trans, marker = 'o', label=f"motor {mot}")
            axs[1,1].plot(sigs, dw_trans, marker = 'o', label=f"motor {mot}")
            axs[2,0].plot(sigs, up_tau, marker = 'o', label=f"motor {mot}")
            axs[2,1].plot(sigs, dw_tau, marker = 'o', label=f"motor {mot}")
            up_delay_all.append(np.mean(up_delay))
            dw_delay_all.append(np.mean(dw_delay))
            up_trans_all.append(np.mean(up_trans))            
            dw_trans_all.append(np.mean(dw_trans))
            up_tau_all.append(np.mean(up_tau))            
            dw_tau_all.append(np.mean(dw_tau))

        axs[0,0].set_title(f"Up delay (mean={np.round(np.mean(up_delay_all),3)})")
        axs[0,1].set_title(f"Down delay (mean={np.round(np.mean(dw_delay_all),3)})")
        axs[1,0].set_title(f"Up trans (mean={np.round(np.mean(up_trans_all),3)})")        
        axs[1,1].set_title(f"Down trans (mean={np.round(np.mean(dw_trans_all),3)})")
        axs[2,0].set_title(f"Up tau (mean={np.round(np.mean(up_tau_all),3)})")        
        axs[2,1].set_title(f"Down tau (mean={np.round(np.mean(dw_tau_all),3)})")
        axs[0,0].set_ylabel("time (s)")
        axs[1,0].set_ylabel("time (s)")
        axs[2,0].set_xlabel("signal")
        axs[2,0].set_ylabel("time (s)")
        axs[2,1].set_xlabel("signal")
        axs[0,0].legend()
        axs[1,0].legend()
        axs[0,1].legend()
        axs[1,1].legend()
        axs[2,0].legend()
        axs[2,1].legend()
        plt.show()


# def getThrust(self, trigger_delay=0.5, plot=False):
#         """
#         Get thrust info
#         Args:
#             trigger_delay: defines from which moment signal is measured
#                             if trigger_delay=0 then as soon the control input rises, the signal start (idx_start) is marked
#                             if trigger_delay=0.5 then the signal start is delayed by 0.5s
#             plot: if True there are some plots created
#         Returns:
#             thrusts: dict of dict thrusts[mot][sig] where mot indicates the motor number and sig the discrete signal of the series
#                      keys:  "idx_start":signal start index, "idx_stop":signal stop index, "time":time (N), "force":force (N,3), 
#                             "bg":background when no force is applied (N,3), "norm":norm of force (N), "mean":mean of signal (1), 
#                             "std":std of signal (1)
#         """
#         u, tu, f, tf = self.data.get(names=["u", "tu", "f" "tf"])
#         u, tu, f, tf = list(u.keys())[0], list(tu.keys())[0], list(f.keys())[0], list(tf.keys())[0]
#         thrusts = deepcopy(thrusts)

#         signal_state_prev = False
#         signal_time_start = 0 
#         mot = 0
#         sig = 0
#         for i, (tui, ui) in enumerate(zip(tu, u)):
            
#             # check if currently a signal is send
#             if np.sum(ui) > len(ui)*self.idle_signal:
#                 signal_state = True
#             else: 
#                 signal_state = False

#             # check if signal state toggled
#             if signal_state is not signal_state_prev or i==len(tu)-1:
#                 # evaluate start and stop indices of signal
#                 assert signal_time_start < tui
#                 idx_start =  np.argmax(tf>signal_time_start)
#                 idx_stop = np.argmax(tf>=tui)

#                 # calc. signal mean and standard deviation
#                 if signal_state_prev:
#                     force_sig = f[idx_start:idx_stop,:]
#                     time = tf[idx_start:idx_stop]
#                     mot = np.argmax(u[i-1])
#                     sig = np.max(u[i-1])
#                     thrusts[mot][sig] = {   "idx_start":idx_start, "idx_stop":idx_stop, "time":time, "force":force_sig, 
#                                             "bg":None, "bg_time":None, "norm":None, "mean":None, "std":None }
#                 else:
#                     if sig in thrusts[mot].keys():
#                         thrusts[mot][sig]["bg"] = f[idx_start:idx_stop,:]
#                         thrusts[mot][sig]["bg_time"] = tf[idx_start:idx_stop]
                
#                 # set starting time of signal while removing 0.5s
#                 signal_time_start = tui + trigger_delay
            
#                 # update previous state
#                 signal_state_prev = signal_state


#         if plot:
#             fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(16, 8)) 
#             axs[0,0].plot(tf, f[:,0], color="b", label="Thrust x")
#             axs[0,1].plot(tf, f[:,1], color="g", label="Thrust y")
#             axs[0,0].set_title(f"Thrust x")
#             axs[0,1].set_title(f"Thrust y")
#             axs[1,0].set_title(f"Thrust x (offset removed)")   
#             axs[1,1].set_title(f"Thrust y (offset removed)")
#             axs[1,2].set_title(f"Thrust norm (offset removed)")

#             offset_plot_x = []
#             offset_plot_y = []
#             offset_plot_t = []


#         for mot in thrusts:
#             for sig in thrusts[mot]:
#                 bg_x = thrusts[mot][sig]["bg"][:,0]
#                 bg_y = thrusts[mot][sig]["bg"][:,1]
#                 bg_idx = thrusts[mot][sig]["idx_stop"]
                
#                 # de-bias x and y quantities
#                 force_x = thrusts[mot][sig]["force"][:,0] - np.mean(bg_x)
#                 force_y = thrusts[mot][sig]["force"][:,1] - np.mean(bg_y)
#                 bg_x = bg_x - np.mean(bg_x)
#                 bg_y = bg_y - np.mean(bg_y)

#                 # calc. norm of forces
#                 force_norm = np.sqrt(np.power(force_x, 2) + np.power(force_y, 2))
#                 bg_norm = np.sqrt(np.power(bg_x, 2) + np.power(bg_y, 2))

#                 # de-bias norm
#                 force_norm = force_norm - np.mean(bg_norm)
#                 bg_norm = bg_norm - np.mean(bg_norm)

#                 thrusts[mot][sig]["norm"] = force_norm
#                 thrusts[mot][sig]["mean"] = np.mean(force_norm)
#                 thrusts[mot][sig]["std"] = np.std(force_norm)
        
#                 if plot:
#                     offset_plot_x.append(np.mean(bg_x))
#                     offset_plot_y.append(np.mean(bg_y))
#                     offset_plot_t.append(tf[bg_idx])
#                     axs[1,0].plot(thrusts[mot][sig]["time"], force_x, color="b")                    
#                     axs[1,1].plot(thrusts[mot][sig]["time"], force_y, color="g")
#                     axs[1,2].plot(thrusts[mot][sig]["time"], thrusts[mot][sig]["norm"], color="m")
#         if plot:
#             axs[0,0].plot(offset_plot_t, offset_plot_x, color="r", label="Offset")
#             axs[0,1].plot(offset_plot_t, offset_plot_y, color="r", label="Offset")
#             axs[1,0].hlines(0, offset_plot_t[0], offset_plot_t[-1], color="r", label="Thrust x")
#             axs[1,1].hlines(0, offset_plot_t[0], offset_plot_t[-1], color="r", label="Thrust y")
#             axs[1,0].plot([], [], color="b", label="Thrust x")
#             axs[1,1].plot([], [], color="g", label="Thrust y")
#             axs[1,2].plot([], [], color="m", label="Thrust norm")
#             axs[0,0].set_xlabel("time [s]")
#             axs[0,0].set_ylabel("force [N]")
#             axs[0,1].set_xlabel("time [s]")
#             axs[0,1].set_ylabel("force [N]")
#             axs[1,0].set_xlabel("time [s]")
#             axs[1,0].set_ylabel("force [N]")
#             axs[1,1].set_xlabel("time [s]")
#             axs[1,1].set_ylabel("force [N]")
#             axs[1,2].set_xlabel("time [s]")
#             axs[1,2].set_ylabel("force [N]")

#             axs[0,0].legend()
#             axs[0,1].legend()
#             axs[1,0].legend()
#             axs[1,1].legend()
#             axs[1,2].legend()
#             fig.delaxes(axs[0,2])
#             plt.show()

#         if plot:

#             fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(12, 8))             
#             for i, mot in enumerate(thrusts):
#                 k = int(i/len(axs[0]))
#                 l = i % len(axs[0])

#                 for sig in thrusts[mot]:
#                     axs[k,l].set_title(f"Motor {mot+1}")
#                     axs[k,l].scatter(sig, thrusts[mot][sig]["mean"], color="b")
#                     axs[k,l].errorbar(sig, thrusts[mot][sig]["mean"], thrusts[mot][sig]["std"], color="r", fmt='.k')
#                 axs[k,l].scatter([], [], color="b", label="Mean")
#                 axs[k,l].errorbar([], [], [], color="r", fmt='.k', label="Std.")
#                 axs[k,l].set_xlabel("signal")
#                 axs[k,l].set_ylabel("thrust [N]")
#                 axs[k,l].legend()
#             plt.show()

#         self.thrusts = deepcopy(thrusts)

# def approxSignal2Thrust(self, thrusts, plot=False, print_coeff=False):
#         """
#         Args:
#             plot: if True plot results
#         """
        
#         for mot in thrusts:
#             signal = []
#             thrust_mean = []
#             thrust_std = []
#             for sig in thrusts[mot]:
#                 signal.append(sig)
#                 thrust_mean.append(thrusts[mot][sig]["mean"])
#                 thrust_std.append(thrusts[mot][sig]["std"])

#             if plot:
#                 fig, axs = plt.subplots(nrows=2, ncols=2, figsize =(12, 8))
#                 fig.suptitle(f"Motor {mot+1}")

#             poly_fcts = [polyFct2, polyFct3, polyFct4, polyFct5]
#             for j, fct in enumerate(poly_fcts):
#                 #coeff = np.polyfit(signal, thrust_mean, deg=deg)
#                 degree = j + 2
#                 coeff, _ = optimize.curve_fit(fct, signal, thrust_mean)

#                 lin_x = np.linspace(-0.2, 1.2, 100)
#                 lin_X = np.zeros((100,degree))
#                 for i in range(degree):
#                     lin_X[:,i] = np.power(lin_x, i+1)
#                 thrust_approx = lin_X @ coeff

#                 stat_x = np.array(signal, dtype=float)
#                 stat_X = np.zeros((len(signal),degree))
#                 for i in range(degree):
#                     stat_X[:,i] = np.power(stat_x, i+1)
#                 stat_approx = stat_X @ coeff
#                 _, _, rvalue, _, _ = stats.linregress(stat_approx, np.array(thrust_mean, dtype=float))

#                 if plot:
#                     k = int(j/len(axs[0]))
#                     l = j % len(axs[0])
#                     axs[k,l].scatter(signal, thrust_mean, color="b", label="Meas.")
#                     axs[k,l].errorbar(signal, thrust_mean, thrust_std, color="r", fmt='.k', label="Std.")
#                     axs[k,l].plot(lin_x, thrust_approx, color="g", label="Approx.")
#                     axs[k,l].set_title(f"Deg={degree} (R^2={np.round(rvalue**2, 4)})")
#                     axs[k,l].set_xlabel("Signal")
#                     axs[k,l].set_ylabel("Thrust")
#                     axs[k,l].legend()

#             if plot:
#                 plt.show()
    
#         if plot:
#             fig, axs = plt.subplots(nrows=1, ncols=1, figsize =(8, 8))

#         for mot in thrusts:
#             signal = []
#             thrust_mean = []
#             thrust_std = []
#             for sig in thrusts[mot]:
#                 signal.append(sig)
#                 thrust_mean.append(thrusts[mot][sig]["mean"])
#                 thrust_std.append(thrusts[mot][sig]["std"])

#             #coeff = np.polyfit(signal, thrust_mean, deg=3)
#             coeff, _ = optimize.curve_fit(polyFct3, signal, thrust_mean)
            
#             if print_coeff:
#                 print(f"\nMotor: signal->thrust {mot+1} (poly.: a1*x + a2*x^2 + ... an*x^n)")
#                 print(f"[{coeff[0]}, {coeff[1]}, {coeff[2]}]")
#                 print(f"Motor: signal->thrust {mot+1} (poly.: an*x^n + a(n-1)*x^(n-1) + ... + a1*x)")
#                 print(f"[{np.flip(coeff)[0]}, {np.flip(coeff)[1]}, {np.flip(coeff)[2]}]")

#             if plot:
#                 lin_x = np.linspace(0, 1, 100)
#                 lin_X = np.zeros((100,3))
#                 for i in range(lin_X.shape[1]):
#                     lin_X[:,i] = np.power(lin_x, i+1)
#                 thrust_approx = lin_X @ coeff

#                 axs.plot(lin_x, thrust_approx, label=f"Motor {mot}")
#                 axs.set_title(f"Thrust approx. (poly. degree 3)")
#                 axs.set_xlabel("Signal")
#                 axs.set_ylabel("Thrust")
#                 axs.legend()

#         if plot:
#             plt.show()