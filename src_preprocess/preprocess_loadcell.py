import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, interpolate
import torch

from src_preprocess.functions import polyFct2, polyFct3, polyFct4, polyFct5, expRise, expFall


class Loadcell():
    def __init__(self, data, plot, sys, model) -> None:
        """
        Preprocess force sensor experiment data
        Args:
            data: Data class instance
            plot: PlotLoadcell class instance
            sys: system class instance
            model: model class instance
        """
        self.M = 6
        self.idle_signal = 0.03
        self.nb_sig_per_mot = 16

        self.data = data
        self.plot = plot
        self.sys = sys
        self.model = model

        # data
        self.sgs = None # contains [signal_start,signal_start,stop] of the thrust for every signal and motor, array (nb. motors, nb. signals, 2)
        self.bgs = None # contains [signal_start,signal_start,stop] of the back-ground for every signal and motor, array (nb. motors, nb. signals, 2)
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
        sgs = np.zeros((self.M,self.nb_sig_per_mot,3), dtype=int)
        bgs = np.zeros((self.M,self.nb_sig_per_mot,3), dtype=int)
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
                    sgs[m,s,1] = np.argmax(tf>tsig_start+trigger_delay)
                    sgs[m,s,2] = np.argmax(tf>=tf[i])
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
                    bgs[m,s,1] = np.argmax(tf>tsig_start+trigger_delay)
                    bgs[m,s,2] = np.argmax(tf>=tf[i])
                    if not bg_in_front:
                        if s < self.nb_sig_per_mot-1:
                            s += 1
                        else:
                            s = 0
                            m += 1

                tsig_start = tf[i]
                prev_state = state
                if m == self.M:
                    break
        
        if plot:
            self.plot.locSig(sgs=sgs, bgs=bgs, ids=ids)

        self.sgs = sgs
        self.bgs = bgs
        self.ids = ids

    def calcNorm(self, plot=False):
        """
        Calc norm of forces:
        1) remove offset from Fx and Fy
        2) calc. norm N with offset-free Fx and Fy
        3) de-bias N
        Args:
            plot: if True then plot the results
        """
        f, tf = self.data.get(names=["f", "tf"])
        exp = list(f.keys())[0]
        f, tf = list(f.values())[0], list(tf.values())[0]
        sgs, bgs = self.sgs.copy(), self.bgs.copy()

        # remove unused back gorund
        if bgs[0,0,1] < sgs[0,0,1]: # back ground before signal
            pass
            # f = f[0:sgs[self.M-1,self.nb_sig_per_mot-1,1],:]
            # tf = tf[0:sgs[self.M-1,self.nb_sig_per_mot-1,1]]
        else: # back ground after signal
            f = f[0:bgs[self.M-1,self.nb_sig_per_mot-1,2],:]
            tf = tf[0:bgs[self.M-1,self.nb_sig_per_mot-1,2]]

        # subtract mean back ground for every region and dimension
        f = self._subtractBG(f=f, sgs=sgs, bgs=bgs)

        # calc. norm of force and center it again
        fn = np.power(f[:,0], 2) + np.power(f[:,1], 2)
        fn = self._subtractBG(f=fn, sgs=sgs, bgs=bgs)
        neg_val = fn<0
        fn = np.sqrt(np.abs(fn))
        fn = np.where(neg_val, -fn, fn)
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

                bg_mean = np.mean(f[bgs[i,j,1]:bgs[i,j,2]], axis=0)

                if bgs[i,j,1] < sgs[i,j,1]: # back ground before signal
                    f[prev_idx:sgs[i,j,2]] = f[prev_idx:sgs[i,j,2]] - bg_mean
                    prev_idx = sgs[i,j,2]
                else: # back ground after signal
                    f[prev_idx:bgs[i,j,2]] = f[prev_idx:bgs[i,j,2]] - bg_mean
                    prev_idx = bgs[i,j,2]

        if bgs[i,j,1] < sgs[i,j,1]: # back ground before signal
            f[prev_idx:] = f[prev_idx:] - np.mean(f[prev_idx+100:], axis=0)

        return f

    def calcMeanNorm(self, plot):
        """
        Calc mean of norm of forces
        Args:
            plot: if True then plot the results
        """
        f, tf = self.data.get(names=["f", "tf"])
        f, tf = list(f.values())[0], list(tf.values())[0]
        sgs, ids = self.sgs.copy(), self.ids.copy()

        means = np.full((self.M,self.nb_sig_per_mot), np.nan)
        stds = np.full((self.M,self.nb_sig_per_mot), np.nan)
        for i in range(sgs.shape[0]):
            for j in range(sgs.shape[1]):

                means[i,j] = np.mean(f[sgs[i,j,1]:sgs[i,j,2],2], axis=0)
                stds[i,j] = np.std(f[sgs[i,j,1]:sgs[i,j,2],2], axis=0)

        if plot:
            self.plot.calcMeanNorm(means=means, stds=stds, ids=ids)

        self.means = means
        self.stds = stds

    def signal2thrustCoeff(self, plot=False, verb=True):
        """
        Determine the signal-to-thrust coefficients
        Args:
            plot: if True then plot the results
            verb: if True then print coefficients of third-order polynomial
        """
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
            self.plot.signal2thrustAllMotors(coeffs=coeffs)

    def thrust2signalCoeff(self, plot=False, verb=True):
        """
        Determine the thrust-to-signal coefficients
        Args:
            plot: if True then plot the results
            verb: if True then print coefficients of third-order polynomial
        """
        ids, means, stds = self.ids.copy(), self.means.copy(), self.stds.copy()

        poly_fcts = [polyFct2, polyFct3, polyFct4, polyFct5]
        coeffs = np.zeros((means.shape[0],len(poly_fcts),5))
        for i in range(means.shape[0]):
           
            for j, fct in enumerate(poly_fcts):
                coeff, _ = optimize.curve_fit(fct, means[i,:], ids[i,:,1])
                coeffs[i,j,:j+2] = coeff
              
                if verb and j==1:
                    print(f"\nMotor: thrust->signal {i+1} (poly.: a1*x + a2*x^2 + ... an*x^n)")
                    print(f"[{coeff[0]}, {coeff[1]}, {coeff[2]}]")
                    print(f"Motor: thrust->signal {i+1} (poly.: an*x^n + a(n-1)*x^(n-1) + ... + a1*x)")
                    print(f"[{np.flip(coeff)[0]}, {np.flip(coeff)[1]}, {np.flip(coeff)[2]}]")
        
        if plot:
            self.plot.thrust2signal(means=means, stds=stds, ids=ids, coeffs=coeffs)          

    def motorTransition(self, thr_y_final=0.95, plot=False, signal_space=False):
        """
        Approximate motor transition zone using first order model
        Args:
            thr_y_final: procentage of final value that must be reached for the transition zone to end
            plot: if True then plot the results
            signal_space: Do the approximation in signal space. This means the thrust is first converted
                            to a signal using the thrust-to-signal mapping found by self.thrust2signalCoeff.
                            Then, delay/trans/tau are calculated for the signal transition.
        """
        f, tf = self.data.get(names=["f", "tf"])
        f, tf = list(f.values())[0], list(tf.values())[0]
        fn = f[:,2]
        sgs, ids, means = self.sgs.copy(), self.ids.copy(), self.means.copy()

        bgs_trans = self.bgs.copy()
        if bgs_trans[0,0,1] < sgs[0,0,1]: # back ground before signal
            bgs_trans = bgs_trans.flatten()[3:]
            bgs_trans = np.append(bgs_trans, values=[sgs[-1,-1,2], sgs[-1,-1,2]+100, len(tf)-1])
            bgs_trans = bgs_trans.reshape(self.M, self.nb_sig_per_mot, 3)

        # convert thrust (fn) into signal using white box model
        fn_thrust = fn.copy()
        if signal_space:
            for i in range(sgs.shape[0]):
                if bgs_trans[0,0,1] < sgs[0,0,1]:
                    fn[bgs_trans[i,0,0]:sgs[i,self.nb_sig_per_mot-1,2]] = \
                                            self._thrust2signal(fn=fn[bgs_trans[i,0,0]:sgs[i,self.nb_sig_per_mot-1,2]], mot=i+1)
                else:
                    fn[sgs[i,0,0]:bgs_trans[i,self.nb_sig_per_mot-1,2]] = \
                                            self._thrust2signal(fn=fn[sgs[i,0,0]:bgs_trans[i,self.nb_sig_per_mot-1,2]], mot=i+1)

        trans = np.full((sgs.shape[0],sgs.shape[1],2), np.nan) # (nb. motors, nb. signals, [up,down])
        delay = np.full((sgs.shape[0],sgs.shape[1],2), np.nan) # (nb. motors, nb. signals, [up,down])
        tau = np.full((sgs.shape[0],sgs.shape[1],2), np.nan) # (nb. motors, nb. signals, [up,down])
        for i in range(sgs.shape[0]):
            for j in range(sgs.shape[1]):
                # skip signal if noise level is too high
                noise_thr = 2*(np.max(fn_thrust[bgs_trans[i,j,1]:bgs_trans[i,j,2]]) - np.min(fn_thrust[bgs_trans[i,j,1]:bgs_trans[i,j,2]]))
                if noise_thr > means[i,j]:
                    continue

                if signal_space:
                    steady_state = ids[i,j,1]
                else:
                    steady_state = means[i,j]

                # calc. first order approx. 
                fit_up_X = (tf[sgs[i,j,0]:sgs[i,j,1]] - tf[sgs[i,j,0]], np.ones(tf[sgs[i,j,0]:sgs[i,j,1]].shape) * steady_state)
                [tau[i,j,0], delay[i,j,0]], _ = optimize.curve_fit(expRise, fit_up_X, fn[sgs[i,j,0]:sgs[i,j,1]], [0.05, 0.04])

                fit_dw_X = (tf[bgs_trans[i,j,0]:bgs_trans[i,j,1]] - tf[bgs_trans[i,j,0]], 
                            np.ones(tf[bgs_trans[i,j,0]:bgs_trans[i,j,1]].shape) * steady_state)
                [tau[i,j,1], delay[i,j,1]], _ = optimize.curve_fit(expFall, fit_dw_X, fn[bgs_trans[i,j,0]:bgs_trans[i,j,1]], [0.05, 0.04])                  

                # calc. time of exponential to reach certain procentage of final value
                trans[i,j,0] = - tau[i,j,0] * np.log(1-thr_y_final)
                trans[i,j,1] = - tau[i,j,1] * np.log(1-thr_y_final)

        if plot:
            self.plot.motorTransZone(fn=fn, fn_thrust=fn_thrust, tf=tf, sgs=sgs, bgs_trans=bgs_trans, ids=ids, means=means, tau=tau, 
                                        delay=delay, trans=trans, signal_space=signal_space)
            self.plot.motorTransStat(ids=ids, tau=tau, delay=delay, trans=trans)

    def _thrust2signal(self, fn, mot):
        """
        Use the white box model for the thrust-to-signal mapping
        Args:
            fn: thrust, numpy array (N)
            mot: motor ID for which the thrust should be converted into a signal, int [1,6]
        """
        thrust = torch.zeros((len(fn), 6), dtype=torch.float32)
        thrust[:,int(mot-1)] = torch.tensor(fn, dtype=torch.float32)

        thrust_poly = self.sys.polyExpandU(thrust)
        sig = self.model.thrust2signal(thrust_poly)

        return sig[:,int(mot-1)].detach().numpy()

    