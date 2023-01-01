import matplotlib.pyplot as plt
import numpy as np
from pysindy.differentiation import SINDyDerivative
from scipy import interpolate, signal
import torch


class PreprocessHolohover():
    def __init__(self, data, plot, sys, model) -> None:
        """
        Preprocess Holohover experiment data
        Args:
            data: Data class instance
            plot: PlotHolohover class instance
            sys: system class instance
            model: model class instance
        """
        self.D = 6
        self.M = 6

        self.data = data
        self.plot = plot
        self.sys = sys
        self.model = model           

    def cropData(self, plot=False):
        """
        Remove data in the beginning and the end
        Args:
            plot: if True plot results
        """
        x, tx, u, tu = self.data.get(names=["x", "tx", "u", "tu"])

        plot_lw_x_idx = {}
        plot_lw_u_idx = {}
        plot_up_x_idx = {}
        plot_up_u_idx = {}
        for exp in self.data.exps:

            # determine time where to cut lower and upper part
            lower_time = np.minimum(tx[exp][0], tu[exp][0])
            for i, umax in enumerate(np.max(np.abs(u[exp]), axis=1)):
                if umax > 0.05:
                    lower_time = tu[exp][i]
                    break
            
            upper_time = np.maximum(tx[exp][-1], tu[exp][-1])
            for i, theta_max in enumerate(np.abs(x[exp][:,2])):
                if theta_max > 3.0:
                    upper_time = tx[exp][i]
                    break

            # determine indices where to cut lower and upper part
            lower_x_idx = 0           
            upper_x_idx = x[exp].shape[0] - 1
            for i, t in enumerate(tx[exp]):
                if t>lower_time and lower_x_idx==0:
                    lower_x_idx = i
                if t > upper_time:
                    upper_x_idx = i - 1
                    break
            
            lower_u_idx = 0
            upper_u_idx = u[exp].shape[0] - 1
            for i, t in enumerate(tu[exp]):
                if t>lower_time and lower_u_idx==0:
                    lower_u_idx = i
                if t > upper_time:
                    upper_u_idx = i - 1
                    break

            # ensure that min(tu)<min(tx) and max(tx)<max(tu) s.t. range(tu) is larger than range(tx)
            # this is necessary because u is intermolated to match x
            while tu[exp][lower_u_idx] > tx[exp][lower_x_idx]:
                lower_x_idx += 1
            while tu[exp][upper_u_idx] < tx[exp][upper_x_idx]:
                upper_x_idx -= 1

            x[exp] = x[exp][lower_x_idx:upper_x_idx+1,:]
            tx[exp] = tx[exp][lower_x_idx:upper_x_idx+1]
            u[exp] = u[exp][lower_u_idx:upper_u_idx+1,:]
            tu[exp] = tu[exp][lower_u_idx:upper_u_idx+1]

            plot_lw_x_idx[exp] = lower_x_idx
            plot_lw_u_idx[exp] = lower_u_idx
            plot_up_x_idx[exp] = upper_x_idx
            plot_up_u_idx[exp] = upper_u_idx      

        if plot:
            self.plot.cropData(plot_lw_x_idx=plot_lw_x_idx, plot_up_x_idx=plot_up_x_idx)
                            
        self.data.set(names=["x", "tx", "u", "tu"], datas=[x, tx, u, tu])       

    def intermolateU(self, plot=False):
        """
        Polynomial interpolation of control input to match with x
        Args:
            plot: if True plot results
        """
        tx, u, tu = self.data.get(names=["tx", "u", "tu"])

        u_inter = {}
        for exp in self.data.exps:

            inter_fct = interpolate.interp1d(tu[exp], u[exp], axis=0)
            u_inter[exp] = inter_fct(tx[exp])

        if plot:
            self.plot.intermolateU(u_inter)

        self.data.set(names=["u"], datas=[u_inter])
        self.data.delete(names=["tu"])

    def firstOrderU(self, tau_up, tau_dw, plot=False):
        """
        Apply first order model to U
        Args:
            tau_up: time constant when accelerating, float
            tau_dw: time constant when decelerating, float
            plot: if True plot results, bool
        """
        tx, u = self.data.get(names=["tx", "u"])
        
        u_approx = {}
        for exp in self.data.exps:
        
            u_approx[exp] = np.zeros(u[exp].shape)
            for i in range(1, u[exp].shape[0]):

                tau = np.zeros((6))
                for j in range(6):
                    if u[exp][i,j] > u_approx[exp][i-1,j]:
                        tau[j] = tau_up
                    else:
                        tau[j] = tau_dw

                u_approx[exp][i,:] = u_approx[exp][i-1,:] + (tx[exp][i]-tx[exp][i-1])*(u[exp][i,:] - u_approx[exp][i-1,:])/tau     

        if plot:
            self.plot.firstOrderU(u_approx)
        self.data.set(names=["u"], datas=[u_approx])

    def diffX(self, plot=False):
        """
        Calc. smooth derivatives from x to get dx and ddx
        Args:
            plot: if True plot results
        """
        tx, x = self.data.get(names=["tx", "x"])

        fd = SINDyDerivative(d=1, kind="savitzky_golay", left=0.05, right=0.05, order=3) # specify widnow size

        dx = {}
        ddx = {}
        for exp in self.data.exps:
            dx[exp] = fd._differentiate(x[exp], tx[exp])
            ddx[exp] = fd._differentiate(dx[exp], tx[exp])

        self.data.set(names=["dx", "ddx"], datas=[dx, ddx])
        if plot:
            self.plot.diffPosition()
    
    def alignData(self, plot=False, verb=True):
        """
        Args:
            plot: if True plot results
            verb: if True print results
        """
        x, dx, ddx, u = self.data.get(names=["x", "dx", "ddx", "u"])       
        ddx_unshifted = ddx.copy()
        ddx_u_unshifted = self._calcDDX_U(x_dict=x, dx_dict=dx, u_dict=u, plot=False)

        if verb:
            print("\nStarting u shift optimization:")
        
        shift = np.Inf
        total_shift = 0
        iter = 0
        max_iter = 40
        while abs(shift)!=0 and iter<max_iter:
            ddx_u = self._calcDDX_U(x_dict=x, dx_dict=dx, u_dict=u, plot=False)

            # calc. highest cross-correlation between ddx_u and ddx data
            shift, shift_std, corr_max = self._calcCrossCorr(sig_a_dict=ddx, sig_b_dict=ddx_u)      

            # shift local data
            assert shift >= 0, "signal a (ddx) must be delayed wrt. signal b (ddx_u)"
            for exp in self.data.exps: 
                u[exp] = u[exp][:-shift,:]
                x[exp] = x[exp][shift:,:]
                dx[exp] = dx[exp][shift:,:]
                ddx[exp] = ddx[exp][shift:,:]
            total_shift += shift

            # update variables
            iter += 1
            if verb:
                print(f"iter: {iter}: \ttot. shift: {total_shift}, \titer. shift: {shift}, " \
                        + f"\tstd. shift: {np.round(shift_std, 3)}, \tcorr. max: {corr_max}")    

        # shift data
        if shift == 0:
            self.data.shift(shift=total_shift, names=["tx", "x", "dx", "ddx"])
        else:
            raise Exception(f"Finding optimal u shift failed!")

        # calc. shifted ddx_u data
        x, dx, u = self.data.get(names=["x", "dx", "u"])   
        ddx_u = self._calcDDX_U(x_dict=x, dx_dict=dx, u_dict=u, plot=False)

        if plot:
            self.plot.uShift(total_shift=total_shift, ddx_u=ddx_u, 
                             ddx_u_unshifted=ddx_u_unshifted, ddx_unshifted=ddx_unshifted)


    def _calcCrossCorr(self, sig_a_dict, sig_b_dict):
        """
        Calc. cross-correlation and determine time shift between signal a and b
        If signal a is delayed with respect to signal b (sig_a=[0,0,0,1,2,3], sig_b=[0,1,2,3,4,5]), then shift is positive.
        Args:
            sig_a_dict: dict of experiments containing signal a measurements
            sig_b_dict: dict of experiments containing signal b measurements
        Returns:
            avg_shift: average shift (nb. indices shifted) over all dimensions and experiments
            std_shift: std. of all shifts
            mean_cc_max: average cross-correlation for shift over all dimensions and experiments
        """
        sig_a_dict = sig_a_dict.copy()
        sig_b_dict = sig_b_dict.copy()

        shifts = []
        cc_maxs = []
        for exp in self.data.exps:
            for i in range(3):
                sig_b = sig_b_dict[exp][:,i]
                sig_a = sig_a_dict[exp][:,i]

                cc = signal.correlate(sig_a, sig_b, mode="full")       
                cc_argmax = np.argmax(cc)
                cc_maxs.append(cc[cc_argmax])

                shift_arr = np.arange(-len(sig_a) + 1, len(sig_b))        
                shift = shift_arr[cc_argmax]
                shifts.append(shift)

        avg_shift = np.mean(shifts)       
        avg_shift = int(np.round(avg_shift))

        return avg_shift, np.std(shifts), np.mean(cc_maxs)


    def _calcDDX_U(self, x_dict, dx_dict, u_dict, plot=False):
        """
        Estimate ddx by using the white box model and u
        Args:
            x_dict: x data, dict
            dx_dict: dx data, dict
            u_dict: u data, dict
            plot: if True plot results
        """
        x_dict = x_dict.copy()
        dx_dict = dx_dict.copy()
        u_dict = u_dict.copy()
        
        ddx_u = {}
        for exp in self.data.exps:
            X1 = torch.tensor(x_dict[exp], dtype=torch.float32)
            X2 = torch.tensor(dx_dict[exp], dtype=torch.float32)
            X = torch.concat([X1, X2], axis=1)
            U = torch.tensor(u_dict[exp], dtype=torch.float32)
            U = self.sys.polyExpandU(U)

            acc = self.model.signal2acc(U=U, X=X)
            ddx_u[exp] = acc.detach().numpy()

            if plot:
                fig, axs = plt.subplots(nrows=4, ncols=1, figsize =(8, 8)) 
                plot_range = ddx_u[exp].shape[0]

                for j in range(0,6):
                    axs[0].plot(self.tx[exp][:plot_range], u_dict[exp][:plot_range,j], label=f"signal {j}")
                # axs[0].legend()  
                axs[0].set_title(f"Signal")        
                for i in range(1,4):
                    axs[i].plot(self.tx[exp][:plot_range], ddx_u[exp][:plot_range,i-1], color="y", label="ddx estimated")
                    axs[i].plot(self.tx[exp][:plot_range], self.ddx[exp][:plot_range,i-1],  color="r", label="ddx")
                    #axs[i].plot(self.tx[exp][:plot_range], self.imu_world[exp][:plot_range,i-1],  color="c", label="imu world")
                    axs[i].legend()
                    axs[i].set_title(f"Dimension {i-1}")
                plt.show()

        return ddx_u     
