import os
import matplotlib.pyplot as plt
import numpy as np
from pysindy.differentiation import SINDyDerivative, SmoothedFiniteDifference
from scipy import interpolate, signal
import torch

from src.args import Args
from src.model_grey import HolohoverModelGrey
from src.system import HolohoverSystem


class Preprocess():
    def __init__(self, series) -> None:
        # experiment
        self.series = series 
        self.D = 6
        self.M = 6

        # testing
        self.keep_nb_rows = "all"

        # data
        self.x = {}
        self.dx = {}
        self.ddx = {}
        self.tx = {}
        self.u = {}
        self.tu = {}
        self.imu_body = {}
        self.imu_world = {}
        self.timu = {}
        self.force = {}
        self.tforce = {}
        self.ddx_u = {}
        # self.torque = {}
        # self.ttorque = {}

        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"
        device = torch.device(dev) 
        args = Args(model_type="HolohoverGrey")
        self.model = HolohoverModelGrey(args=args, dev=device)
        self.sys = HolohoverSystem(args=args, dev=device)

    def loadData(self, crop_data=False, crop_exp=False):
        """
        Loop through all experiments of one series and load data:
        x: [x, y, theta], tx: time stamp of x, u: [u1, ..., u6], tu: time stamp of u
        imu: [ddx, ddy, dtheta], timu: time stamp of imu, force: [thrustx, thrusty, thrustz]
        Args:
            crop_data:  if crop_data=False, all data is kept, 
                        if crop_data>0, only the range [0:crop_data] is used
            crop_exp:   if crop_exp=False, all experiments are kept, 
                        if crop_exp>0, only crop_exp experiments are used
        """
        # loop through all subfolders of the defined experiment serie
        for root, dirs, files in os.walk(os.path.join("experiment", self.series)):

            # loop through all files in subfolder
            for f in files:
                # skip if it is not a .csv file
                if f.find(".csv") == -1:
                    continue

                # calc. number of rows to skip when reading the file
                with open(os.path.join(root, f), "r") as f_count:
                    nb_rows = sum(1 for _ in f_count)
                if self.keep_nb_rows == "all":
                    skip_footer = 0
                else:
                    skip_footer = nb_rows-self.keep_nb_rows

                # add eperiment either to optitrack of control data
                if f.find("optitrack") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self.tx[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[1], dtype=float, skip_footer=skip_footer)
                    self.x[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[2,3,4], dtype=float, skip_footer=skip_footer)
                elif f.find("control") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self.tu[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[1], dtype=float, skip_footer=skip_footer)
                    self.u[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[2,3,4,5,6,7], dtype=float, skip_footer=skip_footer)
                elif f.find("imu") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self.timu[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[1], dtype=float, skip_footer=skip_footer)
                    self.imu_body[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[2,3,4], dtype=float, skip_footer=skip_footer)
                elif f.find("thrust") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self.tforce[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[1], dtype=float, skip_footer=skip_footer)
                    self.force[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[2,3,4], dtype=float, skip_footer=skip_footer)

        if crop_data:
            for exp in self.tx:
                self.x[exp] = self.x[exp][0:crop_data,:]
                self.tx[exp] = self.tx[exp][0:crop_data]
                self.u[exp] = self.u[exp][0:crop_data,:]
                self.tu[exp] = self.tu[exp][0:crop_data]
                self.imu_body[exp] = self.imu_body[exp][0:crop_data,:]
                self.timu[exp] = self.timu[exp][0:crop_data]
                if len(self.force[exp]) > 0:
                    self.force[exp] = self.force[exp][0:crop_data,:]
                    self.tforce[exp] = self.tforce[exp][0:crop_data]

        if crop_exp:
            exp_list = list(self.tx.keys())
            remove_list = exp_list[crop_exp:]
            for exp in exp_list:
                if exp in remove_list:
                    del self.x[exp]
                    del self.tx[exp]
                    del self.u[exp]
                    del self.tu[exp]
                    del self.imu_body[exp]
                    del self.timu[exp]
                    del self.force[exp]
                    del self.tforce[exp]

    def saveData(self):
        """
        Save state and control input vectors in csv file
            state: [x, y, theta, dx, dy, dtheta], (N, D)
            u: [u1, ..., u6], (N, M)
        """
        X = np.empty((0,self.D))
        dX = np.empty((0,self.D))
        U = np.empty((0,self.M))
        for exp in self.tx:
            Xnew = np.concatenate((self.x[exp], self.dx[exp]), axis=1)
            dXnew = np.concatenate((self.dx[exp], self.imu_world[exp][:,0:2], self.ddx[exp][:,2:3]), axis=1)

            X = np.concatenate((X, Xnew), axis=0)
            dX = np.concatenate((dX, dXnew), axis=0)
            U = np.concatenate((U, self.u[exp]), axis=0)

        np.savetxt(os.path.join("experiment", self.series, "data_state.csv"), X, delimiter=",")
        np.savetxt(os.path.join("experiment", self.series, "data_dynamics.csv"), dX, delimiter=",")
        np.savetxt(os.path.join("experiment", self.series, "data_input.csv"), U, delimiter=",")
    
    def stamp2seconds(self):
        """¨
        Converts time stamp to seconds for state x and control input u
        """
        for exp in self.tx:
            # get starting time stamp
            try: tx_min = self.tx[exp][0] 
            except: tx_min = np.Inf
            try: tu_min = self.tu[exp][0]
            except: tu_min = np.Inf
            try: timu_min = self.timu[exp][0]
            except: timu_min = np.Inf
            try: tf_min = self.tforce[exp][0]
            except: tf_min = np.Inf
            
            start_stamp = np.min(np.array([tx_min, tu_min, timu_min, tf_min]))

            # convert tx, tu and tforce from stamp to seconds
            try: self.tx[exp] = (self.tx[exp]-start_stamp) / 1000000000
            except: pass
            try: self.tu[exp] = (self.tu[exp]-start_stamp) / 1000000000
            except: pass
            try: self.timu[exp] = (self.timu[exp]-start_stamp) / 1000000000
            except: pass
            try: self.tforce[exp] = (self.tforce[exp]-start_stamp) / 1000000000
            except: pass
            

    def cropData(self, plot=False):
        """
        Remove data in the beginning and the end
        """
        for exp in self.x:

            # determine time where to cut lower and upper part
            lower_time = np.minimum(self.tx[exp][0], self.tu[exp][0])
            for i, umax in enumerate(np.max(np.abs(self.u[exp]), axis=1)):
                if umax > 0.05:
                    lower_time = self.tu[exp][i]
                    break
            
            upper_time = np.maximum(self.tx[exp][-1], self.tu[exp][-1])
            for i, theta_max in enumerate(np.abs(self.x[exp][:,2])):
                if theta_max > 3.0:
                    upper_time = self.tx[exp][i]
                    break

            # determine indices where to cut lower and upper part
            lower_x_idx = 0           
            upper_x_idx = self.x[exp].shape[0] - 1
            for i, tx in enumerate(self.tx[exp]):
                if tx>lower_time and lower_x_idx==0:
                    lower_x_idx = i
                if tx > upper_time:
                    upper_x_idx = i - 1
                    break
            
            lower_u_idx = 0
            upper_u_idx = self.u[exp].shape[0] - 1
            for i, tu in enumerate(self.tu[exp]):
                if tu>lower_time and lower_u_idx==0:
                    lower_u_idx = i
                if tu > upper_time:
                    upper_u_idx = i - 1
                    break

            lower_imu_idx = 0
            upper_imu_idx = self.imu_body[exp].shape[0] - 1
            for i, timu in enumerate(self.timu[exp]):
                if timu>lower_time and lower_imu_idx==0:
                    lower_imu_idx = i
                if timu > upper_time:
                    upper_imu_idx = i - 1
                    break

            # ensure that min(tu)<min(tx) and max(tx)<max(tu) s.t. range(tu) is larger than range(tx)
            # this is necessary because u is intermolated to match x
            while self.tu[exp][lower_u_idx] > self.tx[exp][lower_x_idx]:
                lower_x_idx += 1
            while self.tu[exp][upper_u_idx] < self.tx[exp][upper_x_idx]:
                upper_x_idx -= 1

            # ensure that min(timu)<min(tx) and max(tx)<max(timu) s.t. range(timu) is larger than range(tx)
            # this is necessary because u is intermolated to match x
            while self.timu[exp][lower_imu_idx] > self.tx[exp][lower_x_idx]:
                lower_x_idx += 1
            while self.timu[exp][upper_imu_idx] < self.tx[exp][upper_x_idx]:
                upper_x_idx -= 1

            if plot:
                plt.plot(self.tx[exp], self.x[exp][:,2], )
                plt.vlines(self.tx[exp][lower_x_idx], ymin=-3.3, ymax=3.3, colors="r")
                plt.vlines(self.tx[exp][upper_x_idx], ymin=-3.3, ymax=3.3, colors="r")
                plt.title(f"Theta of exp. {exp}")
                plt.show()

            self.x[exp] = self.x[exp][lower_x_idx:upper_x_idx+1,:]
            self.tx[exp] = self.tx[exp][lower_x_idx:upper_x_idx+1]
            self.u[exp] = self.u[exp][lower_u_idx:upper_u_idx+1,:]
            self.tu[exp] = self.tu[exp][lower_u_idx:upper_u_idx+1]
            self.imu_body[exp] = self.imu_body[exp][lower_imu_idx:upper_imu_idx+1,:]
            self.timu[exp] = self.timu[exp][lower_imu_idx:upper_imu_idx+1]

    def intermolateU(self, plot=False):
        """
        Polynomial interpolation of control input to match with x
        """
        for exp in self.u:
            print(f"tu min: {self.tu[exp][0]}, tx min: {self.tx[exp][0]}")
            print(f"tu max: {self.tu[exp][-1]}, tx max: {self.tx[exp][-1]}")

            inter_fct = interpolate.interp1d(self.tu[exp], self.u[exp], axis=0)
            u_inter = inter_fct(self.tx[exp])

            if plot:
                fig, axs = plt.subplots(nrows=self.u[exp].shape[1], figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.tu[exp], self.u[exp][:,i], color="b", label="u")
                    ax.plot(self.tx[exp], u_inter[:,i], '--', color="r", label="u inter.")
                    ax.legend()
                    ax.set_title(f"Control input {i}")
                plt.show()

            self.u[exp] = u_inter
            self.tu[exp] = [] # not used anymore

    def intermolateIMU(self, plot=False):
        """
        Polynomial interpolation of imu data to match with x
        """
        for exp in self.u:
            print(f"timu min: {self.timu[exp][0]}, tx min: {self.tx[exp][0]}")
            print(f"timu max: {self.timu[exp][-1]}, tx max: {self.tx[exp][-1]}")

            inter_fct = interpolate.interp1d(self.timu[exp], self.imu_body[exp], axis=0)
            imu_inter = inter_fct(self.tx[exp])

            if plot:
                fig, axs = plt.subplots(nrows=self.imu_body[exp].shape[1], figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.timu[exp], self.imu_body[exp][:,i], color="b", label="u")
                    ax.plot(self.tx[exp], imu_inter[:,i], '--', color="r", label="u inter.")
                    ax.legend()
                    ax.set_title(f"IMU {i}")
                plt.show()

            self.imu_body[exp] = imu_inter
            self.timu[exp] = [] # not used anymore

    def diffPosition(self, plot=False):
        """
        Calc. smooth derivatives from x to get dx and ddx
        """
        # fd = SmoothedFiniteDifference(d=1, axis=0, smoother_kws={'window_length': 101})
        fd = SINDyDerivative(d=1, kind="savitzky_golay", left=0.05, right=0.05, order=3) # specify widnow size
        for exp in self.x:
            self.dx[exp] = fd._differentiate(self.x[exp], self.tx[exp])
            self.ddx[exp] = fd._differentiate(self.dx[exp], self.tx[exp])

        if plot:
            self.diffPositionPlot()

    def diffPositionPlot(self):
        for exp in self.x:
            fig, axs = plt.subplots(nrows=self.x[exp].shape[1], ncols=2, figsize =(8, 8))             
            for i, ax in enumerate(axs[:,0]):
                ax.plot(self.tx[exp], self.x[exp], color="b", label="pos")
                ax.plot(self.tx[exp], self.dx[exp], color="g", label="vel")
                ax.plot(self.tx[exp], self.ddx[exp], color="r", label="acc")
                ax.plot(self.tx[exp], self.imu[exp], color="c", label="imu")
                # ax.set_ylim([np.min(self.ddx[exp][:,i]), np.max(self.ddx[exp][:,i])])
                ax.legend()
                ax.set_title(f"Dimension {i}")

            # integrate dx and calc. error
            d_error = self.integrate(self.dx[exp], self.tx[exp], self.x[exp][0,:])
            d_error = d_error - self.x[exp]

            # double integrate imu and calc. error
            dd_error = self.integrate(self.ddx[exp], self.tx[exp], self.dx[exp][0,:])
            dd_error = self.integrate(dd_error, self.tx[exp], self.x[exp][0,:])
            dd_error = dd_error - self.x[exp]

            # double integrate ddx and calc. error
            imu_error = self.integrate(self.imu[exp], self.tx[exp], self.dx[exp][0,:])
            imu_error = self.integrate(imu_error, self.tx[exp], self.x[exp][0,:])
            imu_error = imu_error - self.x[exp]

            for i, ax in enumerate(axs[:,1]):
                ax.plot(self.tx[exp], d_error[:,i], color="g", label="dx error")
                ax.plot(self.tx[exp], dd_error[:,i], color="r", label="ddx error")
                if i < 2:
                    ax.plot(self.tx[exp], imu_error[:,i], color="c", label="imu error")
                ax.legend()
                ax.set_title(f"Dimension {i}")                    

            plt.show()

    def imuRemoveOffset(self, plot=False):
        """
        Calc. offset of imu data and remove it
        """
        imu_body = self.imu_body.copy()

        imu_offset_free = {}
        for exp in self.x:

            # remove offset
            offset = np.mean(imu_body[exp], axis=0)           
            imu_offset_free[exp] = imu_body[exp] - offset
            # print(f"offset: {offset}")

            if plot:
                fig, axs = plt.subplots(nrows=imu_body[exp].shape[1], ncols=1, figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.tx[exp], imu_body[exp][:,i], color="b", label="imu")
                    ax.plot(self.tx[exp], imu_offset_free[exp][:,i], '--', color="g", label="smooth imu")
                    ax.hlines(offset[i], xmin=self.tx[exp][0], xmax=self.tx[exp][-1], color="r", 
                                    label=f"offset: {np.round(offset[i],2)} (body frame)")
                    ax.legend()
                    ax.set_title(f"Dimension {i}")
                plt.show()

        self.imu_body = imu_offset_free

    def imuBody2world(self, imu_body, x):
        """
        Calc. IMU in world frame
        """
        imu_body = imu_body.copy()
        x = x.copy()

        imu_world = {}
        for exp in x:

            # convert body to world frame
            rotation_b2w = self.rotMatrix(x[exp][:,2]) # (N,S,S)
            imu_world[exp] = np.einsum('ns,nps->np', imu_body[exp], rotation_b2w) # (N, S)

        return imu_world 

    def imuShift(self, plot=False):
        x = self.x.copy()
        ddx = self.ddx.copy()
        imu_body = self.imu_body.copy()

        print("\nStarting imu shift optimization:")
        shift = np.Inf
        total_shift = 0
        iter = 0
        max_iter = 40
        while abs(shift)!=0 and iter<max_iter:
            # convert imu data from body to world frame using shifted theta data
            imu_world = self.imuBody2world(imu_body=imu_body, x=x)

            # calc. highest cross-correlation between imu and ddx data
            shift, shift_std, corr_max = self.calcCrossCorr(sig_a_dict=imu_world, sig_b_dict=ddx)      

            # shift local data
            step = abs(shift)
            for exp in x:
                if shift > 0:
                    x[exp] = x[exp][:-step,:]
                    ddx[exp] = ddx[exp][:-step,:]
                    imu_body[exp] = imu_body[exp][step:,:]
                    total_shift += step
                elif shift < 0:
                    x[exp] = x[exp][step:,:]
                    ddx[exp] = ddx[exp][step:,:]
                    imu_body[exp] = imu_body[exp][:-step,:]
                    total_shift -= step

            # update variables
            iter += 1
            print(f"iter: {iter}: \ttot. shift: {total_shift}, \titer. shift: {shift}, \tstd. shift: {np.round(shift_std, 3)}, \tcorr. max: {corr_max}")

        # calc. unshifted imu_world data
        imu_world_unshifted = self.imuBody2world(imu_body=self.imu_body, x=self.x)

        # shift data
        if shift == 0:
            self.shiftIMUData(shift=total_shift)
        else:
            print(f"Finding optimal imu shift failed!")

        # calc. shifted imu_world data
        self.imu_world = self.imuBody2world(imu_body=self.imu_body, x=self.x)

        if plot:
            for exp in self.x:
                fig, axs = plt.subplots(nrows=imu_body[exp].shape[1], ncols=1, figsize =(8, 8))     

                delay_s = total_shift * (self.tx[exp][-1] - self.tx[exp][0]) / self.tx[exp].shape[0]
                fig.suptitle(f'Avg. delay of imu wrt. optitrack: {np.round(delay_s*1000,1)}ms ({total_shift} shifts)')        
                for i, ax in enumerate(axs):
                    ax.plot(self.tx[exp], imu_world_unshifted[exp][:self.ddx[exp].shape[0],i], '--', color="b", label="imu unshifted")
                    ax.plot(self.tx[exp], self.imu_world[exp][:,i], color="c", label="imu shifted")
                    ax.plot(self.tx[exp], self.ddx[exp][:,i],  color="r", label="ddx")
                    ax.legend()
                    ax.set_title(f"Dimension {i}")
                plt.show()

    def uShift(self, plot=False):
        x = self.x.copy()
        u = self.u.copy()
        imu_world = self.imu_world.copy()

        print("\nStarting u shift optimization:")
        shift = np.Inf
        total_shift = 0
        iter = 0
        max_iter = 40
        while abs(shift)!=0 and iter<max_iter:
            # convert imu data from body to world frame using shifted theta data
            ddx_u = self.calcDDX_U(x_dict=x, u_dict=u, plot=False)

            # calc. highest cross-correlation between imu and ddx data
            shift, shift_std, corr_max = self.calcCrossCorr(sig_a_dict=imu_world, sig_b_dict=ddx_u)      

            # shift local data
            step = abs(shift)
            for exp in x:
                if shift > 0:   
                    u[exp] = u[exp][:-step,:]
                    x[exp] = x[exp][step:,:]
                    imu_world[exp] = imu_world[exp][step:,:]
                    total_shift += step
                elif shift < 0:   
                    u[exp] = u[exp][step:,:]
                    x[exp] = x[exp][:-step,:]
                    imu_world[exp] = imu_world[exp][:-step,:]
                    total_shift -= step

            # update variables
            iter += 1
            print(f"iter: {iter}: \ttot. shift: {total_shift}, \titer. shift: {shift}, \tstd. shift: {np.round(shift_std, 3)}, \tcorr. max: {corr_max}")

        # calc. unshifted imu_world data
        imu_world_unshifted = self.imu_world.copy()

        # shift data
        if shift == 0:
            self.shiftUData(shift=total_shift)
        else:
            print(f"Finding optimal imu shift failed!")

        # calc. shifted ddx_u data
        self.ddx_u = self.calcDDX_U(x_dict=self.x, u_dict=self.u, plot=False)

        if plot:

            for exp in self.x:
                fig, axs = plt.subplots(nrows=self.x[exp].shape[1], ncols=1, figsize =(8, 8))   

                delay_s = total_shift * (self.tx[exp][-1] - self.tx[exp][0]) / self.tx[exp].shape[0]
                fig.suptitle(f'Avg. delay of imu wrt. control: {np.round(delay_s*1000)}ms ({total_shift} shifts)')
                for i, ax in enumerate(axs):
                    ax.plot(self.tx[exp], imu_world_unshifted[exp][:self.imu_world[exp].shape[0],i], '--', color="b", label="imu unshifted")
                    ax.plot(self.tx[exp], self.imu_world[exp][:,i], color="c", label="imu shifted")
                    ax.plot(self.tx[exp], self.ddx_u[exp][:,i],  color="r", label="estimated ddx")
                    ax.legend()
                    ax.set_title(f"Dimension {i}")
                plt.show()


    def calcCrossCorr(self, sig_a_dict, sig_b_dict):
        sig_a_dict = sig_a_dict.copy()
        sig_b_dict = sig_b_dict.copy()

        shifts = []
        cc_maxs = []
        for exp in sig_a_dict:
            for i in range(2):
                sig_b = sig_b_dict[exp][:,i]
                sig_a = sig_a_dict[exp][:,i]
                # ddx = np.array( [0,1,2,3,4,0,0,0,0])
                # imu = np.array( [0,0,0,1,2,3,4,0,0])

                cc = signal.correlate(sig_a, sig_b, mode="full")       
                cc_argmax = np.argmax(cc)
                cc_maxs = cc[cc_argmax]

                shift_arr = np.arange(-len(sig_a) + 1, len(sig_b))        
                shift = shift_arr[cc_argmax]
                shifts.append(shift)

        avg_shift = np.mean(shifts)       
        avg_shift = int(np.round(avg_shift))

        return avg_shift, np.std(shifts), np.mean(cc_maxs)
        
    def shiftIMUData(self, shift):
        for exp in self.x:
            if shift > 0:
                self.tx[exp] = self.tx[exp][:-shift]
                self.x[exp] = self.x[exp][:-shift,:]
                self.dx[exp] = self.dx[exp][:-shift,:]
                self.ddx[exp] = self.ddx[exp][:-shift,:]
                self.u[exp] = self.u[exp][:-shift,:]
                self.imu_body[exp] = self.imu_body[exp][shift:,:]
            elif shift < 0:
                self.tx[exp] = self.tx[exp][-shift:] - self.tx[exp][-shift]
                self.x[exp] = self.x[exp][-shift:,:]
                self.dx[exp] = self.dx[exp][-shift:,:]
                self.ddx[exp] = self.ddx[exp][-shift:,:]
                self.u[exp] = self.u[exp][-shift:,:]
                self.imu_body[exp] = self.imu_body[exp][:shift,:]

    def shiftUData(self, shift):
        for exp in self.x:
            if shift > 0:
                self.u[exp] = self.u[exp][:-shift,:]
                self.tx[exp] = self.tx[exp][shift:] - self.tx[exp][shift]
                self.x[exp] = self.x[exp][shift:,:]
                self.dx[exp] = self.dx[exp][shift:,:]
                self.ddx[exp] = self.ddx[exp][shift:,:]
                self.imu_world[exp] = self.imu_world[exp][shift:,:]
                self.imu_body[exp] = self.imu_body[exp][shift:,:]
            elif shift < 0: 
                self.u[exp] = self.u[exp][-shift:,:]
                self.tx[exp] = self.tx[exp][:shift] 
                self.x[exp] = self.x[exp][:shift,:]
                self.dx[exp] = self.dx[exp][:shift,:]
                self.ddx[exp] = self.ddx[exp][:shift,:]
                self.imu_world[exp] = self.imu_world[exp][:shift,:]
                self.imu_body[exp] = self.imu_body[exp][:shift,:]

    def integrate(self, dX, tX, X0):
        """
        Integrate dX
        Args:
            dX: derivative of X, (N, D)
            tX: time in seconds of X, (N)
            X0: initial condition of X, (D)
        Returns:
            X: integration of dX (N, D)
        """
        dX = np.copy(dX)
        X = np.zeros(dX.shape)
        X[0,:] = X0
        for i in range(1,dX.shape[0]):
            X[i,:] = X[i-1,:] + dX[i,:]*(tX[i]-tX[i-1])
        
        return X

    def rotMatrix(self, theta):
        """
        Calc. 3D rotational matrix for batch
        Args:
            theta: rotation aroung z-axis, tensor (N)
        Returns:
            rot_mat: rotational matrix, tensor (N,S,S)
        """
        rot_mat = np.zeros((theta.shape[0], 3, 3))
        cos = np.cos(theta) # (N)
        sin = np.sin(theta) # (N)
        rot_mat[:,0,0] = cos
        rot_mat[:,1,1] = cos
        rot_mat[:,0,1] = -sin
        rot_mat[:,1,0] = sin
        rot_mat[:,2,2] = np.ones(theta.shape[0])
        return rot_mat

    def calcDDX_U(self, x_dict, u_dict, plot=False):
        x_dict = x_dict.copy()
        u_dict = u_dict.copy()
        
        ddx_u = {}
        for exp in self.x:
            
            X = torch.tensor(x_dict[exp], dtype=torch.float32)
            U = torch.tensor(u_dict[exp], dtype=torch.float32)
            U = self.sys.polyExpandU(U)

            acc = self.model.signal2acc(U=U, X=X)
            ddx_u[exp] = acc.detach().numpy()

            if plot:
                fig, axs = plt.subplots(nrows=4, ncols=1, figsize =(8, 8)) 
                plot_range = 250  

                for j in range(0,6):
                    axs[0].plot(self.tx[exp][:plot_range], u_dict[exp][:plot_range,j], label=f"signal {j}")
                # axs[0].legend()  
                axs[0].set_title(f"Signal")        
                for i in range(1,4):
                    axs[i].plot(self.tx[exp][:plot_range], ddx_u[exp][:plot_range,i-1], color="y", label="ddx estimated")
                    axs[i].plot(self.tx[exp][:plot_range], self.ddx[exp][:plot_range,i-1],  color="r", label="ddx")
                    axs[i].plot(self.tx[exp][:plot_range], self.imu_world[exp][:plot_range,i-1],  color="c", label="imu world")
                    axs[i].legend()
                    axs[i].set_title(f"Dimension {i-1}")
                plt.show()

        return ddx_u     

    def smoothDDX_U(self, plot=False):
        ddx_u = self.calcDDX_U(x_dict=self.x, u_dict=self.u, plot=False)

        for exp in self.x:
            ddx_u_smooth = signal.savgol_filter(ddx_u[exp], window_length=55, polyorder=3, axis=0)    

            if plot:
                fig, axs = plt.subplots(nrows=self.x[exp].shape[1], ncols=1, figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.tx[exp], self.u[exp][:,i], color="b", label="signal")
                    ax.plot(self.tx[exp], ddx_u_smooth[:,i], color="y", label="smoothed ddx estimated")
                    ax.plot(self.tx[exp], self.ddx[exp][:,i],  color="r", label="ddx")
                    ax.plot(self.tx[exp], self.imu_world[exp][:,i],  color="c", label="imu world")
                    ax.legend()
                    ax.set_title(f"Dimension {i}")
                plt.show()

def main():
    pp = Preprocess(series="signal_20221206")
    pp.loadData(crop_data=None, crop_exp=None)
    pp.stamp2seconds()
    pp.cropData(plot=False)
    pp.intermolateU(plot=False)
    pp.intermolateIMU(plot=False)
    pp.diffPosition(plot=False)

    # pp.imuRemoveOffset(plot=False)
    # pp.imuShift(plot=False)

    # pp.uShift(plot=False)
    # # pp.smoothDDX_U(plot=True)

    # ddx_u = pp.calcDDX_U(x_dict=pp.x, u_dict=pp.u, plot=True)

    pp.saveData()

    


if __name__ == "__main__":
    main()