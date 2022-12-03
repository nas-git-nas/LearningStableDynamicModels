import os
import matplotlib.pyplot as plt
import numpy as np
from pysindy.differentiation import SINDyDerivative, SmoothedFiniteDifference
from scipy import interpolate, signal



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
        self.imu = {}
        self.imu_world = {}
        self.timu = {}
        self.force = {}
        self.tforce = {}
        # self.torque = {}
        # self.ttorque = {}

    def loadData(self):
        """
        Loop through all experiments of one series and load data:
        x: [x, y, theta], tx: time stamp of x, u: [u1, ..., u6], tu: time stamp of u
        imu: [ddx, ddy, dtheta], timu: time stamp of imu, force: [thrustx, thrusty, thrustz]
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
                    self.imu[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[2,3,4], dtype=float, skip_footer=skip_footer)
                elif f.find("thrust") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self.tforce[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[1], dtype=float, skip_footer=skip_footer)
                    self.force[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[2,3,4], dtype=float, skip_footer=skip_footer)

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
            dXnew = np.concatenate((self.dx[exp], self.imu[exp][:,0:2], self.ddx[exp][:,2:3]), axis=1)

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
            upper_imu_idx = self.imu[exp].shape[0] - 1
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
            self.imu[exp] = self.imu[exp][lower_imu_idx:upper_imu_idx+1,:]
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

            inter_fct = interpolate.interp1d(self.timu[exp], self.imu[exp], axis=0)
            imu_inter = inter_fct(self.tx[exp])

            if plot:
                fig, axs = plt.subplots(nrows=self.imu[exp].shape[1], figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.timu[exp], self.imu[exp][:,i], color="b", label="u")
                    ax.plot(self.tx[exp], imu_inter[:,i], '--', color="r", label="u inter.")
                    ax.legend()
                    ax.set_title(f"IMU {i}")
                plt.show()

            self.imu[exp] = imu_inter
            self.timu[exp] = [] # not used anymore

    def smoothIMU(self, plot=False):
        """
        Calc. smooth approx. of IMU data
        """
        for exp in self.x:

            # remove offset
            offset = np.mean(self.imu[exp], axis=0)           
            IMU_offset = self.imu[exp] #self.imu[exp] - offset
            # print(f"offset: {offset}")

            # convert body to world frame
            rotation_b2w = self.rotMatrix(self.x[exp][:,2]) # (N,S,S)
            IMU_world = np.einsum('ns,nps->np', IMU_offset, rotation_b2w) # (N, S)

            # smooth data
            IMU_smooth = IMU_world #signal.savgol_filter(IMU_world, window_length=11, polyorder=3, axis=0)

            if plot:
                fig, axs = plt.subplots(nrows=self.imu[exp].shape[1], ncols=1, figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.tx[exp], self.imu[exp][:,i], color="b", label="imu")
                    ax.plot(self.tx[exp], IMU_smooth[:,i], '--', color="g", label="smooth imu")
                    ax.hlines(offset[i], xmin=self.tx[exp][0], xmax=self.tx[exp][-1], color="r", 
                                    label=f"offset: {np.round(offset[i],2)} (body frame)")
                    ax.legend()
                    ax.set_title(f"Dimension {i}")
                plt.show()

            self.imu_world[exp] = IMU_smooth



    def diffPosition(self, plot=False):
        """
        Calc. smooth derivatives from x to get dx and ddx
        """
        # fd = SmoothedFiniteDifference(d=1, axis=0, smoother_kws={'window_length': 101})
        fd = SINDyDerivative(d=1, kind="savitzky_golay", left=0.05, right=0.05, order=3) # specify widnow size
        for exp in self.x:
            plot_range = 1000 #self.tx[exp].shape[0]
            self.dx[exp] = fd._differentiate(self.x[exp][:plot_range,:], self.tx[exp][:plot_range])
            self.ddx[exp] = fd._differentiate(self.dx[exp][:plot_range,:], self.tx[exp][:plot_range])

            if plot: 
                fig, axs = plt.subplots(nrows=self.x[exp].shape[1], ncols=2, figsize =(8, 8))             
                for i, ax in enumerate(axs[:,0]):
                    ax.plot(self.tx[exp][:plot_range], self.x[exp][:plot_range,i], color="b", label="pos")
                    ax.plot(self.tx[exp][:plot_range], self.dx[exp][:plot_range,i], color="g", label="vel")
                    ax.plot(self.tx[exp][:plot_range], self.ddx[exp][:plot_range,i], color="r", label="acc")
                    ax.plot(self.tx[exp][:plot_range], self.imu[exp][:plot_range,i], color="c", label="imu")
                    # ax.set_ylim([np.min(self.ddx[exp][:,i]), np.max(self.ddx[exp][:,i])])
                    ax.legend()
                    ax.set_title(f"Dimension {i}")

                # integrate dx and calc. error
                d_error = self.integrate(self.dx[exp][:plot_range], self.tx[exp][:plot_range], self.x[exp][0,:])
                d_error = d_error - self.x[exp][:plot_range]

                # double integrate imu and calc. error
                dd_error = self.integrate(self.ddx[exp][:plot_range], self.tx[exp][:plot_range], self.dx[exp][0,:])
                dd_error = self.integrate(dd_error, self.tx[exp][:plot_range], self.x[exp][0,:])
                dd_error = dd_error - self.x[exp][:plot_range]

                # double integrate ddx and calc. error
                imu_error = self.integrate(self.imu[exp], self.tx[exp], self.dx[exp][0,:])
                imu_error = self.integrate(imu_error, self.tx[exp], self.x[exp][0,:])
                imu_error = imu_error - self.x[exp]

                for i, ax in enumerate(axs[:,1]):
                    ax.plot(self.tx[exp][:plot_range], d_error[:plot_range,i], color="g", label="dx error")
                    ax.plot(self.tx[exp][:plot_range], dd_error[:plot_range,i], color="r", label="ddx error")
                    if i < 2:
                        ax.plot(self.tx[exp][:plot_range], imu_error[:plot_range,i], color="c", label="imu error")
                    ax.legend()
                    ax.set_title(f"Dimension {i}")                    

                plt.show()

    def crossCorrelation(self):

        delays = []
        for exp in self.x:
            for i in range(2):
                ddx = self.ddx[exp][0:1000,i]
                imu = self.imu_world[exp][0:1000,i]
                # ddx = np.array( [0,0,0,1,2,3,4,0,0])
                # imu = np.array( [0,1,2,3,4,0,0,0,0])

                cc = signal.correlate(imu, ddx, mode="full")       
                cc_argmax = np.argmax(cc)

                delay_arr = np.arange(-len(imu) + 1, len(ddx))        
                delay = delay_arr[cc_argmax]
                delays.append(delay)

        avg_delay = np.mean(delays)       
        avg_delay = int(np.round(avg_delay))
        print(f"avg. delay: {np.round(np.mean(delays),3)}, std: {np.round(np.std(delays),3)}")
        
        for exp in self.x:
            if delay > 0:
                self.tx[exp] = self.tx[exp][:-delay]
                self.x[exp] = self.x[exp][:-delay,:]
                self.dx[exp] = self.dx[exp][:-delay,:]
                self.ddx[exp] = self.ddx[exp][:-delay,:]
                self.u[exp] = self.u[exp][:-delay,:]
                self.imu[exp] = self.imu[exp][delay:,:]
            elif delay < 0:
                self.tx[exp] = self.tx[exp][-delay:] - self.tx[exp][-delay]
                self.x[exp] = self.x[exp][-delay:,:]
                self.dx[exp] = self.dx[exp][-delay:,:]
                self.ddx[exp] = self.ddx[exp][-delay:,:]
                self.u[exp] = self.u[exp][-delay:,:]
                self.imu[exp] = self.imu[exp][:delay,:]

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

            

def main():
    pp = Preprocess(series="holohover_20221130")
    pp.loadData()
    pp.stamp2seconds()
    pp.cropData(plot=False)
    pp.intermolateU(plot=False)
    pp.intermolateIMU(plot=False)
    pp.smoothIMU(plot=False)
    pp.diffPosition(plot=False)
    pp.crossCorrelation()
    pp.smoothIMU(plot=False)
    pp.crossCorrelation()
    # pp.saveData()

    """
    crup data in loadData for test subset
    split diffPosition in calc and plot parts
    write optimization fct. for shift:
        while shift > 1:
            1. calc. imu body to world
            2. calc shift
            3. update local variables
        apply shift to global variables
    """

    


if __name__ == "__main__":
    main()