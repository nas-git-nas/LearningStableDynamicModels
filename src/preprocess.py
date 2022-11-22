import os
import matplotlib.pyplot as plt
import numpy as np
from pysindy.differentiation import SINDyDerivative
from scipy import interpolate, stats

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
        self.force = {}
        self.tforce = {}
        # self.torque = {}
        # self.ttorque = {}

    def loadData(self):
        """
        Loop through all experiments of one series and load data:
        x: [x, y, theta], tx: time stamp of x, u: [u1, ..., u6], tu: time stamp of u
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
            dXnew = np.concatenate((self.dx[exp], self.ddx[exp]), axis=1)

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
            try: tf_min = self.tforce[exp][0]
            except: tf_min = np.Inf
            start_stamp = np.min(np.array([tx_min, tu_min, tf_min]))

            # convert tx, tu and tforce from stamp to seconds
            try: self.tx[exp] = (self.tx[exp]-start_stamp) / 1000000000
            except: pass
            try: self.tu[exp] = (self.tu[exp]-start_stamp) / 1000000000
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
                if theta_max > 3.1:
                    upper_time = self.tx[exp][i]
                    break

            # determine indices where to cut lower and upper part
            lower_x_idx = 0           
            upper_x_idx = self.x[exp].shape[0] - 1
            for i, tx in enumerate(self.tx[exp]):
                if tx>lower_time and lower_x_idx==0:
                    lower_x_idx = i
                if tx > upper_time:
                    upper_x_idx = i
                    break
            
            lower_u_idx = 0
            upper_u_idx = self.u[exp].shape[0] - 1
            for i, tu in enumerate(self.tu[exp]):
                if tu>lower_time and lower_u_idx==0:
                    lower_u_idx = i
                if tu > upper_time:
                    upper_u_idx = i
                    break

            # ensure that min(tu)<min(tx) and max(tx)<max(tu) s.t. range(tu) is larger than range(tx)
            # this is necessary because u is intermolated to match x
            while self.tu[exp][lower_u_idx] > self.tx[exp][lower_x_idx]:
                lower_x_idx += 1
            while self.tu[exp][upper_u_idx] < self.tx[exp][upper_x_idx]:
                upper_x_idx -= 1

            if plot:
                plt.plot(self.tx[exp], self.x[exp][:,2], )
                plt.vlines(self.tx[exp][lower_x_idx], ymin=-3.3, ymax=3.3, colors="r")
                plt.vlines(self.tx[exp][upper_x_idx], ymin=-3.3, ymax=3.3, colors="r")
                plt.show()

            self.x[exp] = self.x[exp][lower_x_idx:upper_x_idx+1,:]
            self.tx[exp] = self.tx[exp][lower_x_idx:upper_x_idx+1]
            self.u[exp] = self.u[exp][lower_u_idx:upper_u_idx+1,:]
            self.tu[exp] = self.tu[exp][lower_u_idx:upper_u_idx+1]

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

    def diffPosition(self, plot=False):
        """
        Calc. smooth derivatives from x to get dx and ddx
        """
        # fd = SmoothedFiniteDifference(d=1, axis=0, smoother_kws={'window_length': 1000})
        fd = SINDyDerivative(d=1, kind="savitzky_golay", left=0.5, right=0.5, order=3)
        for exp in self.x:
            self.dx[exp] = fd._differentiate(self.x[exp], self.tx[exp])
            self.ddx[exp] = fd._differentiate(self.dx[exp], self.tx[exp])

            if plot:
                fig, axs = plt.subplots(nrows=self.x[exp].shape[1], ncols=2, figsize =(8, 8))             
                for i, ax in enumerate(axs[:,0]):
                    ax.plot(self.tx[exp], self.x[exp][:,i], color="b", label="pos")
                    ax.plot(self.tx[exp], self.dx[exp][:,i], color="g", label="vel")
                    ax.plot(self.tx[exp], self.ddx[exp][:,i], color="r", label="acc")
                    ax.set_ylim([np.min(self.ddx[exp][:,i]), np.max(self.ddx[exp][:,i])])
                    ax.legend()
                    ax.set_title(f"Dimension {i}")

                # integrate dx and calc. error
                d_error = self.integrate(self.dx[exp], self.tx[exp], self.x[exp][0,:])
                d_error = d_error - self.x[exp]

                # double integrate ddx and calc. error
                dd_error = self.integrate(self.ddx[exp], self.tx[exp], self.dx[exp][0,:])
                dd_error = self.integrate(dd_error, self.tx[exp], self.x[exp][0,:])
                dd_error = dd_error - self.x[exp]

                for i, ax in enumerate(axs[:,1]):
                    ax.plot(self.tx[exp], d_error[:,i], color="g", label="dx error")
                    ax.plot(self.tx[exp], dd_error[:,i], color="r", label="ddx error")
                    ax.legend()
                    ax.set_title(f"Dimension {i}")                    

                plt.show()

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

    def getThrust(self, plot=False):
        # dict for signals and motors
        thrusts = {}
        for i in range(6):
            thrusts[i] = {}


        # loop through all experiments        
        for exp in self.force:

            signal_state_prev = True
            signal_time_start = 0 
            mot = 0
            sig = 0
            for i, (tu, u) in enumerate(zip(self.tu[exp], self.u[exp])):
                
                # check if currently a signal is send
                if np.sum(u) > 0:
                    signal_state = True
                else: 
                    signal_state = False

                # check if signal state toggled
                if signal_state is not signal_state_prev or i==len(self.tu[exp])-1:
                    # evaluate start and stop indices of signal
                    assert signal_time_start < tu
                    idx_start =  np.argmax(self.tforce[exp]>signal_time_start)
                    idx_stop = np.argmax(self.tforce[exp]>=tu)

                    # calc. signal mean and standard deviation
                    if signal_state_prev:
                        force = self.force[exp][idx_start:idx_stop,:]
                        time = self.tforce[exp][idx_start:idx_stop]
                        mot = np.argmax(self.u[exp][i-1])
                        sig = np.max(self.u[exp][i-1])
                        thrusts[mot][sig] = {   "idx_start":idx_start, "idx_stop":idx_stop, "time":time, "force":force, 
                                                "bg":None, "norm":None, "mean":None, "std":None }
                    else: 
                        thrusts[mot][sig]["bg"] = self.force[exp][idx_start:idx_stop,:]
                    
                    # set starting time of signal while removing 0.5s
                    signal_time_start = tu + 0.5
                
                    # update previous state
                    signal_state_prev = signal_state

        for exp in self.force:
            if plot:
                fig, axs = plt.subplots(nrows=3, ncols=2, figsize =(12, 8)) 
                axs[0,0].plot(self.tforce[exp], self.force[exp][:,0], color="b", label="Thrust x")
                axs[0,1].plot(self.tforce[exp], self.force[exp][:,1], color="g", label="Thrust y")
                axs[0,0].set_title(f"Thrust x")
                axs[0,1].set_title(f"Thrust y")
                axs[1,0].set_title(f"Thrust x (offset removed)")   
                axs[1,1].set_title(f"Thrust y (offset removed)")
                axs[2,0].set_title(f"Thrust norm (offset removed)")

            for mot in thrusts:
                for sig in thrusts[mot]:
                    bg_x = np.mean(thrusts[mot][sig]["bg"][:,0])
                    bg_y = np.mean(thrusts[mot][sig]["bg"][:,1])
                    bg_idx = thrusts[mot][sig]["idx_stop"]
                    

                    thrusts[mot][sig]["force"][:,0] = thrusts[mot][sig]["force"][:,0] - bg_x
                    thrusts[mot][sig]["force"][:,1] = thrusts[mot][sig]["force"][:,1] - bg_y
                    thrusts[mot][sig]["norm"] = np.sqrt(np.power(thrusts[mot][sig]["force"][:,0], 2) + np.power(thrusts[mot][sig]["force"][:,1], 2))
                    thrusts[mot][sig]["mean"] = np.mean(thrusts[mot][sig]["norm"])
                    thrusts[mot][sig]["std"] = np.std(thrusts[mot][sig]["norm"])
            
                    if plot:
                        axs[0,0].scatter(self.tforce[exp][bg_idx], bg_x, color="r")
                        axs[0,1].scatter(self.tforce[exp][bg_idx], bg_y, color="r")
                        axs[1,0].plot(thrusts[mot][sig]["time"], thrusts[mot][sig]["force"][:,0], color="b", label="Thrust x")                    
                        axs[1,1].plot(thrusts[mot][sig]["time"], thrusts[mot][sig]["force"][:,1], color="g", label="Thrust y")
                        axs[2,0].plot(thrusts[mot][sig]["time"], thrusts[mot][sig]["norm"], color="b", label="Thrust norm")

            if plot:
                plt.show()
        if plot:
            for exp in self.force:
                fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(8, 8))             
                for i, mot in enumerate(thrusts):
                    k = int(i/len(axs[0]))
                    l = i % len(axs[0])

                    for sig in thrusts[mot]:
                        axs[k,l].scatter(sig, thrusts[mot][sig]["mean"], color="b", label="Mean thrust")
                        axs[k,l].errorbar(sig, thrusts[mot][sig]["mean"], thrusts[mot][sig]["std"], color="r", fmt='.k')
                plt.show()

        return thrusts

    def approxThrust(self, thrusts, plot=False):
        
        for mot in thrusts:
            signal = []
            thrust = []
            for sig in thrusts[mot]:
                signal.append(sig)
                thrust.append(thrusts[mot][sig]["mean"])

            fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(14, 8))
            for j, deg in enumerate(range(1,7)):
                k = int(j/len(axs[0]))
                l = j % len(axs[0])

                coeff = np.polyfit(np.array(signal, dtype=float), np.array(thrust, dtype=float), deg=deg)

                lin_x = np.linspace(0, 1, 100)
                lin_X = np.zeros((100,deg+1))
                for i in range(lin_X.shape[1]):
                    lin_X[:,i] = np.power(lin_x, i)
                thrust_approx = lin_X @ np.flip(coeff)

                stat_x = np.array(signal, dtype=float)
                stat_X = np.zeros((len(signal),deg+1))
                for i in range(stat_X.shape[1]):
                    stat_X[:,i] = np.power(stat_x, i)
                stat_approx = stat_X @ np.flip(coeff)
                _, _, rvalue, _, _ = stats.linregress(stat_approx, np.array(thrust, dtype=float))

                axs[k,l].scatter(signal, thrust, color="b", label="Meas.")
                axs[k,l].plot(lin_x, thrust_approx, color="g", label="Approx.")
                axs[k,l].set_title(f"Deg={deg} (R^2={np.round(rvalue**2, 4)})")
                axs[k,l].set_xlabel("Signal")
                axs[k,l].set_ylabel("Thrust")
                axs[k,l].legend()


            plt.show()





            

def main_experiment():
    pp = Preprocess(series="holohover_20221025")
    pp.loadData()
    pp.stamp2seconds()
    pp.cropData(plot=True)
    pp.intermolateU(plot=True)
    pp.diffPosition(plot=True)
    pp.saveData()

def main_signal2thrust():
    pp = Preprocess(series="signal_20221121")
    pp.loadData()
    pp.stamp2seconds() 
    thrusts = pp.getThrust(plot=False)
    pp.approxThrust(thrusts, plot=True)

if __name__ == "__main__":
    #main_experiment()
    main_signal2thrust()