import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.differentiation import SINDyDerivative
from scipy import interpolate

class Preprocess():
    def __init__(self) -> None:

        self.series = "holohover_20221025"
        self.x = {}
        self.dx = {}
        self.ddx = {}
        self.tx = {}
        self.u = {}
        self.tu = {}

        self.D = 6
        self.M = 6


        # testing
        self.keep_nb_rows = "all"


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
    
    def stamp2seconds(self):
        """¨
        Converts time stamp to seconds for state x and control input u
        """
        for exp in self.tx:
            # get starting time stamp
            start_stamp = 0
            if self.tx[exp][0] < self.tu[exp][0]:
                start_stamp = self.tx[exp][0]
            else: 
                start_stamp = self.tu[exp][0]

            # convert tx and tu from stamp to seconds
            self.tx[exp] = (self.tx[exp]-start_stamp) / 1000000000
            self.tu[exp] = (self.tu[exp]-start_stamp) / 1000000000

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
                print(f"tu[{lower_u_idx}]: {self.tu[exp][lower_u_idx]}, tx[{lower_x_idx}]: {self.tx[exp][lower_x_idx]}")
                lower_x_idx += 1
            while self.tu[exp][upper_u_idx] < self.tx[exp][upper_x_idx]:
                print(f"tu[{upper_u_idx}]: {self.tu[exp][upper_u_idx]}, tx[{upper_x_idx}]: {self.tx[exp][upper_x_idx]}")
                upper_x_idx -= 1

            if plot:
                plt.plot(self.tx[exp], self.x[exp][:,2], )
                plt.vlines(self.tx[exp][lower_x_idx], ymin=-3.3, ymax=3.3, colors="r")
                plt.vlines(self.tx[exp][upper_x_idx], ymin=-3.3, ymax=3.3, colors="r")
                plt.show()

            self.x[exp] = self.x[exp][lower_x_idx:upper_x_idx,:]
            self.tx[exp] = self.tx[exp][lower_x_idx:upper_x_idx]
            self.u[exp] = self.u[exp][lower_u_idx:upper_u_idx,:]
            self.tu[exp] = self.tu[exp][lower_u_idx:upper_u_idx]

            print(f"tu min: {self.tu[exp][0]}, tx min: {self.tx[exp][0]}")
            print(f"tu max: {self.tu[exp][-1]}, tx max: {self.tx[exp][-1]}")
            return


    def diffPosition(self, plot=False):
        """
        Calc. smooth derivatives from x to get dx and ddx
        """
        # fd = SmoothedFiniteDifference(d=1, axis=0, smoother_kws={'window_length': 1000})
        fd = SINDyDerivative(d=1, kind="savitzky_golay", left=0.5, right=0.5, order=3)
        fd2 = SINDyDerivative(d=2, kind="savitzky_golay", left=0.5, right=0.5, order=3)
        for exp in self.x:
            self.dx[exp] = fd._differentiate(self.x[exp], self.tx[exp])
            self.ddx[exp] = fd._differentiate(self.dx[exp], self.tx[exp])

            if plot:
                fig, axs = plt.subplots(nrows=self.x[exp].shape[1], figsize =(8, 8))             
                for i, ax in enumerate(axs):
                    ax.plot(self.tx[exp], self.x[exp][:,i], color="b", label="pos")
                    ax.plot(self.tx[exp], self.dx[exp][:,i], color="g", label="vel")
                    ax.plot(self.tx[exp], self.ddx[exp][:,i], color="r", label="acc")
                    ax.set_ylim([np.min(self.ddx[exp][:,i]), np.max(self.ddx[exp][:,i])])
                    ax.legend()
                    ax.set_title(f"Dimension {i}")
                plt.show()

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
                    ax.plot(self.tx[exp], self.u[exp][:,i], color="b", label="u")
                    ax.plot(self.tx[exp], u_inter, '--', color="r", label="u inter.")
                    ax.legend()
                    ax.set_title(f"Control input {i}")
                plt.show()

            self.u[exp] = u_inter
            self.tu[exp] = [] # not used anymore
            





def main():
    pp = Preprocess()
    pp.loadData()
    pp.stamp2seconds()
    pp.cropData(plot=False)
    # pp.diffPosition(plot=True)
    pp.intermolateU(plot=False)


if __name__ == "__main__":
    main()