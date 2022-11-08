import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pysindy.differentiation import FiniteDifference

class Preprocess():
    def __init__(self) -> None:

        self.series = "holohover_20221025"
        self.datas = {}
        self.D = 6
        self.M = 6


    def loadData(self):
        # paths of all .csv files in experiment serie
        paths = []

        # loop through all subfolders of the defined experiment serie
        for root, dirs, files in os.walk(os.path.join("experiment", self.series)):

            # loop through all files in subfolder
            for f in files:
                # skip if it is not a .csv file
                if f.find(".csv") == -1:
                    continue

                # add eperiment to data
                self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()

    def filterData(self):
        for name, data in self.datas.items():
            upper_limit = data.shape[0]

            for i, theta in enumerate(data[:,4]):
                if abs(theta) > 3.1:
                    upper_limit = i
                    break

            self.datas[name] = data[:upper_limit,:]


    def diffData(self):
        fd = FiniteDifference(d=1)
        fd2 = FiniteDifference(d=2)
        for name, data in self.datas.items():
            X_exp = np.zeros((data.shape[0], self.D))
            dX_exp = np.zeros((data.shape[0], self.D))
            U_exp = np.zeros((data.shape[0], self.M))

            X_exp[:,0] = data[:,2]
            X_exp[:,1] = data[:,3]
            X_exp[:,2] = data[:,4]
            X_exp[:,3] = fd._differentiate(data[:,2], data[:,1])
            X_exp[:,4] = fd._differentiate(data[:,3], data[:,1])            
            X_exp[:,5] = fd._differentiate(data[:,4], data[:,1])

            dX_exp[:,0] = X_exp[:,3]
            dX_exp[:,1] = X_exp[:,4]
            dX_exp[:,2] = X_exp[:,5]
            dX_exp[:,3] = fd2._differentiate(data[:,2], data[:,1])
            dX_exp[:,4] = fd2._differentiate(data[:,3], data[:,1])
            dX_exp[:,5] = fd2._differentiate(data[:,4], data[:,1])


    def plotTheta(self):

        for name, data in self.datas.items():
            thetas = data[:,4]

            prev_theta = 0
            for i, theta in enumerate(thetas):
                if abs(theta) > 3.1:
                    print(f"{i}: theta={theta}")



            print(f"max = {np.max(thetas)}")
            print(f"min = {np.min(thetas)}")

            plt.plot(thetas)
            plt.show()




def main():
    pp = Preprocess()
    pp.loadData()
    pp.plotTheta()


if __name__ == "__main__":
    main()