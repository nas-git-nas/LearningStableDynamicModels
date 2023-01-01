import os
from copy import deepcopy
import numpy as np

class Data():
    def __init__(self, series_name, crop_data=False, crop_exp=False) -> None:
        """
        Initialization of Data instance
        Args:
            series_name: name of experiment series to load, str
            crop_data: nb. of data points to keep for each experiment, if False then all data is kept, bool or int
            crop_exp: nb. of experiments to keep, if False then all data is kept, bool or int
        """
        # public variables
        self.exps = []
        self.D = 6
        self.M = 6
        self.series_name = series_name

        # private varibales
        self._x = {}
        self._dx = {}
        self._ddx = {}
        self._tx = {}
        self._u = {}
        self._tu = {}
        self._f = {}
        self._tf = {}

        self._loadData(series_name=series_name, crop_data=crop_data, crop_exp=crop_exp)
        self._stamp2seconds()

    def get(self, names):
        """
        Get privat data
        Args:
            names: list of data name, list of str
        Returns:
            datas: list of data, list of dict
        """
        datas = []
        for name in names:
            if name == "x":
                datas.append(deepcopy(self._x)) 
            if name == "dx":
                datas.append(deepcopy(self._dx))
            if name == "ddx":
                datas.append(deepcopy(self._ddx))
            if name == "tx":
                datas.append(deepcopy(self._tx))
            if name == "u":
                datas.append(deepcopy(self._u))
            if name == "tu":
                datas.append(deepcopy(self._tu))
            if name == "f":
                datas.append(deepcopy(self._f))
            if name == "tf":
                datas.append(deepcopy(self._tf))
        return datas

    def set(self, names, datas):
        """
        Set privat data
        Args:
            names: list of data name, list of str
            datas: list of data, list of dict
        """
        for name, data in zip(names, datas):
            if name == "x":
                self._x = deepcopy(data)
            if name == "dx":
                self._dx = deepcopy(data)
            if name == "ddx":
                self._ddx = deepcopy(data)
            if name == "tx":
                self._tx = deepcopy(data)
            if name == "u":
                self._u = deepcopy(data)
            if name == "tu":
                self._tu = deepcopy(data)
            if name == "f":
                self._f = deepcopy(data)
            if name == "tf":
                self._tf = deepcopy(data)

    def delete(self, names):
        """
        Delete privat data
        Args:
            names: list of data name, list of str
        """
        for name in names:
            if name == "x":
                del self._x
            if name == "dx":
                del self._dx
            if name == "ddx":
                del self._ddx
            if name == "tx":
                del self._tx
            if name == "u":
                del self._u
            if name == "tu":
                del self._tu
            if name == "f":
                del self._f
            if name == "tf":
                del self._tf

    def shift(self, shift, names):
        """
        Shift data back in time: if shift=2 and unshifted_data=[0,0,0,1,2,3] then shifted_data=[0,1,2,3]
        All data that is not shifted is cropped at the end st. all data has the same length
        Args:
            shift: nb. of indices to shift, int
            names: list of names to shif
        """
        assert shift>0, f"Shift must be greater than zero, but shift={shift}"
        not_shifted_names = ["x", "dx", "ddx", "tx", "u", "tu", "f", "tf"]

        for name in names:
            not_shifted_names.remove(name)

            for exp in self.exps:
                if name == "x":
                    self._x[exp] = self._x[exp][shift:,:]
                if name == "dx":
                    self._dx[exp] = self._dx[exp][shift:,:]
                if name == "ddx":
                    self._ddx[exp] = self._ddx[exp][shift:,:]
                if name == "tx":
                    self._tx[exp] = self._tx[exp][shift:] - self._tx[exp][shift]
                if name == "u":
                    self._u[exp] = self._u[exp][shift:,:]
                if name == "tu":
                    self._tu[exp] = self._tu[exp][shift:] - self._tu[exp][shift]
                if name == "f":
                    self._f[exp] = self._f[exp][shift:,:]
                if name == "tf":
                    self._tf[exp] = self._tf[exp][shift:] - self._tf[exp][shift]

        for name in not_shifted_names:
            if "_"+name not in self.__dict__.keys():
                continue

            for exp in self.exps:
                if name == "x" and len(self._x)>0:
                    self._x[exp] = self._x[exp][:-shift,:]
                if name == "dx" and len(self._dx)>0:
                    self._dx[exp] = self._dx[exp][:-shift,:]
                if name == "ddx" and len(self._ddx)>0:
                    self._ddx[exp] = self._ddx[exp][:-shift,:]
                if name == "tx" and len(self._tx)>0:
                    self._tx[exp] = self._tx[exp][:-shift]
                if name == "u" and len(self._u)>0:
                    self._u[exp] = self._u[exp][:-shift,:]
                if name == "tu" and len(self._tu)>0:
                    self._tu[exp] = self._tu[exp][:-shift]
                if name == "f" and len(self._f)>0:
                    self._f[exp] = self._f[exp][:-shift,:]
                if name == "tf" and len(self._tf)>0:
                    self._tf[exp] = self._tf[exp][:-shift]

    def save(self, names):
        """
        Save data defined by 'names' in csv files
            state: [x, y, theta, dx, dy, dtheta], (N, D)
            u: [u1, ..., u6], (N, M)
            thrust: [Fx, Fy, Fz], (N, D)
        Args:
            names: list of objects to save, list of str
        """
        X = np.empty((0,self.D))
        dX = np.empty((0,self.D))
        U = np.empty((0,self.M))
        tX = np.empty((0))
        for exp in self.exps:
            Xnew = np.concatenate((self._x[exp], self._dx[exp]), axis=1)
            dXnew = np.concatenate((self._dx[exp], self._ddx[exp]), axis=1)

            X = np.concatenate((X, Xnew), axis=0)
            dX = np.concatenate((dX, dXnew), axis=0)
            U = np.concatenate((U, self._u[exp]), axis=0)
            tX = np.concatenate((tX, self._tx[exp]), axis=0)

        for name in names:
            if name == "state":
                np.savetxt(os.path.join("experiment", self.series_name, "data_state.csv"), X, delimiter=",")
            if name == "dynamics":
                np.savetxt(os.path.join("experiment", self.series_name, "data_dynamics.csv"), dX, delimiter=",")
            if name == "u":
                np.savetxt(os.path.join("experiment", self.series_name, "data_input.csv"), U, delimiter=",")
            if name == "tx":
                np.savetxt(os.path.join("experiment", self.series_name, "data_time.csv"), tX, delimiter=",")

    def _loadData(self, series_name, crop_data=False, crop_exp=False):
        """
        Loop through all experiments of one series and load data:
        x: [x, y, theta], tx: time stamp of x, u: [u1, ..., u6], tu: time stamp of u
        imu: [ddx, ddy, dtheta], timu: time stamp of imu, force: [thrustx, thrusty, thrustz]
        Args:
            series_name: name of experiment series, str
            crop_data:  if crop_data=False, all data is kept, 
                        if crop_data>0, only the range [0:crop_data] is used
            crop_exp:   if crop_exp=False, all experiments are kept, 
                        if crop_exp>0, only crop_exp experiments are used
        """
        nb_exp_read = 0

        # loop through all subfolders of the defined experiment serie
        for root, dirs, files in os.walk(os.path.join("experiment", series_name)):

            # loop through all files in subfolder
            for f in files:
                # skip if it is not a .csv file
                if f.find(".csv") == -1:
                    continue

                # calc. number of rows to skip when reading the file
                skip_footer = 0
                if crop_data:
                    with open(os.path.join(root, f), "r") as f_count:
                        nb_rows = sum(1 for _ in f_count)
                    skip_footer = nb_rows-crop_data

                # add eperiment either to optitrack of control data
                if f.find("optitrack") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self._tx[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[1], dtype=float, skip_footer=skip_footer)
                    self._x[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[2,3,4], dtype=float, skip_footer=skip_footer)
                elif f.find("control") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self._tu[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[1], dtype=float, skip_footer=skip_footer)
                    self._u[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[2,3,4,5,6,7], dtype=float, skip_footer=skip_footer)
                elif f.find("thrust") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self._tf[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[1], dtype=float, skip_footer=skip_footer)
                    self._f[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[2,3,4], dtype=float, skip_footer=skip_footer)

            if crop_exp and nb_exp_read>=crop_exp:
                break
            nb_exp_read += 1

        keys = list(self._x.keys())
        keys.extend(list(self._u.keys()))
        keys.extend(list(self._f.keys()))
        self.exps = np.unique(keys)
    
    def _stamp2seconds(self):
        """¨
        Converts time stamp to seconds for state x and control input u
        """
        for exp in self._tx:
            # get starting time stamp
            try: tx_min = self._tx[exp][0] 
            except: tx_min = np.Inf
            try: tu_min = self._tu[exp][0]
            except: tu_min = np.Inf
            try: tf_min = self._tf[exp][0]
            except: tf_min = np.Inf
            
            start_stamp = np.min([tx_min, tu_min, tf_min])

            # convert tx, tu and tf from stamp to seconds
            try: self._tx[exp] = (self._tx[exp]-start_stamp) / 1000000000
            except: pass
            try: self._tu[exp] = (self._tu[exp]-start_stamp) / 1000000000
            except: pass
            try: self._tf[exp] = (self._tf[exp]-start_stamp) / 1000000000
            except: pass
