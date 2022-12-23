import os
import numpy as np



class Data():
    def __init__(self, series_name, crop_data=False, crop_exp=False) -> None:
        # experiment
        self.series_name = series_name 

        # data
        self._x = {}
        self._dx = {}
        self._ddx = {}
        self._tx = {}
        self._u = {}
        self._tu = {}
        self._imu_body = {}
        self._imu_world = {}
        self._timu = {}
        self._force = {}
        self._tforce = {}
        self._ddx_u = {}

        self._loadData(crop_data, crop_exp)
        self._stamp2seconds()

    def get(self, name):
        """
        Get privat data
        Args:
            name: data name, str
        Returns:
            data: data, dict
        """
        if name == "x":
            return self._x
        if name == "dx":
            return self._dx
        if name == "dx":
            return self._ddx
        if name == "dx":
            return self._tx
        if name == "dx":
            return self._u
        if name == "dx":
            return self._tu
        if name == "dx":
            return self._imu_body
        if name == "dx":
            return self._imu_world
        if name == "dx":
            return self._timu
        if name == "dx":
            return self._force
        if name == "dx":
            return self._tforce
        if name == "dx":
            return self._ddx_u

    def set(self, name, data):
        """
        Set privat data
        Args:
            name: data name, str
            data: data, dict
        """
        if name == "x":
            self._x = data
        if name == "dx":
            self._dx = data
        if name == "dx":
            self._ddx = data
        if name == "dx":
            self._tx = data
        if name == "dx":
            self._u = data
        if name == "dx":
            self._tu = data
        if name == "dx":
            self._imu_body = data
        if name == "dx":
            self._imu_world = data
        if name == "dx":
            self._timu = data
        if name == "dx":
            self._force = data
        if name == "dx":
            self._tforce = data
        if name == "dx":
            self._ddx_u = data

    def _loadData(self, crop_data=False, crop_exp=False):
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
        nb_exp_read = 0

        # loop through all subfolders of the defined experiment serie
        for root, dirs, files in os.walk(os.path.join("experiment", self.series)):

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
                    skip_footer = nb_rows-self.keep_nb_rows

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
                elif f.find("imu") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self._timu[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[1], dtype=float, skip_footer=skip_footer)
                    self._imu_body[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                    usecols=[2,3,4], dtype=float, skip_footer=skip_footer)
                elif f.find("thrust") != -1:
                    #self.datas[f[:-4]] = pd.read_csv(os.path.join(root, f)).to_numpy()
                    self._tforce[root] = np.genfromtxt(  os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[1], dtype=float, skip_footer=skip_footer)
                    self._force[root] = np.genfromtxt(   os.path.join(root, f), delimiter=",", skip_header=1, 
                                                        usecols=[2,3,4], dtype=float, skip_footer=skip_footer)

            if crop_exp and nb_exp_read>=crop_exp:
                break
            nb_exp_read += 1
    
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
            try: timu_min = self._timu[exp][0]
            except: timu_min = np.Inf
            try: tf_min = self._tforce[exp][0]
            except: tf_min = np.Inf
            
            start_stamp = np.min([tx_min, tu_min, timu_min, tf_min])

            # convert tx, tu and tforce from stamp to seconds
            try: self._tx[exp] = (self._tx[exp]-start_stamp) / 1000000000
            except: pass
            try: self._tu[exp] = (self._tu[exp]-start_stamp) / 1000000000
            except: pass
            try: self._timu[exp] = (self._timu[exp]-start_stamp) / 1000000000
            except: pass
            try: self._tforce[exp] = (self._tforce[exp]-start_stamp) / 1000000000
            except: pass