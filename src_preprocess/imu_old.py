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
        for i, tx in enumerate(tx[exp]):
            if tx>lower_time and lower_x_idx==0:
                lower_x_idx = i
            if tx > upper_time:
                upper_x_idx = i - 1
                break
        
        lower_u_idx = 0
        upper_u_idx = u[exp].shape[0] - 1
        for i, tu in enumerate(tu[exp]):
            if tu>lower_time and lower_u_idx==0:
                lower_u_idx = i
            if tu > upper_time:
                upper_u_idx = i - 1
                break

        # lower_imu_idx = 0
        # upper_imu_idx = self.imu_body[exp].shape[0] - 1
        # for i, timu in enumerate(self.timu[exp]):
        #     if timu>lower_time and lower_imu_idx==0:
        #         lower_imu_idx = i
        #     if timu > upper_time:
        #         upper_imu_idx = i - 1
        #         break

        # ensure that min(tu)<min(tx) and max(tx)<max(tu) s.t. range(tu) is larger than range(tx)
        # this is necessary because u is intermolated to match x
        while tu[exp][lower_u_idx] > tx[exp][lower_x_idx]:
            lower_x_idx += 1
        while tu[exp][upper_u_idx] < tx[exp][upper_x_idx]:
            upper_x_idx -= 1

        # ensure that min(timu)<min(tx) and max(tx)<max(timu) s.t. range(timu) is larger than range(tx)
        # this is necessary because u is intermolated to match x
        # while self.timu[exp][lower_imu_idx] > self.tx[exp][lower_x_idx]:
        #     lower_x_idx += 1
        # while self.timu[exp][upper_imu_idx] < self.tx[exp][upper_x_idx]:
        #     upper_x_idx -= 1

        x[exp] = x[exp][lower_x_idx:upper_x_idx+1,:]
        tx[exp] = tx[exp][lower_x_idx:upper_x_idx+1]
        u[exp] = u[exp][lower_u_idx:upper_u_idx+1,:]
        tu[exp] = tu[exp][lower_u_idx:upper_u_idx+1]

        plot_lw_x_idx[exp] = lower_x_idx
        plot_lw_u_idx[exp] = lower_u_idx
        plot_up_x_idx[exp] = upper_x_idx
        plot_up_u_idx[exp] = upper_u_idx      

    if plot:
        self.plot.cropData(plot_lw_x_idx=plot_lw_x_idx, plot_lw_u_idx=plot_lw_u_idx, 
                            plot_up_x_idx=plot_up_x_idx, plot_up_u_idx=plot_up_u_idx)
                        
    self.data.set(names=["x", "tx", "u", "tu"], datas=[x, tx, u, tu])

        # self.imu_body[exp] = self.imu_body[exp][lower_imu_idx:upper_imu_idx+1,:]
        # self.timu[exp] = self.timu[exp][lower_imu_idx:upper_imu_idx+1]

def intermolateIMU(self, plot=False):
    """
    Polynomial interpolation of imu data to match with x
    Args:
        plot: if True plot results
    """
    for exp in self.data.exps:
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

def imuRemoveOffset(self, plot=False):
    """
    Calc. offset of imu data and remove it
    Args:
        plot: if True plot results
    """
    imu_body = self.imu_body.copy()

    imu_offset_free = {}
    for exp in self.data.exps:

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
    for exp in data.exps:

        # convert body to world frame
        rotation_b2w = self.rotMatrix(x[exp][:,2]) # (N,S,S)
        imu_world[exp] = np.einsum('ns,nps->np', imu_body[exp], rotation_b2w) # (N, S)

    return imu_world 

def imuShift(self, plot=False):
    """
    Args:
        plot: if True plot results
    """
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
