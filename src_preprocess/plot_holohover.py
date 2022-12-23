import matplotlib.pyplot as plt
import numpy as np


class PlotHolohover():
    def __init__(self, data) -> None:
        self.data = data

    def cropData(self, plot_lw_x_idx, plot_lw_u_idx, plot_up_x_idx, plot_up_u_idx):
        """
        Plot removed data in the beginning and the end
        """
        x, tx, u, tu = self.data.get(names=["x", "tx", "u", "tu"])

        for exp in self.data.exps:
            plt.plot(tx[exp], x[exp][:,2], )
            plt.vlines(tx[exp][plot_lw_x_idx[exp]], ymin=-3.3, ymax=3.3, colors="r")
            plt.vlines(tx[exp][plot_up_x_idx[exp]], ymin=-3.3, ymax=3.3, colors="r")
            plt.title(f"Theta of exp. {exp}")
            plt.show()

    def intermolateU(self, u_inter):
        """
        Plot plynomial interpolation of control input to match with x
        Args:
            u_inter: interpolation of u, dict
        """
        tx, u, tu = self.data.get(names=["tx", "u", "tu"])
        u_inter = u_inter.copy()

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=u[exp].shape[1], figsize =(8, 8))             
            for i, ax in enumerate(axs):
                ax.plot(tu[exp], u[exp][:,i], color="b", label="u")
                ax.plot(tx[exp], u_inter[exp][:,i], '--', color="r", label="u inter.")
                ax.legend()
                ax.set_title(f"Control input {i}")
            plt.show()

    def firstOrderU(self, u_approx):
        """
        Plot first order model approximation of U
        Args:
            u_approx: first order model approximation of u, dict
        """
        tx, u = self.data.get(names=["tx", "u"])
        u_approx = u_approx.copy()

        for exp in self.data.exps:    
            fig, axs = plt.subplots(nrows=len(self.data.exps), figsize =(8, 8))             
            for i, ax in enumerate(axs):
                ax.plot(tx[exp], u[exp][:,i], color="b", label="u")
                ax.plot(tx[exp], u_approx[exp][:,i], '--', color="r", label="u approx.")
                ax.legend()
                ax.set_title(f"Control input {i}")
            plt.show()

    def diffPosition(self):

        tx, x, dx, ddx = self.data.get(names=["tx", "x", "dx", "ddx"])

        for exp in self.exps:
            fig, axs = plt.subplots(nrows=self.x[exp].shape[1], ncols=2, figsize =(8, 8))             
            for i, ax in enumerate(axs[:,0]):
                ax.plot(tx[exp], x[exp][:,i], color="b", label="pos")
                ax.plot(tx[exp], dx[exp][:,i], color="g", label="vel")
                ax.plot(tx[exp], ddx[exp][:,i], color="r", label="acc")
                ax.legend()
                ax.set_title(f"Dimension {i}")

            # integrate dx and calc. error
            d_error = self._integrate(dx[exp], tx[exp], x[exp][0,:])
            d_error = d_error - x[exp]

            # double integrate imu and calc. error
            dd_error = self._integrate(ddx[exp], tx[exp], dx[exp][0,:])
            dd_error = self._integrate(dd_error, tx[exp], x[exp][0,:])
            dd_error = dd_error - x[exp]

            for i, ax in enumerate(axs[:,1]):
                ax.plot(tx[exp], d_error[:,i], color="g", label="dx error")
                ax.plot(tx[exp], dd_error[:,i], color="r", label="ddx error")
                ax.legend()
                ax.set_title(f"Dimension {i}")                    

            plt.show()

    def uShift(self, total_shift, ddx_u, ddx_u_unshifted, ddx_unshifted):
        """
        Args:
            total_shift: nb. data shifts
            ddx_u: white box estimation of ddx using u, dict
            ddx_u_unshifted: white box estimation of ddx using u before shifting data, dict
            ddx_unshifted: differentiated opti-track ddx before shifting, dict
        """
        tx, x, ddx = self.data.get(names=["tx", "x", "ddx"])

        for exp in self.x:
            fig, axs = plt.subplots(nrows=x[exp].shape[1], ncols=1, figsize =(8, 8))   

            delay_s = total_shift * (tx[exp][-1] - tx[exp][0]) / tx[exp].shape[0]
            
            fig.suptitle(f'Avg. delay of ddx wrt. control: {np.round(delay_s*1000)}ms ({total_shift} shifts)')
            for i, ax in enumerate(axs):               
                ax.plot(tx[exp], ddx_u[exp][:,i], color="c", label="ddx_u shifted")
                ax.plot(tx[exp], ddx_u_unshifted[exp][:ddx_u[exp].shape[0],i], '--', color="b", label="ddx_u unshifted")
                ax.plot(tx[exp], ddx[exp][:,i],  color="r", label="ddx")
                ax.plot(tx[exp], ddx_unshifted[exp][:ddx_u[exp].shape[0],i], '--', color="g", label="ddx unshifted")
                ax.legend()
                error = (1/ddx_u[exp].shape[0]) * np.sum(np.abs(ddx_u[exp][:,i]-ddx[exp][:,i]))
                unshifted_error = (1/ddx_u[exp].shape[0]) * np.sum(np.abs(ddx_u_unshifted[exp][:ddx_u[exp].shape[0],i]-ddx[exp][:,i]))
                ax.set_title(f"Dimension {i}, mean error: {np.round(error,3)}, unshifted error: {np.round(unshifted_error,3)}")
            plt.show()

    def _integrate(self, dX, tX, X0):
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