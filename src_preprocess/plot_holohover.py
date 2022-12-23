import matplotlib.pyplot as plt
import numpy as np


class PlotHolohover():
    def __init__(self) -> None:
        pass

    def diffPosition(self):

        tx, x, dx, ddx = self.data.get(names=["tx", "x", "dx", "ddx"])

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=self.x[exp].shape[1], ncols=2, figsize =(8, 8))             
            for i, ax in enumerate(axs[:,0]):
                ax.plot(tx[exp], x[exp][:,i], color="b", label="pos")
                ax.plot(tx[exp], dx[exp][:,i], color="g", label="vel")
                ax.plot(tx[exp], ddx[exp][:,i], color="r", label="acc")
                #ax.plot(self.tx[exp], self.imu_world[exp], color="c", label="imu")
                # ax.set_ylim([np.min(self.ddx[exp][:,i]), np.max(self.ddx[exp][:,i])])
                ax.legend()
                ax.set_title(f"Dimension {i}")

            # integrate dx and calc. error
            d_error = self.integrate(dx[exp], tx[exp], x[exp][0,:])
            d_error = d_error - x[exp]

            # double integrate imu and calc. error
            dd_error = self.integrate(ddx[exp], tx[exp], dx[exp][0,:])
            dd_error = self.integrate(dd_error, tx[exp], x[exp][0,:])
            dd_error = dd_error - x[exp]

            # double integrate ddx and calc. error
            # imu_error = self.integrate(self.imu[exp], self.tx[exp], self.dx[exp][0,:])
            # imu_error = self.integrate(imu_error, self.tx[exp], self.x[exp][0,:])
            # imu_error = imu_error - self.x[exp]

            for i, ax in enumerate(axs[:,1]):
                ax.plot(tx[exp], d_error[:,i], color="g", label="dx error")
                ax.plot(tx[exp], dd_error[:,i], color="r", label="ddx error")
                # if i < 2:
                #     ax.plot(self.tx[exp], imu_error[:,i], color="c", label="imu error")
                ax.legend()
                ax.set_title(f"Dimension {i}")                    

            plt.show()
