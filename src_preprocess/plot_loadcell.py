import matplotlib.pyplot as plt
import numpy as np

class PlotLoadcell():
    def __init__(self, data) -> None:
        self.data = data
    
    def interpolateU(self, u_inter):
        """
        Plot interpolation of u
        Args:
            u_inter: interpolated u
        """
        u, tu, tf = self.data.get(names=["u", "tu", "tf"])

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=u[exp].shape[1], figsize =(8, 8))             
            for i, ax in enumerate(axs):
                ax.plot(tu[exp], u[exp][:,i], color="b", label="u")
                ax.plot(tf[exp], u_inter[exp][:,i], '--', color="r", label="u inter.")
                ax.legend()
                ax.set_title(f"Control input {i}")
            plt.show()

    def locSig(self, sgs, bgs):
        f, tf = self.data.get(names=["f", "tf"])

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=f[exp].shape[1], figsize =(8, 8))             
            for i, ax in enumerate(axs):
                y_min = np.min(f[exp][:,i])
                y_max = np.max(f[exp][:,i])

                ax.plot(tf[exp], f[exp][:,i], color="b", label=f"thrust({i})")
                for row in sgs:
                    for s in row:
                        ax.vlines(x=tf[exp][s["start"]], ymax=y_max, ymin=y_min, colors="r")
                        ax.vlines(x=tf[exp][s["stop"]], ymax=y_max, ymin=y_min, colors="r", linestyle="--")
                for row in bgs:
                    for b in row:
                        ax.vlines(x=tf[exp][b["start"]], ymax=y_max, ymin=y_min, colors="g")
                        ax.vlines(x=tf[exp][b["stop"]], ymax=y_max, ymin=y_min, colors="r", linestyle="--")
                ax.legend()
                ax.set_title(f"Signal detection {i}")
            plt.show()