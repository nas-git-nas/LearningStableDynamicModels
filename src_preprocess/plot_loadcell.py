import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy import stats

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

    def locSig(self, sgs, bgs, ids):
        f, tf = self.data.get(names=["f", "tf"])

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=f[exp].shape[1], figsize =(8, 8))             
            for i, ax in enumerate(axs):
                y_min = np.min(f[exp][:,i])
                y_max = np.max(f[exp][:,i])

                ax.plot(tf[exp], f[exp][:,i], color="b", label=f"thrust({i})")
                for i in range(sgs.shape[0]):
                    for j in range(sgs.shape[1]):
                        ax.vlines(x=tf[exp][sgs[i,j,0]], ymax=y_max, ymin=y_min, colors="r")
                        ax.vlines(x=tf[exp][sgs[i,j,1]], ymax=y_max, ymin=y_min, colors="r", linestyle="--")
                for i in range(bgs.shape[0]):
                    for j in range(bgs.shape[1]):
                        ax.vlines(x=tf[exp][bgs[i,j,0]], ymax=y_max, ymin=y_min, colors="g")
                        ax.vlines(x=tf[exp][bgs[i,j,1]], ymax=y_max, ymin=y_min, colors="g", linestyle="--")
                ax.legend()
                ax.set_title(f"Signal detection {i}")
            plt.show()

    def calcNorm(self):
        f, tf = self.data.get(names=["f", "tf"])

        for exp in self.data.exps:
            fig, axs = plt.subplots(nrows=f[exp].shape[1], figsize =(8, 8))
            axs[0].plot(tf[exp], f[exp][:,0], color="b", label=f"force x") 
            axs[0].legend()
            axs[1].plot(tf[exp], f[exp][:,1], color="b", label=f"force y") 
            axs[1].legend()
            axs[2].plot(tf[exp], f[exp][:,2], color="b", label=f"force norm")     
            axs[2].legend() 
            plt.show()

    def calcMeanNorm(self, means, stds, ids):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(12, 8))             
        for i in range(means.shape[0]):
            k = int(i/len(axs[0]))
            l = i % len(axs[0])

            axs[k,l].set_title(f"Motor {int(ids[i,0,0])}")
            for j in range(means.shape[1]):
                axs[k,l].scatter(ids[i,j,1], means[i,j], color="b")
                axs[k,l].errorbar(ids[i,j,1], means[i,j], stds[i,j], color="r", fmt='.k')
            axs[k,l].scatter([], [], color="b", label="Mean")
            axs[k,l].errorbar([], [], [], color="r", fmt='.k', label="Std.")
            axs[k,l].set_xlabel("signal")
            axs[k,l].set_ylabel("thrust [N]")
            axs[k,l].legend()
        plt.show()

    def signal2thrust(self, means, stds, ids, coeffs):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(12, 8))
        color_map = cm.get_cmap('cool', 12)

        for i in range(means.shape[0]):
            k = int(i/len(axs[0]))
            l = i % len(axs[0])

            axs[k,l].set_title(f"Motor {int(ids[i,0,0])}")
            colors = np.linspace(0, 1, coeffs.shape[1])
            for j in range(coeffs.shape[1]):
                # calc. polynomial approximation of thrust
                x = np.linspace(-0.2, 1.2, 100)
                thrust_approx = self._polyCurve(x=x, coeff=coeffs[i,j,:])

                # calc. R value
                stat_approx = self._polyCurve(x=ids[i,:,1], coeff=coeffs[i,j,:])
                _, _, rvalue, _, _ = stats.linregress(stat_approx, means[i,:])

                axs[k,l].plot(x, thrust_approx, color=color_map(colors[j]), label=f"Deg={j+2} (R^2={np.round(rvalue**2, 4)})")

            for j in range(means.shape[1]):
                axs[k,l].scatter(ids[i,j,1], means[i,j], color="b")
                axs[k,l].errorbar(ids[i,j,1], means[i,j], stds[i,j], color="r", fmt='.k')
            axs[k,l].scatter([], [], color="b", label="Mean")
            axs[k,l].errorbar([], [], [], color="r", fmt='.k', label="Std.")
            axs[k,l].set_xlabel("signal")
            axs[k,l].set_ylabel("thrust [N]")
            axs[k,l].legend()
        plt.show()

    def _polyCurve(self, x, coeff):

        Xpoly = np.zeros((len(x),len(coeff)))
        for i in range(len(coeff)):
            Xpoly[:,i] = np.power(x, i+1)
        
        return Xpoly @ coeff