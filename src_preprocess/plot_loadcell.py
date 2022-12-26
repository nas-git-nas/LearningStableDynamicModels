import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from brokenaxes import brokenaxes
import numpy as np
from scipy import stats

from src_preprocess.functions import polyFct2, polyFct3, polyFct4, polyFct5, expRise, expFall

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

                axs[k,l].plot(x, thrust_approx, color=color_map(colors[j]), label=f"Deg={j+2} (R^2={np.round(rvalue**2, 3)})")

            for j in range(means.shape[1]):
                axs[k,l].scatter(ids[i,j,1], means[i,j], color="b")
                axs[k,l].errorbar(ids[i,j,1], means[i,j], stds[i,j], color="r", fmt='.k')
            axs[k,l].scatter([], [], color="b", label="Mean")
            axs[k,l].errorbar([], [], [], color="r", fmt='.k', label="Std.")
            axs[k,l].set_xlabel("signal")
            axs[k,l].set_ylabel("thrust [N]")
            axs[k,l].legend()
        plt.show()

    def thrust2signal(self, means, stds, ids, coeffs):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize =(12, 8))
        color_map = cm.get_cmap('cool', 12)

        for i in range(means.shape[0]):
            k = int(i/len(axs[0]))
            l = i % len(axs[0])

            axs[k,l].set_title(f"Motor {int(ids[i,0,0])}")
            colors = np.linspace(0, 1, coeffs.shape[1])
            for j in range(coeffs.shape[1]):
                # calc. polynomial approximation of thrust
                x = np.linspace(-0.1, 0.7, 100)
                thrust_approx = self._polyCurve(x=x, coeff=coeffs[i,j,:])

                # calc. R value
                stat_approx = self._polyCurve(x=ids[i,:,1], coeff=coeffs[i,j,:])
                _, _, rvalue, _, _ = stats.linregress(stat_approx, means[i,:])

                axs[k,l].plot(x, thrust_approx, color=color_map(colors[j]), label=f"Deg={j+2} (R^2={np.round(rvalue**2, 3)})")

            for j in range(means.shape[1]):
                axs[k,l].scatter(means[i,j], ids[i,j,1], color="b")
                axs[k,l].errorbar(means[i,j], ids[i,j,1], xerr=stds[i,j], color="r", fmt='.k')
            axs[k,l].scatter([], [], color="b", label="Mean")
            axs[k,l].errorbar([], [], [], color="r", fmt='.k', label="Std.")
            axs[k,l].set_xlabel("thrust [N]")
            axs[k,l].set_ylabel("signal")
            axs[k,l].legend()
        plt.show()

    def _polyCurve(self, x, coeff):
        Xpoly = np.zeros((len(x),len(coeff)))
        for i in range(len(coeff)):
            Xpoly[:,i] = np.power(x, i+1)
        
        return Xpoly @ coeff

    def motorTransition(self, sgs, bgs_trans, ids, means, tau, delay, trans, signal_space=False):
        f, tf = self.data.get(names=["f", "tf"])
        f, tf = list(f.values())[0], list(tf.values())[0]
        fn = f[:,2]

        for i in range(sgs.shape[0]):
            # fig, axs = plt.subplots(nrows=4, ncols=4, figsize =(12, 8))
            # fig.suptitle(f"Motor: {i + 1}")
            fig = plt.figure(figsize =(12, 8))
            fig.suptitle(f"Motor: {i + 1}")
            sps = GridSpec(nrows=4, ncols=4, hspace=0.5)

            for j in range(sgs.shape[1]):
                k = int(j/4)
                l = j % 4

                # ax = fig.add_subplot(sps[k, l])
                bax = brokenaxes(xlims=((tf[sgs[i,j,0]], tf[sgs[i,j,1]]), (tf[bgs_trans[i,j,0]]-0.2, tf[bgs_trans[i,j,1]])), 
                                    subplot_spec=sps[k,l], d=0.007, wspace=0.4)
                bax.plot(tf[sgs[i,j,0]:sgs[i,j,2]], fn[sgs[i,j,0]:sgs[i,j,2]], color="m")                    
                bax.plot(tf[bgs_trans[i,j,0]:bgs_trans[i,j,2]], fn[bgs_trans[i,j,0]:bgs_trans[i,j,2]], color="b")

                bax.set_title(f"Signal: {ids[i,j,1]}")
                bax.axs[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                locs = bax.axs[0].get_xticks()
                bax.axs[0].set_xticks(np.array([locs[2]]))
                bax.axs[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                locs = bax.axs[1].get_xticks()
                bax.axs[1].set_xticks(np.array([locs[1],locs[-1]]))

                # for ax in bax.axs:
                #     ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #     locs = ax.get_xticks()
                #     # print(locs)
                #     ax.set_xticks(np.array([locs[1],locs[-1]]))
                #     # labels = ax.get_xlabels()
                    
                #     locs = ax.get_xticks()
                    # print(locs)
                    # print(labels)
                # locs, labels = plt.xticks()
                # print(locs)
                # print(labels)
                if signal_space:
                    if l == 0: bax.set_ylabel("signal")
                else:
                    if l == 0: bax.set_ylabel("thrust (N)")
                if k == 3: bax.set_xlabel("seconds (s)")

                # skip signal if noise level is too high
                noise_thr = 2*(np.max(fn[bgs_trans[i,j,1]:bgs_trans[i,j,2]]) - np.min(fn[bgs_trans[i,j,1]:bgs_trans[i,j,2]]))
                if noise_thr > means[i,j]:
                    continue

                if signal_space:
                    steady_state = ids[i,j,1]
                else:
                    steady_state = means[i,j]
                
                bax.axvspan(xmin=tf[sgs[i,j,0]], xmax=tf[sgs[i,j,0]]+delay[i,j,0], color="tab:gray", alpha=0.25)
                bax.axvspan(xmin=tf[sgs[i,j,0]]+delay[i,j,0], xmax=tf[sgs[i,j,0]]+delay[i,j,0]+trans[i,j,0], 
                                    color="tab:brown", alpha=0.5)
                bax.axvspan(xmin=tf[bgs_trans[i,j,0]], xmax=tf[bgs_trans[i,j,0]]+delay[i,j,1], color="tab:gray", alpha=0.25)
                bax.axvspan(xmin=tf[bgs_trans[i,j,0]]+delay[i,j,1], xmax=tf[bgs_trans[i,j,0]]+delay[i,j,1]+trans[i,j,1], 
                                    color="tab:brown", alpha=0.5)

                # if not signal_space:
                fit_up_X = (tf[sgs[i,j,0]:sgs[i,j,1]] - tf[sgs[i,j,0]], np.ones(tf[sgs[i,j,0]:sgs[i,j,1]].shape) * steady_state)
                bax.plot(tf[sgs[i,j,0]:sgs[i,j,1]], expRise(X=fit_up_X, tau=tau[i,j,0], delay=delay[i,j,0]), color="g")
                fit_dw_X = (tf[bgs_trans[i,j,0]:bgs_trans[i,j,1]] - tf[bgs_trans[i,j,0]], 
                            np.ones(tf[bgs_trans[i,j,0]:bgs_trans[i,j,1]].shape) * steady_state)
                bax.plot(tf[bgs_trans[i,j,0]:bgs_trans[i,j,1]], expFall(X=fit_dw_X, tau=tau[i,j,1], delay=delay[i,j,1]), color="g")
                
                
            plt.show()