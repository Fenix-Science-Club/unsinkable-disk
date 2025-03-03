import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# from scipy.optimize import minimize

parent_dir = Path(__file__).parent

# import data
disk4_path = parent_dir / "transition/disk_4.csv"
disk4 = np.loadtxt(disk4_path, delimiter=",", skiprows=1)

disk6_path = parent_dir / "transition/disk_6.csv"
disk6 = np.loadtxt(disk6_path, delimiter=",", skiprows=1)

disk8_path = parent_dir / "transition/disk_8.csv"
disk8 = np.loadtxt(disk8_path, delimiter=",", skiprows=1)

# fitted value of xi = 0.441, scaling is xi*H^{*}
xi = 0.441


def scaling(H):
    return xi * H


xargs = np.linspace(0, 15, 100)
yargs = scaling(xargs)


# add horizontal line for capillary length
def capillary(H):
    return 2.7 + H * 0


capargs = capillary(xargs)

fontsize = 23

plt.figure(figsize=(10, 5.5))
plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})

plt.errorbar(
    disk4[:, 0],
    disk4[:, 2],
    xerr=disk4[:, 1],
    yerr=disk4[:, 3],
    fmt="o",
    label=r"Disk radius $R = 4$ cm",
)
plt.errorbar(
    disk6[:, 0],
    disk6[:, 2],
    xerr=disk6[:, 1],
    yerr=disk6[:, 3],
    fmt="s",
    ms=7,
    color="#d39400",
    label=r"Disk radius $R = 6$ cm",
)
plt.errorbar(
    disk8[:, 0],
    disk8[:, 2],
    xerr=disk8[:, 1],
    yerr=disk8[:, 3],
    fmt="o",
    mfc="w",
    color="g",
    label=r"Disk radius $R = 8$ cm",
)

plt.plot(xargs, yargs, label=r"Linear fit $\xi H^*$ with $\xi=0.441$", color="black")
plt.plot(xargs, capargs, color="gray", linestyle="--")


plt.xlabel(r"Characteristic height $\left(H^*\right)$ [mm]", fontsize=fontsize)
plt.ylabel(r"Critical height $\left(H_{\textrm{crit}}\right)$ [mm]", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize, frameon=False)

plt.minorticks_on()
plt.tick_params(which="both", top=True, right=True)

plt.xlim(0, 15)
plt.ylim(0, 7.99)

plt.savefig("graphs/plot_scaling.eps", format="eps", bbox_inches="tight")
plt.show()
