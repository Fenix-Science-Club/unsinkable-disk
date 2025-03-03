import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors

tableau = list(mcolors.TABLEAU_COLORS)


parent_dir = Path(__file__).parent / "float_nofloat"

# import data
disk4_fl_path = parent_dir / "disk_4_float.csv"
disk4_fl = np.loadtxt(disk4_fl_path, delimiter=",", skiprows=1)

disk5_fl_path = parent_dir / "disk_5_float.csv"
disk5_fl = np.loadtxt(disk5_fl_path, delimiter=",", skiprows=1)

disk7_fl_path = parent_dir / "disk_7_float.csv"
disk7_fl = np.loadtxt(disk7_fl_path, delimiter=",", skiprows=1)

disk4_no_path = parent_dir / "disk_4_nofloat.csv"
disk4_no = np.loadtxt(disk4_no_path, delimiter=",", skiprows=1)
disk4_no = disk4_no[disk4_no[:, 2].argsort()]

disk5_no_path = parent_dir / "disk_5_nofloat.csv"
disk5_no = np.loadtxt(disk5_no_path, delimiter=",", skiprows=1)
disk5_no = disk5_no[disk5_no[:, 2].argsort()]

disk7_no_path = parent_dir / "disk_7_nofloat.csv"
disk7_no = np.loadtxt(disk7_no_path, delimiter=",", skiprows=1)
disk7_no = disk7_no[disk7_no[:, 2].argsort()]

sinking = [disk4_no[:-5], disk5_no[:-4], disk7_no[:-2]]
oscilating = [disk4_no[-5:], disk5_no[-4:], disk7_no[-2:]]
floating = [disk4_fl, disk5_fl, disk7_fl]


"""
Set of paramters for plots in form:
[radius in cm, mass of disk in g, uncertainty of mass in g, volume in ml, uncertainty of volume in ml]
"""

params = [
    [4, 11.7, 0.1, 2.77, 0.05],
    [5, 30.7, 0.1, 14.92, 0.05],
    [7, 50.9, 0.2, 12.16, 0.05],
]


"""
theoretical line is given by ploting the following expression for Q(v):

\\xi \\sqrt(\\rho^2 Q^3 g/4 \\pi \\nu) == g (m_{disk} - \\rho V)+\\rho Q U

for derivation and physical meaning of symbols please refere to the manuscript
https://arxiv.org/abs/2312.13099
"""


def Froude_fixed(Q, U, V, m_disk, R, sigma_Q, sigma_U):
    rho = 997
    g = 9.81
    m_eff = m_disk - rho * V
    a = np.sqrt(Q / (np.pi * U))
    froude = Q / (np.pi * a ** (5 / 2) * np.sqrt(g))
    dimensionless_V = m_eff / (np.pi * rho * a**3)
    return froude / np.sqrt(dimensionless_V), (1 / 2) * np.sqrt(
        (rho / (g * m_eff * Q * U)) * ((U * sigma_Q) ** 2 + (Q * sigma_U) ** 2)
    )


def Reynolds_fixed(Q, U, V, m_disk, R, sigma_Q, sigma_U):
    rho = 997
    nu = 10 ** (-6)
    m_eff = m_disk - rho * V
    a = np.sqrt(Q / (np.pi * U))
    dimensionless_V = m_eff / (np.pi * rho * a**3)
    reynolds = U * 2 * a / nu
    return np.sqrt(reynolds / dimensionless_V), np.sqrt(
        (rho / (2 * np.pi * m_eff * U**3 * nu))
        * ((2 * U * sigma_Q) ** 2 + (Q * sigma_U) ** 2)
    )


def relation(froude_omega):
    xi = 0.441
    return (np.sqrt(8) / xi) * (froude_omega + 1 / froude_omega)


def give_re_fr_sigma(triple_to_give):
    froude = [
        Froude_fixed(
            10 ** (-6) * floa[:, 2],
            floa[:, 0],
            pars[3] * 10 ** (-6),
            pars[1] * 10 ** (-3),
            pars[0] * 10 ** (-2),
            10 ** (-6) * floa[:, 3],
            floa[:, 1],
        )
        for floa, pars in zip(triple_to_give, params)
    ]
    froude_val = np.concatenate(tuple([value for value, uncer in froude]))
    froude_uncer = np.concatenate(tuple([uncer for value, uncer in froude]))
    reynolds = [
        Reynolds_fixed(
            10 ** (-6) * floa[:, 2],
            floa[:, 0],
            pars[3] * 10 ** (-6),
            pars[1] * 10 ** (-3),
            pars[0] * 10 ** (-2),
            10 ** (-6) * floa[:, 3],
            floa[:, 1],
        )
        for floa, pars in zip(triple_to_give, params)
    ]
    reynolds_val = np.concatenate(tuple([value for value, uncer in reynolds]))
    reynolds_uncer = np.concatenate(tuple([uncer for value, uncer in reynolds]))

    return froude_val, froude_uncer, reynolds_val, reynolds_uncer


fontsize = 11.5 * 1.5

plt.figure(figsize=(11, 5.5))
plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})

floating_data = give_re_fr_sigma(floating)
floats = plt.errorbar(
    floating_data[0],
    floating_data[2],
    xerr=floating_data[1],
    yerr=floating_data[3],
    fmt="o",
    mfc="w",
    c=tableau[1],
    label=r"Disk floats",
)


sinking_data = give_re_fr_sigma(sinking)
sinks = plt.errorbar(
    sinking_data[0],
    sinking_data[2],
    xerr=sinking_data[1],
    yerr=sinking_data[3],
    fmt="o",
    color=tableau[0],
    label=r"Disk sinks",
)


oscilating_data = give_re_fr_sigma(oscilating)
oscilates = plt.errorbar(
    oscilating_data[0],
    oscilating_data[2],
    xerr=oscilating_data[1],
    yerr=oscilating_data[3],
    fmt="s",
    ms=6,
    color=tableau[2],
    label=r"Disk sinks by oscilations",
)


predictions, = plt.plot(
    np.linspace(0, 2, 1000),
    relation(np.linspace(0, 2, 1000)),
    color="k",
    linestyle="--",
    linewidth=1.5,
    zorder=-1,
    label=r"$\frac{\sqrt{8}}{\xi}\left(\frac{\textrm{Fr}}{\omega} + \frac{\omega}{\textrm{Fr}}\right)$",
)


legend1_handles = [floats, sinks, oscilates]
legend2_handles = [predictions]

legend1 = plt.legend(
    fontsize=fontsize,
    frameon=False,
    loc=(0.6, 0.03),
    handles=legend1_handles,
)
legend2 = plt.legend(
    fontsize=fontsize,
    frameon=False,
    loc=(0.01, 0.8),
    handles=legend2_handles,
)

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)


plt.xlabel(r"$\textrm{Fr}/\omega$", fontsize=fontsize)
plt.ylabel(r"$\sqrt{\textrm{Re}}/\omega$", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.minorticks_on()
plt.tick_params(which="both", top=True, right=True)

plt.xlim(0.3, 2)
plt.ylim(7, 29)

plt.savefig(
    "graphs/froude_reynolds.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.savefig(
    "graphs/froude_reynolds.png",
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.savefig(
    "graphs/froude_reynolds.eps",
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
