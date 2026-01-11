import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib import pyplot as plt

# >>>>>>>> matplotlib settings >>>>>>>>
# https://www.adobe.com/uk/creativecloud/design/discover/a4-format.html
A4_PAPER_SIZE_IN_INCHES = (8.27, 11.67)

# https://www.nature.com/documents/Final_guide_to_authors.pdf
# https://jonathansoma.com/lede/data-studio/matplotlib/exporting-from-matplotlib-to-open-in-adobe-illustrator/

SANS_SERIF_FONT = "Arial"
AXES_LABEL_FONT_SIZE = 7


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Use LaTeX for math formatting
        "text.usetex": False,
        "text.latex.preamble": (
            r"\usepackage{amsmath} "
            r"\usepackage{amssymb} "
            r"\usepackage{sfmath}"
        ),
        "font.family": "sans-serif",
        # > font ["Helvetica", "Arial"]
        "font.sans-serif": [SANS_SERIF_FONT],
        # > other text size: 5-7 pt
        "font.size": 7,
        # > xticks
        "xtick.labelsize": 12,
        "xtick.direction": "in",
        "xtick.major.size": 3,
        "xtick.major.width": 0.5,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.5,
        "xtick.minor.visible": False,
        "xtick.top": False,
        # > yticks
        "ytick.labelsize": 12,
        "ytick.direction": "in",
        "ytick.major.size": 3,
        "ytick.major.width": 0.5,
        "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.5,
        "ytick.minor.visible": False,
        "ytick.right": False,
        # > legend
        "legend.fontsize": 12,
        "legend.frameon": False,  # Remove legend frame
        "legend.columnspacing": 0.3,
        "legend.labelspacing": 0.3,
        "legend.markerscale": 0.8,
        # > savefig
        # "savefig.bbox": "tight",
        # "savefig.pad_inches": 0.0,
        # > lines
        "lines.linewidth": 1,
        "lines.markersize": 3,
        # > axes and grid
        "axes.titlesize": "medium",
        "axes.linewidth": 0.5,
        "axes.labelpad": 0.0,
        "axes.prop_cycle": cycler(
            "color",
            [
                "0C5DA5",
                "00B945",
                "FF9500",
                "FF2C00",
                "845B97",
                "474747",
                "9e9e9e",
            ],
        ),
        "grid.linewidth": 0.5,
        "axes.formatter.use_mathtext": True,
        # > mathtext
        "mathtext.fontset": "custom",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "cmsy10",  # default: cursive
        "mathtext.it": "sans:italic",
        "mathtext.rm": "sans",
        "mathtext.sf": "sans",
        "mathtext.tt": "monospace",
        # > figure
        # "figure.labelsize": "medium",
        "figure.titlesize": "medium",
    }
)
plt.ion()
with open("./Result/D3_v3_d=1",'rb')as f:
    y0 = pickle.load(f)
with open("./Result/D3_v3_d=3",'rb')as f:
    y1 = pickle.load(f)
with open("./Result/D3_v3_d=5",'rb')as f:
    y2 = pickle.load(f)
with open("./Result/D3_v3_d=7",'rb')as f:
    y3 = pickle.load(f)
# 数据
x = np.logspace(np.log10(0.01), np.log10(0.05), num=10)
xlist = [x,x,x[1:],x[2:]]
# 整理成列表
y_lists = [y0,y1, y2[1:],y3[2:]]
colors = ['black','blue', 'green', 'red']

stdlist1 = []
# 绘图
plt.figure(figsize=(7, 3.5))

for i, y in enumerate(y_lists):
    y_solid = [pair[0] for pair in y]
    std1 = [np.sqrt((1-x)*x/(1.28*10**7)) for x in y_solid]
    y_dashed = [pair[1] for pair in y]
    std2 = [np.sqrt((1-x)*x/(1.28*10**7)) for x in y_dashed]
    if i ==0:
        plt.scatter(xlist[i],y_solid,marker='s', color=colors[i],s=20,zorder=100, label=f'Physical qubits, '+r'$P_{1,2}$')
        plt.scatter(xlist[i], y_dashed,marker='s',facecolor='none',  color=colors[i],s=20,zorder=100, label=f'Physical qubits, $P_1P_2$')
    else:
        plt.scatter(xlist[i],y_solid,marker='s', color=colors[i],s=20,zorder=100, label=f'd = {2*i+1}, '+r'$P_{1,2}$')
        plt.scatter(xlist[i], y_dashed,marker='s',facecolor='none',  color=colors[i],s=20,zorder=100, label=f'd = {2*i+1}, $P_1P_2$')    
    plt.plot(xlist[i], y_solid, linestyle='-', color=colors[i])
    plt.plot(xlist[i], y_dashed, linestyle='--', color=colors[i])
    # plt.errorbar(xlist[i],y_solid,yerr=std1,marker='s',color=colors[i], label=f'd = {2*i+3}, '+r'$P_{1,2}$')
    # plt.errorbar(xlist[i],y_dashed,yerr=std2,marker='s',color=colors[i], label=f'd = {2*i+3}, $P_1P_2$')
# std = np.sqrt((1-y_solid[0])*y_solid[0]/(1.28*10**7))
# plt.errorbar(xlist[2][0],y_solid[0],yerr=std,color=colors[i])
plt.xlabel(r"$p$")
plt.ylabel(r"$P_L$")
plt.xscale('log')
plt.yscale('log')
# plt.title("")
plt.legend()
# plt.legend().get_frame().set_facecolor('none')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
plt.savefig('./Logical_Error_with_correlation_Version_3D_v2.pdf')