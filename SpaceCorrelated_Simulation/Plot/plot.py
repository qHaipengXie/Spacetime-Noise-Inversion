import numpy as np
import pickle
import matplotlib.pyplot as plt


colors = plt.cm.viridis(np.linspace(0, 1, 6))
marks = ['o','v','s','D']
plt.figure(figsize=(10, 6))


L = range(21)[1:]


title = './SNI_x_n'
SNI = []
SNIstd = []
with open(title,'rb')as f:
    a= pickle.load(f)
for i in range(len(a)):
    SNI.append(np.real(a[i][0]))
    SNIstd.append(np.real(a[i][1]))


title = './cPEC_x_n'
cPEC = []
cPECstd = []
with open(title,'rb')as f:
    a= pickle.load(f)
for i in range(len(a)):
    cPEC.append(np.real(a[i][0]))
    cPECstd.append(np.real(a[i][1]))

title = './Raw'
with open(title,'rb')as f:
    raw= pickle.load(f)
raw_std = [np.sqrt((1-raw[i])**2/25600000) for i in range(len(raw))]
plt.scatter(L,raw,color=colors[0],label = r'Raw',marker=marks[0],zorder=100)
# plt.errorbar(L, raw, yerr=raw_std, fmt=marks[0],alpha=0.5, markerfacecolor='none',markeredgecolor=colors[0],ecolor=colors[0],capsize=3, capthick=1, zorder=100)

plt.scatter(L,SNI,color=colors[1],label = r'SNI',marker=marks[1],zorder=100)
# plt.errorbar(L, SNI, yerr=SNIstd, fmt=marks[1],alpha=0.5, markerfacecolor='none',markeredgecolor=colors[1],ecolor=colors[1],capsize=3, capthick=1, zorder=100)

plt.scatter(L,cPEC,color=colors[2],label = r'cPEC',marker=marks[2],zorder=100)
# plt.errorbar(L, cPEC, yerr=cPECstd, fmt=marks[2],alpha=0.5, markerfacecolor='none',markeredgecolor=colors[2],ecolor=colors[2],capsize=3, capthick=1, zorder=100)

plt.xlabel('n')

plt.ylabel(r'$\langle O \rangle$')
plt.legend()
title_save = 'space'
plt.savefig(
        './' +(title_save + ".pdf"),
        dpi=3000,
        bbox_inches='tight',
    )