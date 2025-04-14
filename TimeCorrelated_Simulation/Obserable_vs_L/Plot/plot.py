import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


colors = plt.cm.viridis(np.linspace(0, 1, 6))
marks = ['o','v','s','D']
plt.figure(figsize=(10, 6))


L = range(9)[1:]

title = './Data/SNI_L='
SNI = []
SNIstd = []
for i in range(len(L)):
    with open(title+str(i+1),'rb')as f:
        a= pickle.load(f)
    SNI.append(np.real(a[0]))
    SNIstd.append(np.real(a[1]))


title0 = './Data/ErrorFree'
with open(title0,'rb')as f1:
    b = pickle.load(f1)
plt.plot(L,b,color='black',linestyle='-',label = "Error-Free")


title = './Data/No_mitigate'
with open(title,'rb')as f:
    raw= pickle.load(f)
raw_std = [np.sqrt((1-raw[i])**2/25600000) for i in range(len(raw))]
plt.scatter(L,raw,color=colors[0],label = r'Raw',marker=marks[0],zorder=100)
plt.errorbar(L, raw, yerr=raw_std, fmt=marks[0],alpha=0.5, markerfacecolor='none',markeredgecolor=colors[0],ecolor=colors[0],capsize=3, capthick=1, zorder=100)

plt.scatter(L,SNI,color=colors[1],label = r'SNI',marker=marks[1],zorder=100)
plt.errorbar(L, SNI, yerr=SNIstd, fmt=marks[1],alpha=0.5, markerfacecolor='none',markeredgecolor=colors[1],ecolor=colors[1],capsize=3, capthick=1, zorder=100)

       
plt.xlabel('L')
plt.ylabel(r'$\langle O \rangle$')
plt.legend()
title_save = 'Obserable_vs_L'
plt.savefig(
        './' +(title_save + ".pdf"),
        dpi=3000,
        bbox_inches='tight',
    )