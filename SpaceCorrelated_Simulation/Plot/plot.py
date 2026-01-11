import numpy as np
import pickle
import matplotlib.pyplot as plt



colors = plt.cm.viridis(np.linspace(0, 1, 6))
marks = ['o','v','s','D']
plt.figure(figsize=(7, 3.5))



title = './../Result/SNI_vs_Mp_n=4'
SNI = []
SNIstd = []
with open(title,'rb')as f:
    a= pickle.load(f)
for i in range(len(a)):
    SNI.append(np.real(a[i][0]))
    SNIstd.append(np.real(a[i][1]))

# title = './cPEC'
# cPEC = []
# cPECstd = []
# with open(title,'rb')as f:
#     a= pickle.load(f)
# for i in range(len(a)):
#     cPEC.append(np.real(a[i][0]))
#     cPECstd.append(np.real(a[i][1]))
title0 = './../Result/Free_n=4'
with open(title0,'rb')as f1:
    b = pickle.load(f1)
title = './../Result/No mitigation_n=4'
with open(title,'rb')as f:
    raw= pickle.load(f)

title = './../Result/cPEC_n=4'
with open(title,'rb')as f:
    cPEC_theo_bias= pickle.load(f)

x = [256*2**(i+1) for i in range(20)]
plt.plot(x,[np.abs(raw-b) for i in range(len(x))],linestyle='--',label = 'No mitigation',color= 'red')
plt.plot(x,[cPEC_theo_bias-b for i in range(len(x))],label = r'cPEC theoretical bias',color= 'black')


# x = [256*2**(i+1) for i in range(15)]
# plt.scatter(x,cPEC,color=colors[2],label = r'cPEC',marker=marks[2],zorder=100)
# plt.errorbar(x, cPEC, yerr=cPECstd, fmt=marks[2],alpha=0.5, markerfacecolor='none',markeredgecolor=colors[2],ecolor=colors[2],capsize=3, capthick=1, zorder=100)

x = [256*2**(i+1) for i in range(20)]
plt.scatter(x,SNI,color=colors[3],label = r'SNI',marker=marks[1],zorder=100)
plt.errorbar(x, SNI, yerr=SNIstd, fmt=marks[1],alpha=1, markerfacecolor='none',markeredgecolor=colors[3],ecolor=colors[3],capsize=6, capthick=1, zorder=100)

plt.tick_params(which='both', direction='in')
plt.xscale('log')
plt.yscale('log')      
plt.xlabel(r'$M_p$')
# plt.xlabel(r"p")
plt.ylabel(r'$|\langle O \rangle -\langle O \rangle_I$|')
plt.legend(frameon=False)

title_save = 'Bias_vs_Mp'
plt.savefig(
        './' +(title_save + ".pdf"),
        dpi=3000,
        bbox_inches='tight',
    )