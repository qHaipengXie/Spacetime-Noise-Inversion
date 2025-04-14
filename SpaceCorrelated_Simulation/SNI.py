import numpy as np
import pickle
from mpi4py import MPI
def Pestimation(Mp):
        Blist = np.random.binomial(1,P0,Mp)
        return sum(Blist)/Mp
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank ==0:
    np.random.seed(6)
    seeds = np.random.randint(0, 10000, size=size)
else:
    seeds = None
root_seed = comm.bcast(seeds,root=0)
np.random.seed(root_seed[rank])

p = 0.01
n = range(20)
Mp = 1000000
M = int(Mp/10)
res = []
gammalist = []
for i in n:
    qubits = i+1
    P0 = p*4**qubits/(4**(qubits)-1)
    P = Pestimation(Mp)
    Pall = comm.gather(P,root=0)
    if rank==0:
        P = sum(Pall)/len(Pall)
    else:
        P = 0
    P = comm.bcast(P,root=0)
    res0 = 0
    # P = Pestimation(Mp)
    for j in range(M):
        res0_one = 1
        flag = np.random.choice([0,1],size=1,p=[P0,1-P0])[0]
        if flag ==0:
            pau = np.random.choice([0,1],size=qubits,p=[0.5,0.5])
            res0_one = res0_one if sum(pau)%2==0 else -res0_one
        else:
            None

        k = np.random.geometric((1-2*P)/(1-P))-1
        sig = (-1)**k
        if k == 0:
            None
        else:
            paulist = np.random.choice([0,1],size=k,p=[0.5,0.5])
            res0_one = res0_one if sum(paulist)%2==0 else -res0_one
        res0+=sig*res0_one
    gamma = 1/(1-2*P)
    gammalist.append(gamma)
    mean = gamma*res0/M
    res.append(mean)
res_all = comm.gather(res,root=0)

if rank == 0:
    Result = []
    for i in range(len(res_all[0])):
        mean = sum([x[i] for x in res_all])/size
        std = np.sqrt((gammalist[i]**2-mean**2)/(M*size))
        Result.append((mean,std))
    print(Result)
    with open('./Result/SNI_x_n','wb')as f:
        pickle.dump(Result,f)



        
        



