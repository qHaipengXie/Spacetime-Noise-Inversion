import numpy as np
import pickle
from mpi4py import MPI
def Pestimation(Mp,qubits):
    Blist = np.random.binomial(1,P0,Mp)
    plist = [[0,0,0,0] for i in range(qubits)]
    for i in range(len(Blist)):
        if Blist[i] ==0:
            None
        else:
            res = ''.join(np.random.choice(['I','X','Y','Z'],size=qubits))
            for j in range(len(res)):
                if res[j] == 'X':
                    plist[j][1]+=1
                elif res[j] == 'Y':
                    plist[j][2]+=1
                elif res[j] == 'Z':
                    plist[j][3]+=1
    plist1 = np.array(plist)/Mp
    for i in range(qubits):
        plist1[i][0] = 1-sum(plist1[i])
    return plist1

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
gammalist_all_qubits = []
for i in n:
    qubits = i+1
    P0 = p*4**qubits/(4**(qubits)-1)
    P = Pestimation(Mp,qubits)
    Pall = comm.gather(P,root=0)
    if rank==0:
        P = sum(Pall)/size
    else:
        P = 0
    P = comm.bcast(P,root=0)
    inv_P_list = []
    gammalist = []
    signlist = []
    Har = 1/4*np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])
    for j in range(len(P)):
        a = np.array([1,1/(P[j][0]+P[j][1]-P[j][2]-P[j][3]),1/(P[j][0]+P[j][2]-P[j][1]-P[j][3]),1/(P[j][0]+P[j][3]-P[j][1]-P[j][2])])
        b = list(Har@a)
        c = np.abs(b)
        gamma = sum(c)
        gammalist.append(gamma)
        signlist.append(np.sign(b))
        inv_P_list.append([x/gamma for x in c])
    
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
        for r in range(qubits):
            pau = np.random.choice(range(4),p=inv_P_list[r])
            sig = signlist[r][pau]
            if pau == 0 or pau== 3 :
                None
            else:
                res0_one = -res0_one
            res0_one = res0_one*sig
        res0 = res0 + res0_one


    gammalist_all_qubits.append(np.prod(gammalist))
    mean = np.prod(gammalist)*res0/M
    res.append(mean)

res_all = comm.gather(res,root=0)

if rank == 0:
    Result = []
    for i in range(len(res_all[0])):
        mean = sum([x[i] for x in res_all])/size
        std = np.sqrt((gammalist_all_qubits[i]**2-mean**2)/(M*size))
        Result.append((mean,std))
    print(Result)
    print(gammalist_all_qubits)
    with open('./Plot/cPEC_x_n','wb')as f:
        pickle.dump(Result,f)

        
        






