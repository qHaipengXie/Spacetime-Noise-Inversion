from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator,SuperOp,DensityMatrix,Pauli,Chi
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
import itertools
import numpy as np
import pickle
import time
import pickle   
from mpi4py import MPI

n = 4
with open('./TMP/Sample_n='+str(n),'rb')as f:
    Dic = pickle.load(f)

# strlist = ['SP','Stab','cz1','cz2','TLayer','Mea']

strlist = list(Dic.keys())
# Dic.update({'Stab1':Dic['Stab']})
# Dic.update({'Stab2':Dic['Stab']})
# Dic.update({'TLayer1':Dic['TLayer']})
# Dic.update({'TLayer2':Dic['TLayer']})
# del Dic['Stab']
# del Dic['TLayer']
Keydic = {}

Plist0 = []
for key in strlist:
    P = Dic[key][0][1]
    keylist = [Dic[key][i][0] for i in range(len(Dic[key]))]
    Plist0.append(P)
    Keydic.update({key:keylist})

P0 = 1-np.prod(Plist0)

def TotalErrorRate(Mp):
    
    plist = np.random.binomial(1, P0,size = Mp)
    p_estimation = sum(plist)/len(plist)
    return p_estimation

# PAll_all = TotalErrorRate(1000)
# print(PAll_all)
def Practical_ProcessedErrorSampler():
    Id = ''.join(['I' for i in range(n*len(strlist))])
    Error0 = {}
    k = np.random.geometric((1-2*PAll_all)/(1-PAll_all))
    flag1 = (-1)**(k-1)
    pau = Pauli(Id)
    for j in range(k):
        pau1 = Id
        while pau1==Id:
            pau1 = ''
            for key in strlist:

                valueList = [x[1] for x in Dic[key]]

                pau1 = pau1+''.join(np.random.choice(Keydic[key], p=valueList))
        pau = pau.dot(pau1)
    pau = pau.to_label()
    while not pau[0].isupper():
        pau = pau[1:]
    sta = 0
    for i in range(len(strlist)):
        Error0.update({strlist[i]:pau[sta:sta+n]})
        sta = sta +n
    return Error0,flag1
# print(TotalErrorRate(1000))
def InserSP(cir,qubit):
    pauli = Pauli(Error['SP'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

    
def InserMex(cir,qubit):
    pauli = Pauli(Error['Mea'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('X')@T
b= SuperOp(a)
clix = b.to_instruction()

def InserStab1(cir,qubit):
    pauli = Pauli(Error['Stab1_'+str(np.random.choice(12))])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir


qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('Y')@T
b= SuperOp(a)
cliy = b.to_instruction()
def InserStab2(cir,qubit):
    pauli = Pauli(Error['Stab2_'+str(np.random.choice(12))])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir


def InserTlayer1(cir,qubit):
    pauli = Pauli(Error['TLayer1'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def InserTlayer2(cir,qubit):
    pauli = Pauli(Error['TLayer2'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def Insercz1(cir,qubit):
    pauli = Pauli(Error['cz1'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def Insercz2(cir,qubit):
    pauli = Pauli(Error['cz2'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

# def Tlayer1(cir):
#     for i in range(n):
#         cir.t(i)

# def Tlayer2(cir):
#     for i in range(n):
#         cir.t(i)

labels = [''.join(p) for p in itertools.product('IXYZ', repeat=n)]
labelsIZ = [''.join(p) for p in itertools.product('IZ', repeat=n//2)]
def pauli_add(cir,pauli,label):
    if pauli == 'X':
        cir.x(label)
    elif pauli == 'Y':
        cir.y(label)
    elif pauli == 'Z':
        cir.z(label)
    else:
        None

def InserTwirlinglayer1(cir,qubit):
    pauli = np.random.choice(labels)
    pauli_T = pauli[0]+pauli[2]
    if pauli_T in labelsIZ:
        for i in range(n):
            pauli_add(cir,pauli[i],i)
            if i%2==0:
                cir.t(i)
        InserTlayer1(cir,qubits)
        for i in range(n):
            pauli_add(cir,pauli[i],i)
           
    else:
        for i in range(n):
            if i%2==0:
                if pauli[i] == 'X':
                    cir.append(clix,[i])
                elif pauli[i] == 'Y':
                    cir.append(cliy,[i])
                elif pauli[i] == 'Z':
                    cir.z(i)
                else:
                    None
            else:
                pauli_add(cir,pauli[i],i)
        InserStab1(cir,qubits)
        for i in range(n):
            if i%2==0:
                cir.t(i)
        InserTlayer1(cir,qubits)
        for i in range(n):
            pauli_add(cir,pauli[i],i)
        



def InserTwirlinglayer2(cir,qubit):
    pauli = np.random.choice(labels)
    pauli_T = pauli[1]+pauli[3]
    if pauli_T in labelsIZ:
        for i in range(n):
            pauli_add(cir,pauli[i],i)
            if i%2==1:
                cir.t(i)
        InserTlayer2(cir,qubits)
        for i in range(n):
            pauli_add(cir,pauli[i],i)
           
    else:
        for i in range(n):
            if i%2==1:
                if pauli[i] == 'X':
                    cir.append(clix,[i])
                elif pauli[i] == 'Y':
                    cir.append(cliy,[i])
                elif pauli[i] == 'Z':
                    cir.z(i)
                else:
                    None
            else:
                pauli_add(cir,pauli[i],i)
        InserStab2(cir,qubits)
        for i in range(n):
            if i%2==1:
                cir.t(i)
        InserTlayer2(cir,qubits)
        for i in range(n):
            pauli_add(cir,pauli[i],i)

qubits = range(n)

ob = 'XXXX'
def task():
    global Error
    Error,flag = Practical_ProcessedErrorSampler()
    cir = QuantumCircuit(n)
    for i in range(n):
        cir.h(i)
    InserSP(cir,qubits)
    InserTwirlinglayer1(cir,qubits)
    for i in range(n//2):
        cir.cz(2*i,2*i+1)
    Insercz1(cir,qubits)
    InserTwirlinglayer2(cir,qubits)
    for i in range((n-1)//2):
        cir.cz(2*i+1,2*i+2)
    Insercz2(cir,qubits)
    InserMex(cir,qubits)
    rho = DensityMatrix.from_instruction(cir)
    
    Ovalue = np.real(rho.expectation_value(Operator.from_label(ob)))
    res = Ovalue*flag
    return res

with open('./Result/Free_n='+str(n),'rb')as f1:
    c = np.real(pickle.load(f1))


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank ==0:
        meanlist = []
    else:
        meanlist = None
      
    M = 100
    for i in range(20):
        Mp = 2**(i+1)
        if rank ==0:
            np.random.seed(Mp)
            seeds = np.random.randint(0, 10000, size=size)
        else:
            seeds = None
        root_seed = comm.bcast(seeds,root=0)
        np.random.seed(root_seed[rank])
        res0 = []
        for j in range(100): ## 100 instances
            star = time.time()
            PAll = TotalErrorRate(Mp)
            
            PAllList = comm.gather(PAll,root=0)
            if rank ==0:
                PAll_all = sum(PAllList)/len(PAllList)
                
            else:
                PAll_all = 0
            PAll_all = comm.bcast(PAll_all,root = 0)
            local_results = [task() for i in range(M)]
            local_results = [sum(local_results)/len(local_results)]

            gathered_results = comm.gather(local_results, root=0)
            
            if rank == 0:
                p = P0
                gamma = 1/((1-2*PAll_all)*(1-PAll_all))
                results = [item for sublist in gathered_results for item in sublist]
                res = np.abs(gamma*sum(results)*(p - PAll_all)/len(results) -(p - PAll_all)*c/(1-PAll_all))
                res0.append(res)
            else:
                None
            
        if rank == 0:
            
            meanlist.append((np.mean(res0),np.std(res0)))
        else:
            None

    if rank ==0:
        print(meanlist)
        end = time.time()
        print(end-star)
        fname = './Result/SNI_vs_Mp_n='+str(n)
        with open(fname, 'wb') as fp:
            pickle.dump(meanlist, fp)