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

with open('./TMP/Sample_max','rb')as f:
    Dic_Max = pickle.load(f)

with open('./TMP/Sample_min','rb')as f:
    Dic_Min = pickle.load(f)
Keydic = {}
strlist = ['SP','PT1X','PT1Y','layer1','PT2X','PT2Y','layer2','Mea']
labels = [''.join(p) for p in itertools.product('IXYZ', repeat=4)]
labelsIZ = [''.join(p) for p in itertools.product('IZ', repeat=4)]
Plist0 = []
for key in strlist:
    P = (Dic_Max[key][0],Dic_Min[key][0])
    Plist0.append(P)
    Keydic.update({key:labels})
Keydic.update({'SP':labelsIZ})
Keydic.update({'Mea':labelsIZ})
def all_tuple_products(data):
    choices = itertools.product(*data)  # 枚举每个位置取哪个
    return [1-np.prod(choice) for choice in choices]

Plist = all_tuple_products(Plist0)
print(Plist)
print(np.mean(Plist))

def TotalErrorRate(Mp):
    p = np.random.choice(Plist,size = Mp)
    plist = np.random.binomial(1, p)
    p_estimation = sum(plist)/len(plist)
    return p_estimation

# PAll_all = TotalErrorRate(1000)
def Practical_ProcessedErrorSampler():
    Id = ''.join(['I' for i in range(4*len(strlist))])
    Error0 = {}
    k = np.random.geometric((1-2*PAll_all)/(1-PAll_all))
    flag1 = (-1)**(k-1)
    for key in strlist:
        Error0.update({key:'IIII'})
    if k == 0:
        return Error0,flag1
    else:
        pau = Pauli(''.join(list(Error0.values())))
        for j in range(k):
            pau1 = Id
            while pau1==Id:
                pau1 = ''
                for key in strlist:
                    random_value = np.random.choice([0, 1])
                    if random_value ==0:
                        valueList = Dic_Max[key]
                    else:
                        valueList = Dic_Min[key]
                    pau1 = pau1+''.join(np.random.choice(Keydic[key], p=valueList))
            pau = pau.dot(pau1)
        pau = pau.to_label()
        while not pau[0].isupper():
            pau = pau[1:]
        sta = 0
        for i in range(len(strlist)):
            Error0.update({strlist[i]:pau[sta:sta+4]})
            sta = sta +4 
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

def InserPT1X(cir,qubit):
    pauli = Pauli(Error['PT1X'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def InserPT1Y(cir,qubit):
    pauli = Pauli(Error['PT1Y'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('Y')@T
b= SuperOp(a)
cliy = b.to_instruction()
def InserPT2X(cir,qubit):
    pauli = Pauli(Error['PT2X'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def InserPT2Y(cir,qubit):
    pauli = Pauli(Error['PT2Y'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def Inserlayer1(cir,qubit):
    pauli = Pauli(Error['layer1'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def Inserlayer2(cir,qubit):
    global NH
    pauli = Pauli(Error['layer2'])
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def layer1(cir):
    cir.cx(0,1)
    cir.t(2)
    cir.h(3)

def layer2(cir):
    cir.t(0)
    cir.cx(1,2)
    cir.s(3)

def InserTwirlinglayer1(cir,qubit):
    a = np.random.choice([0,1,2,3])
    if a ==0:
        layer1(cir)
        Inserlayer1(cir,qubit)
    elif a==1:
        cir.z(2)
        layer1(cir)
        Inserlayer1(cir,qubit)
        cir.z(2)
    elif a==2:
        cir.append(clix,[2])
        
        InserPT1X(cir,qubit)
        layer1(cir)
        Inserlayer1(cir,qubit)
        cir.x(2)
    else:
        cir.append(cliy,[2])
        
        InserPT1Y(cir,qubit)
        layer1(cir)
        Inserlayer1(cir,qubit)
        cir.y(2)


def InserTwirlinglayer2(cir,qubit):
    a = np.random.choice([0,1,2,3])
    if a ==0:
        layer2(cir)
        Inserlayer2(cir,qubit)
    elif a==1:
        cir.z(0)
        layer2(cir)
        Inserlayer2(cir,qubit)
        cir.z(0)
    elif a==2:
        cir.append(clix,[0])
        
        InserPT2X(cir,qubit)
        layer2(cir)
        Inserlayer2(cir,qubit)
        cir.x(0)
    else:
        cir.append(cliy,[0])
        
        InserPT2Y(cir,qubit)
        layer2(cir)
        Inserlayer2(cir,qubit)
        cir.y(0)


def task():
    global Error
    Error,flag = Practical_ProcessedErrorSampler()
    qubits = [0,1,2,3]
    cir = QuantumCircuit(4)
    cir.h(0)
    cir.h(1)
    cir.h(2)
    cir.h(3)
    InserSP(cir,qubits)
    InserTwirlinglayer1(cir,qubits)
    InserTwirlinglayer2(cir,qubits)
    InserMex(cir,qubits)
    rho = DensityMatrix.from_instruction(cir)
    Ovalue = np.real(rho.expectation_value(Operator.from_label('IIXI')))
    res = Ovalue*flag
    return res

with open('./Result/Free','rb')as f1:
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
                p = np.mean(Plist)
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
        fname = './Result/SNI_vs_Mp'
        with open(fname, 'wb') as fp:
            pickle.dump(meanlist, fp)