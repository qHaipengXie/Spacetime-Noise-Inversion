from qiskit import QuantumCircuit,transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SuperOp,Pauli,Operator,DensityMatrix
from qiskit_aer.noise import NoiseModel,pauli_error,depolarizing_error,coherent_unitary_error
import numpy as np
import time
import pickle   
from mpi4py import MPI
L = 8
title = './../TMP/SamRate_max_'
with open(title + 'Idle','rb')as f1:
    rateIdle = pickle.load(f1)
keyIdle = list(rateIdle.keys())
valueIdle_max = list(rateIdle.values())

with open(title + 'Cx','rb')as f1:
    rateCx = pickle.load(f1)
keyCx = list(rateCx.keys())
valueCx = list(rateCx.values())
PCx = 1-valueCx[0]


with open(title + 'Mex','rb')as f1:
    rateMex = pickle.load(f1)
keyMex = list(rateMex.keys())
valueMex = list(rateMex.values())
PMex = 1-valueMex[0]


with open(title + 'sPx','rb')as f1:
    ratesPx = pickle.load(f1)
keysPx = list(ratesPx.keys())
valuesPx = list(ratesPx.values())
PsPx = 1-valuesPx[0]

with open(title + 'TX','rb')as f1:
    rateTX = pickle.load(f1)
keyTX = list(rateTX.keys())
valueTX = list(rateTX.values())
PTX = 1-valueTX[0]

with open(title + 'TY','rb')as f1:
    rateTY = pickle.load(f1)
keyTY = list(rateTY.keys())
valueTY = list(rateTY.values())
PTY = 1-valueTY[0]


with open(title + 'T','rb')as f1:
    rateT = pickle.load(f1)
keyT = list(rateT.keys())
valueT = list(rateT.values())
PT = 1-valueT[0]

with open(title + 'H','rb')as f1:
    rateH = pickle.load(f1)
keyH = list(rateH.keys())
valueH = list(rateH.values())
PH = 1-valueH[0]

valueList_max = [valueMex,valuesPx,valueTX,valueTY,valueT,valueH,valueCx]

title = './../TMP/SamRate_min_'
with open(title + 'Idle','rb')as f1:
    rateIdle = pickle.load(f1)
keyIdle = list(rateIdle.keys())
valueIdle_min = list(rateIdle.values())

with open(title + 'Cx','rb')as f1:
    rateCx = pickle.load(f1)
keyCx = list(rateCx.keys())
valueCx = list(rateCx.values())
PCx = 1-valueCx[0]


with open(title + 'Mex','rb')as f1:
    rateMex = pickle.load(f1)
keyMex = list(rateMex.keys())
valueMex = list(rateMex.values())
PMex = 1-valueMex[0]


with open(title + 'sPx','rb')as f1:
    ratesPx = pickle.load(f1)
keysPx = list(ratesPx.keys())
valuesPx = list(ratesPx.values())
PsPx = 1-valuesPx[0]

with open(title + 'TX','rb')as f1:
    rateTX = pickle.load(f1)
keyTX = list(rateTX.keys())
valueTX = list(rateTX.values())
PTX = 1-valueTX[0]

with open(title + 'TY','rb')as f1:
    rateTY = pickle.load(f1)
keyTY = list(rateTY.keys())
valueTY = list(rateTY.values())
PTY = 1-valueTY[0]


with open(title + 'T','rb')as f1:
    rateT = pickle.load(f1)
keyT = list(rateT.keys())
valueT = list(rateT.values())
PT = 1-valueT[0]

with open(title + 'H','rb')as f1:
    rateH = pickle.load(f1)
keyH = list(rateH.keys())
valueH = list(rateH.values())
PH = 1-valueH[0]

valueList_min = [valueMex,valuesPx,valueTX,valueTY,valueT,valueH,valueCx]

keyList = [keyMex,keysPx,keyTX,keyTY,keyT,keyH,keyCx]
NumList = [2,2,3*L,3*L,3*L,4*L,2*L]
strList = ['mex','spx','TX','TY','T','H','cx']


P_max = 0
P_min = 0
for i in range(len(valueList_max)):
    P_max = P_max + NumList[i]*np.log(valueList_max[i][0])
    P_min = P_min + NumList[i]*np.log(valueList_min[i][0])
P_max = 1-np.exp(P_max)
P_min = 1-np.exp(P_min)
def TotalErrorRate(Mp):
    p = np.random.choice([P_max,P_min],size = Mp)
    plist = np.random.binomial(1, p)
    p_estimation = sum(plist)/len(plist)
    return p_estimation


def Practical_ProcessedErrorSampler(Fluctuation):
    global Ncx
    global Nmex
    global Nspx
    global NT
    global NTX
    global NTY
    global NH

    Ncx = 0
    Nmex = 0
    Nspx = 0
    NT = 0
    NTX = 0
    NTY = 0
    NH = 0
    pauList = []
    for i in range(len(NumList)):
        pau = ''
        for j in range(NumList[i]):
            pau = pau +'I'
        pauList.append(pau)
    pauList[-1] = pauList[-1]+pauList[-1]
    Id = ''.join(pauList)
    Error = {}
    k = np.random.geometric((1-2*PAll_all)/(1-PAll_all))
    flag1 = (-1)**(k-1)
    if Fluctuation ==0:
        valueIdle = valueIdle_max
    else:
        valueIdle = valueIdle_min
    a = ''
    for j in range(len(pauList[0])):
        a = a + ''.join('I')
    Error.update({strList[0]:a})
    for i in range(len(pauList)-1):
        a = ''
        for j in range(len(pauList[i+1])):
            a = a + 'I'
        Error.update({strList[i+1]:a})
    if k == 0:
        return Error,flag1
    else:
        pau = Pauli(''.join(list(Error.values())))
        for j in range(k):
            pau1 = Id
            while pau1==Id:
                random_value = np.random.choice([0, 1])
                if random_value ==0:
                    valueList = valueList_max
                else:
                    valueList = valueList_min
                pau1 = ''
                for i in range(len(pauList)):
                    pau1 =pau1+''.join(np.random.choice(keyList[i], size=NumList[i], p=valueList[i],replace=True))
            pau = pau.dot(pau1)
        pau = pau.to_label()
        while not pau[0].isupper():
            pau = pau[1:]
        sta = 0
        for i in range(len(pauList)):
            Error.update({strList[i]:pau[sta:sta+len(pauList[i])]})
            sta = sta +len(pauList[i])
        return Error,flag1
    
def InsersPx(cir,qubit):
    global Nspx
    pauli = Pauli(Error['spx'][Nspx])
    Nspx=Nspx+1
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

    
def InserMex(cir,qubit):
    global Nmex
    pauli = Pauli(Error['mex'][Nmex])
    Nmex=Nmex+1
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('X')@T
b= SuperOp(a)
clix = b.to_instruction()
def InserTX(cir,qubit):
    global NTX
    pauli = Pauli(Error['TX'][NTX])
    NTX=NTX+1
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('Y')@T
b= SuperOp(a)
cliy = b.to_instruction()
def InserTY(cir,qubit):
    global NTY
    pauli = Pauli(Error['TY'][NTY])
    NTY=NTY+1
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def InserT(cir,qubit):
    global NT
    pauli = Pauli(Error['T'][NT])
    NT=NT+1
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir

def InserH(cir,qubit):
    global NH
    pauli = Pauli(Error['H'][NH])
    NH=NH+1
    pauli = pauli.to_instruction()
    cir.append(pauli,qubit)
    return cir



def InserTwirlingT(cir,qubit):
    a = np.random.choice([0,1,2,3])
    if a ==0:
        cir.id(qubit)
        cir.t(qubit)
        
        InserT(cir,qubit)
        cir.id(qubit)
    elif a==1:
        cir.z(qubit)
        cir.t(qubit)
        
        InserT(cir,qubit)
        cir.z(qubit)
    elif a==2:
        cir.append(clix,qubit)
        
        InserTX(cir,qubit)
        cir.t(qubit)
        
        InserT(cir,qubit)
        cir.x(qubit)
    else:
        cir.append(cliy,qubit)
        
        InserTY(cir,qubit)
        cir.t(qubit)
        
        InserT(cir,qubit)
        cir.y(qubit)

def InserCx(cir,qubit):
    global Ncx
    pauli = Pauli(Error['cx'][2*Ncx:2*Ncx+2])
    Ncx=Ncx+1
    pauli = pauli.to_instruction()
    qubit = qubit[::-1]
    cir.append(pauli,qubit)
    return cir

def DepoError(p,n):
    noise = depolarizing_error(p,n)
    return noise


def task():
    global Error
    Fluctuation = np.random.choice([0,1])
    Error,flag = Practical_ProcessedErrorSampler(Fluctuation)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)

    InsersPx(qc,[0])
    InsersPx(qc,[1])
    for i in range(L):
        qc.cx(0,1)
        
        InserCx(qc,[0,1])
        InserTwirlingT(qc,[1])
        qc.cx(0,1)
        
        InserCx(qc,[0,1])
        qc.h(0)
        
        InserH(qc,[0])
        InserTwirlingT(qc,[0])
        qc.h(0)
        
        InserH(qc,[0])
        qc.h(1)
        
        InserH(qc,[1])
        InserTwirlingT(qc,[1])
        qc.h(1)
        
        InserH(qc,[1])
    InserMex(qc,[0])
    InserMex(qc,[1])
    
    qc.h(0)
    qc.h(1)
    rho = DensityMatrix.from_instruction(qc)
    Ovalue = np.real(rho.expectation_value(Operator.from_label('ZI')))
    res = Ovalue*flag
    return res


with open('./Plot/ErrorFree','rb')as f1:
    c = np.real(pickle.load(f1)[-1])


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
                p = (P_min+P_max)/2
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
        end = time.time()
        print(end-star)
        fname = './Plot/SNI_Mp'
        with open(fname, 'wb') as fp:
            pickle.dump(meanlist, fp)

