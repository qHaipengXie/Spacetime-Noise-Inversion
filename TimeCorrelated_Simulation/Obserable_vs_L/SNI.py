from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SuperOp,Pauli,Operator
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
import numpy as np
import pickle
from mpi4py import MPI
L = 1
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
NumList = [2,2,3*L,3*L,3*L,4*L,2*L] ##The maximum number of operations for each operation in the maximum spacetime error model.
strList = ['mex','spx','TX','TY','T','H','cx']##The types of operations in the maximum spacetime error model.

def TotalErrorRate(Mp):  ##--Algorithm 2: Since only the number of errors is needed here, it is not necessary to generate specific error types.
##(Common case: If the samples are also to be used in the subsequent computational circuit, then specific error types must be generated. This approach can reduce the cost and works well in numerical simulations, but it is not conducive to rigorous theoretical analysis.)
    Mp0 = 0
    for i in range(Mp):
        random_value = np.random.choice([0, 1])
        if random_value ==0:
            valueList = valueList_max
        else:
            valueList = valueList_min
        flag = []
        for j in range(len(valueList)):
            for k in range(NumList[j]):
                flag.append(np.random.choice([0,1],p=[valueList[j][0],1-valueList[j][0]]))
        if sum(flag)!=0:
            Mp0 = Mp0 + 1
    return Mp0/Mp
  
def Practical_ProcessedErrorSampler(Fluctuation):   ##--Algorithm 9
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
    ##---- Construct an identity operator with the same format as that of the maximum spacetime error.
    pauList = []
    for i in range(len(NumList)):
        pau = ''
        for j in range(NumList[i]):
            pau = pau +'I'
        pauList.append(pau)
    pauList[-1] = pauList[-1]+pauList[-1]##The last type of operation is two-qubit gates.
    Id = ''.join(pauList) 

    Error = {}
    k = np.random.geometric((1-2*PAll_all)/(1-PAll_all))-1  ##Generate a non-negative integer k 
    flag1 = (-1)**k
    ##---- Sample the encoding and decoding errors of the maximum spacetime error model for error boosting, with the error rate consistent with the current computation circuit.
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
            a = a + ''.join(np.random.choice(keyIdle,p=valueIdle))
        Error.update({strList[i+1]:a})
    
    if k == 0:
        return Error,flag1
    else:
        ##SpacetimeErrorSampler  (Algorithm 1)
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
            pau = pau[1:] ##Eliminate any potential occurrences of '-' and 'i'.
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

def InserTError(cir,qubit):
    cir.append(T_Error_Pauli,[qubit])
    cir.append(T_Error_Coherent,[qubit])    
    return cir

def InserTwirlingT(cir,qubit):
    a = np.random.choice([0,1,2,3])
    if a ==0:
        cir.id(qubit)
        cir.t(qubit)
        InserTError(cir,qubit)
        InserT(cir,qubit)
        cir.id(qubit)
    elif a==1:
        cir.z(qubit)
        cir.t(qubit)
        InserTError(cir,qubit)
        InserT(cir,qubit)
        cir.z(qubit)
    elif a==2:
        cir.append(clix,qubit)
        cir.append(Single_qubit_Error,qubit)
        InserTX(cir,qubit)
        cir.t(qubit)
        InserTError(cir,qubit)
        InserT(cir,qubit)
        cir.x(qubit)
    else:
        cir.append(cliy,qubit)
        cir.append(Single_qubit_Error,qubit)
        InserTY(cir,qubit)
        cir.t(qubit)
        InserTError(cir,qubit)
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

n = np.array([0, 0, 1])  # 
n = n/np.linalg.norm(n)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

n_dot_sigma = n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z


p = 0.002
ped = 1/3*p
pst = p

r_min= 0.5
theta = np.sqrt(r_min*p / 2)
U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * n_dot_sigma
T_Error_Coherent_min = coherent_unitary_error(U)
T_Error_Pauli_min = DepoError(1/2*p*r_min,1)
Single_qubit_Error_min = DepoError(pst*r_min,1)
Two_qubit_Error_min = DepoError(pst*r_min,2)


r_max= 1.5
theta = np.sqrt(r_max*p / 2)
U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * n_dot_sigma
T_Error_Coherent_max = coherent_unitary_error(U)
T_Error_Pauli_max = DepoError(1/2*p*r_max,1)
Single_qubit_Error_max = DepoError(pst*r_max,1)
Two_qubit_Error_max = DepoError(pst*r_max,2)

simulator = AerSimulator()






def task():
    global Error
    Fluctuation = np.random.choice([0,1])
    Error,flag = Practical_ProcessedErrorSampler(Fluctuation)
    global Single_qubit_Error,Two_qubit_Error,T_Error_Pauli,T_Error_Coherent
    if Fluctuation==0:
        Single_qubit_Error,Two_qubit_Error,T_Error_Pauli,T_Error_Coherent = Single_qubit_Error_max,Two_qubit_Error_max,T_Error_Pauli_max,T_Error_Coherent_max
    else:
        Single_qubit_Error,Two_qubit_Error,T_Error_Pauli,T_Error_Coherent = Single_qubit_Error_min,Two_qubit_Error_min,T_Error_Pauli_min,T_Error_Coherent_min
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.append(Single_qubit_Error,[0])
    qc.append(Single_qubit_Error,[1])
    InsersPx(qc,[0])
    InsersPx(qc,[1])
    for i in range(L):
        qc.cx(0,1)
        qc.append(Two_qubit_Error,[0,1])
        InserCx(qc,[0,1])
        InserTwirlingT(qc,[1])
        qc.cx(0,1)
        qc.append(Two_qubit_Error,[0,1])
        InserCx(qc,[0,1])
        qc.h(0)
        qc.append(Single_qubit_Error,[0])
        InserH(qc,[0])
        InserTwirlingT(qc,[0])
        qc.h(0)
        qc.append(Single_qubit_Error,[0])
        InserH(qc,[0])
        qc.h(1)
        qc.append(Single_qubit_Error,[1])
        InserH(qc,[1])
        InserTwirlingT(qc,[1])
        qc.h(1)
        qc.append(Single_qubit_Error,[1])
        InserH(qc,[1])
    InserMex(qc,[0])
    InserMex(qc,[1])
    qc.append(Single_qubit_Error,[0])
    qc.append(Single_qubit_Error,[1])
    qc.h(0)
    qc.h(1)
    qc.measure_all()
    result = simulator.run(qc, shots=1).result().get_counts()
    # Ovalue = -1 if list(result.keys())[0].count('0') % 2 == 1 else 1
    Ovalue = -1 if list(result.keys())[0][0] == '1' else 1
    res = Ovalue*flag
    return res




if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank ==0:
        np.random.seed(L)
        seeds = np.random.randint(0, 10000, size=size)
    else:
        seeds = None
    root_seed = comm.bcast(seeds,root=0)
    np.random.seed(root_seed[rank])
    Mp = 1000000
    PAll = TotalErrorRate(Mp)
    PAllList = comm.gather(PAll,root=0)
    if rank ==0:
        PAll_all = sum(PAllList)/len(PAllList)
    else:
        PAll_all = 0
    PAll_all = comm.bcast(PAll_all,root = 0)
    gamma = 1/(1-2*PAll_all)
    M = int(Mp/10)
    local_results0 = [task()*gamma for i in range(M)]
    local_results = sum(local_results0)/M

    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        mean = sum(gathered_results)/len(gathered_results)
        std = np.sqrt(gamma**2-mean**2)/(M*size)
        fname = './Plot/Data/SNI_'+str(L)
        with open(fname, 'wb') as fp:
            pickle.dump((mean,std), fp)

