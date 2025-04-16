from qiskit import QuantumCircuit,transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SuperOp,Pauli,Operator
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


def Error_model_learning(Mp):
    valueList_estimation = []
    a = []
    for i in range(len(strList)):
        a.append({char: 0 for char in keyList[i]})
    for i in range(Mp):
        random_value = np.random.choice([0, 1])
        if random_value ==0:
            valueList = valueList_max
        else:
            valueList = valueList_min
        for j in range(len(valueList)):
            for k in range(NumList[j]):
                sample_pauli = np.random.choice(keyList[j],p=valueList[j])
                a[j][sample_pauli] += 1
    for i in range(len(a)):
        b = list(a[i].values())
        c = sum(b)
        d = [x/c for x in b]
        valueList_estimation.append(d)
    return valueList_estimation



qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a0=T.adjoint()@Operator.from_label('X')@T
b0= SuperOp(a0)


qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a1=T.adjoint()@Operator.from_label('Y')@T
b1= SuperOp(a1)


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
En_Decoding_Error_min = DepoError(ped*r_min,1)

r_max= 1.5
theta = np.sqrt(r_max*p / 2)
U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * n_dot_sigma
T_Error_Coherent_max = coherent_unitary_error(U)
T_Error_Pauli_max = DepoError(1/2*p*r_max,1)
Single_qubit_Error_max = DepoError(pst*r_max,1)
Two_qubit_Error_max = DepoError(pst*r_max,2)
En_Decoding_Error_max = DepoError(ped*r_max,1)


SuperMatdic_max = {}
SuperMatdic_max.update({'Gate_H':np.array(SuperOp(Operator.from_label('H')))})
SuperMatdic_max.update({'Gate_X':np.array(SuperOp(Operator.from_label('X')))})
SuperMatdic_max.update({'Gate_Y':np.array(SuperOp(Operator.from_label('Y')))})
SuperMatdic_max.update({'Gate_Z':np.array(SuperOp(Operator.from_label('Z')))})
SuperMatdic_max.update({'Gate_T':np.array(SuperOp(Operator.from_label('T')))})
SuperMatdic_max.update({'Gate_cx':np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]))})
SuperMatdic_max.update({'Gate_TX':np.array(b0)})
SuperMatdic_max.update({'Gate_TY':np.array(b1)})

SuperMatdic_min = SuperMatdic_max.copy()

SuperMatdic_max.update({'Single_qubit_Error':np.array(SuperOp(Single_qubit_Error_max))})
SuperMatdic_max.update({'Two_qubit_Error':np.array(SuperOp(Two_qubit_Error_max))})
SuperMatdic_max.update({'En_Decoding_Error':np.array(SuperOp(En_Decoding_Error_max))})
SuperMatdic_max.update({'T_Error_Pauli':np.array(SuperOp(T_Error_Pauli_max))})
SuperMatdic_max.update({'T_Error_Coherent':np.array(SuperOp(T_Error_Coherent_max))})

SuperMatdic_min.update({'Single_qubit_Error':np.array(SuperOp(Single_qubit_Error_min))})
SuperMatdic_min.update({'Two_qubit_Error':np.array(SuperOp(Two_qubit_Error_min))})
SuperMatdic_min.update({'En_Decoding_Error':np.array(SuperOp(En_Decoding_Error_min))})
SuperMatdic_min.update({'T_Error_Pauli':np.array(SuperOp(T_Error_Pauli_min))})
SuperMatdic_min.update({'T_Error_Coherent':np.array(SuperOp(T_Error_Coherent_min))})


commute_mat = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
commute_mat0 = np.kron(np.eye(2),commute_mat)
commute_mat1 = np.kron(commute_mat,np.eye(2))
O0 = Operator.from_label('ZI').to_matrix()
O = O0.flatten()
def FunRes():
    mat = np.zeros((16,1))
    mat[0] = 1
    Error_dic = {}
    for i in range(len(strList)):
        a = [Error[strList[i]][1][j]*SuperOp(Operator.from_label(Error[strList[i]][0][j])) for j in range(len(Error[strList[i]][1]))]
        b = np.array(sum(a))
        Error_dic.update({strList[i]:np.linalg.inv(b)})

    SuperMatdic = SuperMatdic_max
    a0 = Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']
    a1 = SuperMatdic['Gate_Z']@Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']@SuperMatdic['Gate_Z']
    a2 = SuperMatdic['Gate_X']@Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']@Error_dic['TX']@SuperMatdic['Single_qubit_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['Gate_TX']
    a3 = SuperMatdic['Gate_Y']@Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']@Error_dic['TY']@SuperMatdic['Single_qubit_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['Gate_TY']
    SuperMatdic.update({'TwirlingT':0.25*sum([a0,a1,a2,a3])})
    SuperMatdic.update(Error_dic)
    for i in range(len(qc)):
        mat0 = SuperMatdic[qc[i][0]]
        if type(qc[i][1])== int:
            if qc[i][1]==0:
                mat0 = np.kron(mat0,np.eye(2))
                mat0 = commute_mat0@mat0@commute_mat0.T
                mat0 = np.kron(mat0,np.eye(2))
            else:
                mat0 = np.kron(np.eye(2),mat0)
                mat0 = commute_mat1@mat0@commute_mat1.T
                mat0 = np.kron(np.eye(2),mat0)
        else:
            None
        mat = mat0@mat
    Ovalue_max = O@mat

    mat = np.zeros((16,1))
    mat[0] = 1
    SuperMatdic = SuperMatdic_min
    a0 = Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']
    a1 = SuperMatdic['Gate_Z']@Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']@SuperMatdic['Gate_Z']
    a2 = SuperMatdic['Gate_X']@Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']@Error_dic['TX']@SuperMatdic['Single_qubit_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['Gate_TX']
    a3 = SuperMatdic['Gate_Y']@Error_dic['T']@SuperMatdic['T_Error_Coherent']@SuperMatdic['T_Error_Pauli']@SuperMatdic['Gate_T']@Error_dic['TY']@SuperMatdic['Single_qubit_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['En_Decoding_Error']@SuperMatdic['Gate_TY']
    SuperMatdic.update({'TwirlingT':0.25*sum([a0,a1,a2,a3])})
    SuperMatdic.update(Error_dic)
    for i in range(len(qc)):
        mat0 = SuperMatdic[qc[i][0]]
        if type(qc[i][1])== int:
            if qc[i][1]==0:
                mat0 = np.kron(mat0,np.eye(2))
                mat0 = commute_mat0@mat0@commute_mat0.T
                mat0 = np.kron(mat0,np.eye(2))
            else:
                mat0 = np.kron(np.eye(2),mat0)
                mat0 = commute_mat1@mat0@commute_mat1.T
                mat0 = np.kron(np.eye(2),mat0)
        else:
            None
        mat = mat0@mat
    Ovalue_min = O@mat

    return (Ovalue_max+Ovalue_min)/2
    
    

qc = []
qc.append(('Gate_H',0))
qc.append(('Gate_H',1))
qc.append(('Single_qubit_Error',0))
qc.append(('Single_qubit_Error',1))
qc.append(('spx',0))
qc.append(('spx',1))
for i in range(L):
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('Gate_cx',[0,1]))
    qc.append(('Two_qubit_Error',[0,1]))
    qc.append(('cx',[0,1]))
    qc.append(('En_Decoding_Error',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('TwirlingT',1))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('Gate_cx',[0,1]))
    qc.append(('Two_qubit_Error',[0,1]))
    qc.append(('cx',[0,1]))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',0))     
    qc.append(('Gate_H',0))
    qc.append(('Single_qubit_Error',0))
    qc.append(('H',0))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',0))
    qc.append(('TwirlingT',0))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',0))
    qc.append(('Gate_H',0))
    qc.append(('Single_qubit_Error',0))
    qc.append(('H',0))
    qc.append(('En_Decoding_Error',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('Gate_H',1))
    qc.append(('Single_qubit_Error',1))
    qc.append(('H',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('TwirlingT',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('Gate_H',1))
    qc.append(('Single_qubit_Error',1))
    qc.append(('H',1))
qc.append(('En_Decoding_Error',0))
qc.append(('En_Decoding_Error',0))
qc.append(('En_Decoding_Error',1))
qc.append(('En_Decoding_Error',1))
qc.append(('mex',0))
qc.append(('mex',1))
qc.append(('Single_qubit_Error',0))
qc.append(('Single_qubit_Error',1))
qc.append(('Gate_H',0))
qc.append(('Gate_H',1))

def task():
    Ovalue = np.real(FunRes())
    return Ovalue


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
    le = [1+i for i in range(15)]
    for i in le:
        num = i
        Mp = 2**num
        if rank ==0:
            np.random.seed(num)
            seeds = np.random.randint(0, 10000, size=size)
        else:
            seeds = None
        root_seed = comm.bcast(seeds,root=0)
        np.random.seed(root_seed[rank])
        res0 = []
        star = time.time()
        for j in range(100):## 100 instances
            PAll = Error_model_learning(Mp)
            PAllList = comm.gather(PAll,root=0)
            if rank ==0:
                valueList_est = []
                for i in range(len(PAllList[0])):
                    row = [sum(x)/len(PAllList) for x in zip(*[lst[i] for lst in PAllList])]
                    valueList_est.append(row)
                Error = {}
                for i in range(len(strList)):
                    Error.update({strList[i]:(keyList[i],valueList_est[i])})
                res = task()
                # print(res)
                res0.append(np.abs(res-c))
            else:
                None
        if rank ==0:
            meanlist.append((np.mean(res0),np.std(res0)))
        else:
            None
    if rank ==0:
        print(meanlist)
        fname = './Plot/cPEC_Mp'
        with open(fname, 'wb') as fp:
            pickle.dump(res, fp)


