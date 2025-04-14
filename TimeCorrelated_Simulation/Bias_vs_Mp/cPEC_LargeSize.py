from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SuperOp,Operator
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
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


# def Error_model_learning(L,Mp):
#     valueList_estimation = []
#     a = []
#     for i in range(len(strList)):
#         a.append({char: 0 for char in keyList[i]})
#     for i in range(Mp):
#         random_value = np.random.choice([0, 1])
#         if random_value ==0:
#             valueList = valueList_max
#         else:
#             valueList = valueList_min
#         for j in range(len(valueList)):
#             for k in range(NumList[j]):
#                 sample_pauli = np.random.choice(keyList[j],p=valueList[j])
#                 a[j][sample_pauli] += 1
#     for i in range(len(a)):
#         b = list(a[i].values())
#         c = sum(b)
#         d = [x/c for x in b]
#         valueList_estimation.append(d)
#     return valueList_estimation



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
MagError2_min = coherent_unitary_error(U)
MagError_min = DepoError(1/2*p*r_min,1)
sQError_min = DepoError(pst*r_min,1)
tQError_min = DepoError(pst*r_min,2)
edQError_min = DepoError(ped*r_min,1)

r_max= 1.5
theta = np.sqrt(r_max*p / 2)
U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * n_dot_sigma
MagError2_max = coherent_unitary_error(U)
MagError_max = DepoError(1/2*p*r_max,1)
sQError_max = DepoError(pst*r_max,1)
tQError_max = DepoError(pst*r_max,2)
edQError_max = DepoError(ped*r_max,1)
simulator = AerSimulator()

SuperMatdic_max = {}
SuperMatdic_max.update({'Gateh':np.array(SuperOp(Operator.from_label('H')))})
SuperMatdic_max.update({'GateX':np.array(SuperOp(Operator.from_label('X')))})
SuperMatdic_max.update({'GateY':np.array(SuperOp(Operator.from_label('Y')))})
SuperMatdic_max.update({'GateZ':np.array(SuperOp(Operator.from_label('Z')))})
SuperMatdic_max.update({'GateT':np.array(SuperOp(Operator.from_label('T')))})
SuperMatdic_max.update({'Gatecx':np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]))})
SuperMatdic_max.update({'GateTX':np.array(b0)})
SuperMatdic_max.update({'GateTY':np.array(b1)})

SuperMatdic_min = SuperMatdic_max.copy()

SuperMatdic_max.update({'sQError':np.array(SuperOp(sQError_max))})
SuperMatdic_max.update({'tQError':np.array(SuperOp(tQError_max))})
SuperMatdic_max.update({'edError':np.array(SuperOp(edQError_max))})
SuperMatdic_max.update({'MagError1':np.array(SuperOp(MagError_max))})
SuperMatdic_max.update({'MagError2':np.array(SuperOp(MagError2_max))})

SuperMatdic_min.update({'sQError':np.array(SuperOp(sQError_min))})
SuperMatdic_min.update({'tQError':np.array(SuperOp(tQError_min))})
SuperMatdic_min.update({'edError':np.array(SuperOp(edQError_min))})
SuperMatdic_min.update({'MagError1':np.array(SuperOp(MagError_min))})
SuperMatdic_min.update({'MagError2':np.array(SuperOp(MagError2_min))})


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
    a0 = Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']
    a1 = SuperMatdic['GateZ']@Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']@SuperMatdic['GateZ']
    a2 = SuperMatdic['GateX']@Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']@Error_dic['TX']@SuperMatdic['sQError']@SuperMatdic['edError']@SuperMatdic['edError']@SuperMatdic['GateTX']
    a3 = SuperMatdic['GateY']@Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']@Error_dic['TY']@SuperMatdic['sQError']@SuperMatdic['edError']@SuperMatdic['edError']@SuperMatdic['GateTY']
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
    a0 = Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']
    a1 = SuperMatdic['GateZ']@Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']@SuperMatdic['GateZ']
    a2 = SuperMatdic['GateX']@Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']@Error_dic['TX']@SuperMatdic['sQError']@SuperMatdic['edError']@SuperMatdic['edError']@SuperMatdic['GateTX']
    a3 = SuperMatdic['GateY']@Error_dic['T']@SuperMatdic['MagError2']@SuperMatdic['MagError1']@SuperMatdic['GateT']@Error_dic['TY']@SuperMatdic['sQError']@SuperMatdic['edError']@SuperMatdic['edError']@SuperMatdic['GateTY']
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
qc.append(('Gateh',0))
qc.append(('Gateh',1))
qc.append(('sQError',0))
qc.append(('sQError',1))
qc.append(('spx',0))
qc.append(('spx',1))
for i in range(L):
    qc.append(('edError',0))
    qc.append(('edError',0))
    qc.append(('edError',1))
    qc.append(('edError',1))
    qc.append(('Gatecx',[0,1]))
    qc.append(('tQError',[0,1]))
    qc.append(('cx',[0,1]))
    qc.append(('edError',1))
    qc.append(('edError',1))
    qc.append(('TwirlingT',1))
    qc.append(('edError',0))
    qc.append(('edError',0))
    qc.append(('edError',1))
    qc.append(('edError',1))
    qc.append(('Gatecx',[0,1]))
    qc.append(('tQError',[0,1]))
    qc.append(('cx',[0,1]))
    qc.append(('edError',0))
    qc.append(('edError',0))     
    qc.append(('Gateh',0))
    qc.append(('sQError',0))
    qc.append(('H',0))
    qc.append(('edError',0))
    qc.append(('edError',0))
    qc.append(('TwirlingT',0))
    qc.append(('edError',0))
    qc.append(('edError',0))
    qc.append(('Gateh',0))
    qc.append(('sQError',0))
    qc.append(('H',0))
    qc.append(('edError',1))
    qc.append(('edError',1))
    qc.append(('Gateh',1))
    qc.append(('sQError',1))
    qc.append(('H',1))
    qc.append(('edError',1))
    qc.append(('edError',1))
    qc.append(('TwirlingT',1))
    qc.append(('edError',1))
    qc.append(('edError',1))
    qc.append(('Gateh',1))
    qc.append(('sQError',1))
    qc.append(('H',1))
qc.append(('edError',0))
qc.append(('edError',0))
qc.append(('edError',1))
qc.append(('edError',1))
qc.append(('mex',0))
qc.append(('mex',1))
qc.append(('sQError',0))
qc.append(('sQError',1))
qc.append(('Gateh',0))
qc.append(('Gateh',1))

def task():
    Ovalue = FunRes()
    return Ovalue

def AllTask():

    reslist0 = task()
    return np.real(reslist0)

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
    le = [1+i for i in range(1)]
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
        for j in range(100):
           
            # PAll = Error_model_learning(L,Mp)
            PAll = []
            for k in range(len(valueList_max)):
                p0 = list((np.array(valueList_max[k])+np.array(valueList_min[k]))/2)
                PAll.append(p0)
            end = time.time()
            PAllList = comm.gather(PAll,root=0)
            if rank ==0:
                valueList_est = []
                for i in range(len(PAllList[0])):
                    row = [sum(x)/len(PAllList) for x in zip(*[lst[i] for lst in PAllList])]
                    valueList_est.append(row)
                Error = {}
                for i in range(len(strList)):
                    Error.update({strList[i]:(keyList[i],valueList_est[i])})
                res = AllTask()
                print(res)
                res0.append(np.abs(res-c))
            else:
                None
        if rank ==0:
            meanlist.append((np.mean(res0),np.std(res0)))
        else:
            None
    if rank ==0:
        
        end = time.time()
        print(end-star)
        fname = './Plot/cPEC_bias'
        with open(fname, 'wb') as fp:
            pickle.dump(res, fp)


