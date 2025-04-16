from qiskit import QuantumCircuit
from qiskit.quantum_info import SuperOp,DensityMatrix,Operator
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
import numpy as np


def ProjectorSingle(flag1,flag2):
    pro = (Id + flag1*stab1)/2
    pro = pro@((Id + flag2*stab2)/2)
    return pro

def ProjectorTwo(flag1,flag2,flag3,flag4):
    pro = (Id + flag1*stab1)/2
    pro = pro@((Id + flag2*stab2)/2)
    pro = pro@((Id + flag3*stab3)/2)
    pro = pro@((Id + flag4*stab4)/2)
    return pro

def composePart(error0,flag):  
    if flag ==(1,1):
        error=error0+'I'
    elif flag ==(1,-1):
        error=error0+'X'
    elif flag ==(-1,1):
        error=error0+'Z'
    elif flag ==(-1,-1):
        error=error0+'Y'
    return error

import pickle

def DepoError(p,n):
    noise = depolarizing_error(p,n)
    return noise
## Error rate Fluctuation
# r= 1.5
# title = './TMP/SamRate_max_'
r= 0.5
title = './TMP/SamRate_min_'


p = 0.002
ped = 1/3*p
pst = p
theta = np.sqrt(r*p / 2)
n = np.array([0, 0, 1])  # 
n = n/np.linalg.norm(n)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
n_dot_sigma = n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z
U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * n_dot_sigma
MagError2 = coherent_unitary_error(U)
MagError1 = DepoError(1/2*p*r,1)
sQError = DepoError(pst*r,1)

tQError = DepoError(pst*r,2)

edQError = DepoError(ped*r,1)







Id = Operator.from_label('II')
stab1 = Operator.from_label('XX')
stab2 = Operator.from_label('ZZ')

Idle0={}
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.append(edQError,[0])
qc.append(edQError,[0])
choi = DensityMatrix.from_instruction(qc)
proj = ProjectorSingle(1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'I':prob})

proj = ProjectorSingle(1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'X':prob})

proj = ProjectorSingle(-1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Y':prob})

proj = ProjectorSingle(-1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Z':prob})

nor=sum(Idle0.values())
Idle = {key: value / nor for key, value in Idle0.items()}

with open(title + 'Idle','wb')as f1:
    pickle.dump(Idle,f1)

Idle0={}
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.h(0)
qc.append(edQError,[0])
qc.h(0)
qc.append(sQError,[0])
qc.append(edQError,[0])
choi = DensityMatrix.from_instruction(qc)
proj = ProjectorSingle(1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'I':prob})

proj = ProjectorSingle(1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'X':prob})

proj = ProjectorSingle(-1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Y':prob})

proj = ProjectorSingle(-1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Z':prob})

nor=sum(Idle0.values())
H = {key: value / nor for key, value in Idle0.items()}

with open(title + 'H','wb')as f1:
    pickle.dump(H,f1)

    
qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('X')@T
b= SuperOp(a)

c1 = b.to_instruction()

Idle0={}
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.append(c1,[0])
qc.append(edQError,[0])
qc.append(c1,[0])
qc.append(sQError,[0])
qc.append(edQError,[0])
choi = DensityMatrix.from_instruction(qc)
proj = ProjectorSingle(1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'I':prob})

proj = ProjectorSingle(1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'X':prob})

proj = ProjectorSingle(-1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Y':prob})

proj = ProjectorSingle(-1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Z':prob})

nor=sum(Idle0.values())
TX = {key: value / nor for key, value in Idle0.items()}


with open(title + 'TX','wb')as f1:
    pickle.dump(TX,f1)

qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('Y')@T
b= SuperOp(a)
c1 = b.to_instruction()


Idle0={}
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.append(c1,[0])
qc.append(edQError,[0])
qc.append(c1,[0])
qc.append(sQError,[0])
qc.append(edQError,[0])
choi = DensityMatrix.from_instruction(qc)
proj = ProjectorSingle(1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'I':prob})

proj = ProjectorSingle(1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'X':prob})

proj = ProjectorSingle(-1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Y':prob})

proj = ProjectorSingle(-1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Z':prob})

nor=sum(Idle0.values())
Idle = {key: value / nor for key, value in Idle0.items()}


with open(title + 'TY','wb')as f1:
    pickle.dump(Idle,f1)


Id = Operator.from_label('II')
stab1 = Operator.from_label('XX')
stab2 = Operator.from_label('ZZ')

Idle0={}
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.tdg(0)
qc.append(edQError,[0])
qc.t(0)
qc.append(MagError1,[0])
qc.append(MagError2,[0])
qc.append(edQError,[0])
choi = DensityMatrix.from_instruction(qc)
proj = ProjectorSingle(1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'I':prob})

proj = ProjectorSingle(1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'X':prob})

proj = ProjectorSingle(-1,-1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Y':prob})

proj = ProjectorSingle(-1,1)
prob = np.real(np.trace(proj@choi@proj))
Idle0.update({'Z':prob})

nor=sum(Idle0.values())
Idle = {key: value / nor for key, value in Idle0.items()}


with open(title + 'T','wb')as f1:
    pickle.dump(Idle,f1)



sP0={}
qc = QuantumCircuit(1)
qc.append(sQError,[0])
qc.append(edQError,[0])
choi = DensityMatrix.from_instruction(qc)
proj = (Operator.from_label('Z')+Operator.from_label('I'))/2
prob = np.real(np.trace(proj@choi@proj))
sP0.update({'I':prob})
sP0.update({'Z':1-prob})



nor=sum(sP0.values())
sP = {key: value / nor for key, value in sP0.items()}


with open(title + 'sPx','wb')as f1:
    pickle.dump(sP,f1)




Me0={}
qc = QuantumCircuit(1)
qc.append(edQError,[0])
qc.append(sQError,[0])
choi = DensityMatrix.from_instruction(qc)
proj = (Operator.from_label('Z')+Operator.from_label('I'))/2
prob = np.real(np.trace(proj@choi@proj))
Me0.update({'I':prob})
Me0.update({'Z':1-prob})

nor=sum(Me0.values())
Me = {key: value / nor for key, value in Me0.items()}


with open(title + 'Mex','wb')as f1:
    pickle.dump(Me,f1)



Id = Operator.from_label('IIII')
stab1 = Operator.from_label('XXXX')
stab2 = Operator.from_label('IZIZ')
stab3 = Operator.from_label('XIXI')
stab4 = Operator.from_label('ZZZZ')
Cx0 = {}
qc = QuantumCircuit(4)
qc.h(0)
qc.h(1)
qc.cx(0,2)
qc.cx(1,3)
qc.cx(0,1)
qc.append(edQError,[0])
qc.append(edQError,[1])
qc.cx(0,1)
qc.append(tQError,[0,1])
qc.append(edQError,[0])
qc.append(edQError,[1])
choi = DensityMatrix.from_instruction(qc)

import itertools

for a, b, c, d in itertools.product([+1, -1], repeat=4):
    error = ''
    flag = (a,b)
    error = composePart(error,flag)
    flag = (c,d)
    error = composePart(error,flag)
    proj = ProjectorTwo(a,b,c,d)
    prob = np.real(np.trace(proj@choi@proj))
    Cx0.update({error:prob})

nor=sum(Cx0.values())
Cx = {key: value / nor for key, value in Cx0.items()}
# print(Cx)

with open(title + 'Cx','wb')as f1:
    pickle.dump(Cx,f1)


