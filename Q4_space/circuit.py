from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,SuperOp,DensityMatrix
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
import numpy as np
def depo_noise(p):
    return depolarizing_error(p,4)

def cohe_noise(p,n):
    theta = np.sqrt(p/2)
    if n == 1 :
        sigma_z = np.array([[1, 0], [0, -1]])
        U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * sigma_z
    else:
        sigma_z = np.array([[1, 0], [0, -1]])
        ZZ = np.kron(sigma_z,sigma_z)
        U = np.cos(theta) * np.eye(4) - 1j * np.sin(theta) * ZZ
    return coherent_unitary_error(U)

qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('X')@T
b= SuperOp(a)
clix = b.to_instruction()

qc = QuantumCircuit(1)
qc.t(0)
T = Operator(qc)
a=T.adjoint()@Operator.from_label('Y')@T
b= SuperOp(a)
cliy = b.to_instruction()

p0 = 0.01

single_noise = depolarizing_error(p0,1)
two_noise = depolarizing_error(p0,2)
idle_SPAM_noise = cohe_noise(0.5*p0,1)
Depo_noise = depolarizing_error(p0/2,4)

p1 = 0.03

single_noise1 = depolarizing_error(p1,1)
two_noise1 = depolarizing_error(p1,2)
idle_SPAM_noise1 = cohe_noise(0.5*p1,1)
Depo_noise1 = depolarizing_error(p1/2,4)

# state layer
qc = QuantumCircuit(4)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
state = np.array(DensityMatrix.from_instruction(qc))
sta = state.flatten()

# SP noise layer
qc = QuantumCircuit(4)
qc.append(Depo_noise,[0,1,2,3])
qc.append(idle_SPAM_noise,[0])
qc.append(idle_SPAM_noise,[1])
qc.append(idle_SPAM_noise,[2])
qc.append(idle_SPAM_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[0])
qc.append(idle_SPAM_noise1,[1])
qc.append(idle_SPAM_noise1,[2])
qc.append(idle_SPAM_noise1,[3])
mat_max = SuperOp(qc)
mat0 = np.array(0.5*mat_min+0.5*mat_max)

# PT10 noise layer
qc = QuantumCircuit(4)
qc.append(clix,[2])
qc.append(Depo_noise,[0,1,2,3])
qc.append(idle_SPAM_noise,[0])
qc.append(idle_SPAM_noise,[1])
qc.append(single_noise,[2])
qc.append(idle_SPAM_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(clix,[2])
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[0])
qc.append(idle_SPAM_noise1,[1])
qc.append(single_noise1,[2])
qc.append(idle_SPAM_noise1,[3])
mat_max = SuperOp(qc)

mat10= np.array(0.5*mat_min+0.5*mat_max)

# PT11 noise layer
qc = QuantumCircuit(4)
qc.append(cliy,[2])
qc.append(Depo_noise,[0,1,2,3])
qc.append(idle_SPAM_noise,[0])
qc.append(idle_SPAM_noise,[1])
qc.append(single_noise,[2])
qc.append(idle_SPAM_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(cliy,[2])
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[0])
qc.append(idle_SPAM_noise1,[1])
qc.append(single_noise1,[2])
qc.append(idle_SPAM_noise1,[3])
mat_max = SuperOp(qc)

mat11= np.array(0.5*mat_min+0.5*mat_max)

# first gate and noise layer
qc = QuantumCircuit(4)
qc.cx(0,1)
qc.t(2)
qc.h(3)
qc.append(Depo_noise,[0,1,2,3])
qc.append(two_noise,[0,1])
qc.append(single_noise,[2])
qc.append(single_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.cx(0,1)
qc.t(2)
qc.h(3)
qc.append(Depo_noise1,[0,1,2,3])
qc.append(two_noise1,[0,1])
qc.append(single_noise1,[2])
qc.append(single_noise1,[3])
mat_max = SuperOp(qc)

mat2= np.array(0.5*mat_min+0.5*mat_max)


# PT20 noise layer
qc = QuantumCircuit(4)
qc.append(clix,[0])
qc.append(Depo_noise,[0,1,2,3])
qc.append(idle_SPAM_noise,[2])
qc.append(idle_SPAM_noise,[1])
qc.append(single_noise,[0])
qc.append(idle_SPAM_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(clix,[0])
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[2])
qc.append(idle_SPAM_noise1,[1])
qc.append(single_noise1,[0])
qc.append(idle_SPAM_noise1,[3])
mat_max = SuperOp(qc)

mat30= np.array(0.5*mat_min+0.5*mat_max)

# PT21 noise layer
qc = QuantumCircuit(4)
qc.append(cliy,[0])
qc.append(Depo_noise,[0,1,2,3])
qc.append(idle_SPAM_noise,[2])
qc.append(idle_SPAM_noise,[1])
qc.append(single_noise,[0])
qc.append(idle_SPAM_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(cliy,[0])
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[2])
qc.append(idle_SPAM_noise1,[1])
qc.append(single_noise1,[0])
qc.append(idle_SPAM_noise1,[3])
mat_max = SuperOp(qc)

mat31= np.array(0.5*mat_min+0.5*mat_max)

# second gate and noise layer
qc = QuantumCircuit(4)
qc.t(0)
qc.cx(1,2)
qc.s(3)
qc.append(Depo_noise,[0,1,2,3])
qc.append(single_noise,[0])
qc.append(two_noise,[1,2])
qc.append(single_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.t(0)
qc.cx(1,2)
qc.s(3)
qc.append(Depo_noise1,[0,1,2,3])
qc.append(single_noise1,[0])
qc.append(two_noise1,[1,2])
qc.append(single_noise1,[3])
mat_max = SuperOp(qc)

mat4= np.array(0.5*mat_min+0.5*mat_max)


# measure error
qc = QuantumCircuit(4)
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise,[0])
qc.append(idle_SPAM_noise,[1])
qc.append(idle_SPAM_noise,[2])
qc.append(idle_SPAM_noise,[3])
mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[0])
qc.append(idle_SPAM_noise1,[1])
qc.append(idle_SPAM_noise1,[2])
qc.append(idle_SPAM_noise1,[3])
mat_max = SuperOp(qc)
mat5 = np.array(0.5*mat_min+0.5*mat_max)


qc = QuantumCircuit(4)
qc.x(2)
X2 = np.array(SuperOp(qc))
qc = QuantumCircuit(4)
qc.y(2)
Y2 = np.array(SuperOp(qc))
qc = QuantumCircuit(4)
qc.z(2)
Z2 = np.array(SuperOp(qc))

qc = QuantumCircuit(4)
qc.x(0)
X0 = np.array(SuperOp(qc))
qc = QuantumCircuit(4)
qc.y(0)
Y0 = np.array(SuperOp(qc))
qc = QuantumCircuit(4)
qc.z(0)
Z0 = np.array(SuperOp(qc))

mea = np.array(Operator.from_label('IIXI'))
mea = mea.flatten()

raw = mea@mat5@(0.25*X0@mat4@mat30+0.25*Y0@mat4@mat31+0.25*Z0@mat4@Z0+0.25*mat4)@(0.25*X2@mat2@mat10+0.25*Y2@mat2@mat11+0.25*Z2@mat2@Z2+0.25*mat2)@mat0@sta
print(raw)
free = mea@mat5@mat4@mat2@mat0@sta
print(free)
qc = QuantumCircuit(4)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)

qc.cx(0,1)
qc.t(2)
qc.h(3)

qc.t(0)
qc.cx(1,2)
qc.s(3)


print(DensityMatrix.from_instruction(qc).expectation_value(Operator.from_label('IZZI')))
# print(DensityMatrix.from_instruction(qc))
