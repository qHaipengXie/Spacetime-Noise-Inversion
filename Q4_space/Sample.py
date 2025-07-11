from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator,SuperOp,DensityMatrix,Pauli,Chi
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
from scipy.linalg import block_diag
import itertools
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
def pauli_twirl_from_choi(choi: Chi, num_qubits: int):
    # 获取 Choi 矩阵
    chi = choi.data
    diag_elements = np.diag(chi)
    d = 2 ** num_qubits

    # Pauli basis
    labels = [''.join(p) for p in itertools.product('IXYZ', repeat=num_qubits)]
    # pauli_ops = [Pauli(l).to_matrix() for l in labels]

    probs = {}
    for label, P in zip(labels, diag_elements):
        if np.abs(P) > 1e-10:
            probs[label] = float(np.real_if_close(P))

    # # Choi basis元素是 P ⊗ P*
    # for label, P in zip(labels, pauli_ops):
    #     basis_op = np.kron(P, P.conj())
    #     coeff = np.trace(basis_op.conj().T @ chi) / (d ** 2)
    #     # if np.abs(coeff) > 1e-10:
    #     probs[label] = float(np.real_if_close(coeff))

    # 归一化
    # print(probs)
    # print(labels)
    total = sum(probs.values())
    for k in probs:
        probs[k] /= total

    return probs
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
idle_SPAM_noise = cohe_noise(p0/5,1)
Depo_noise = depolarizing_error(p0/2,4)
ende_noise = depolarizing_error(p0/5,1)

p1 = 0.03

single_noise1 = depolarizing_error(p1,1)
two_noise1 = depolarizing_error(p1,2)
idle_SPAM_noise1 = cohe_noise(p1/5,1)
Depo_noise1 = depolarizing_error(p1/2,4)
ende_noise1 = depolarizing_error(p1/5,1)

prob_dic_min = {}
prob_dic_max = {}

# # state layer
# qc = QuantumCircuit(4)
# qc.h(0)
# qc.h(1)
# qc.h(2)
# qc.h(3)
# state = np.array(DensityMatrix.from_instruction(qc))
# sta = state.flatten()

# SP noise layer
qc = QuantumCircuit(4)
qc.append(Depo_noise,[0,1,2,3])
qc.append(idle_SPAM_noise,[0])
qc.append(idle_SPAM_noise,[1])
qc.append(idle_SPAM_noise,[2])
qc.append(idle_SPAM_noise,[3])
qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
chi = Chi(qc)
prob = pauli_twirl_from_choi(chi,4)
# print(prob)
new_prob = {}
for key in prob:
    new_key = ''
    for i in range(len(key)):
        if key[i] == 'I' or key[i] =='X':
            new_key = new_key+'I'
        else:
            new_key = new_key+'Z'
    if new_key in new_prob:
        v0 = new_prob[new_key]
        v1 = prob[key]
        v2 = v0 + v1
        new_prob.update({new_key:v2})
    else:
        new_prob.update({new_key:prob[key]})
prob = list(new_prob.values())
prob_dic_min.update({'SP':prob})


qc = QuantumCircuit(4)
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[0])
qc.append(idle_SPAM_noise1,[1])
qc.append(idle_SPAM_noise1,[2])
qc.append(idle_SPAM_noise1,[3])
qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])
chi = Chi(qc)
prob = pauli_twirl_from_choi(chi,4)
# print(prob)
new_prob = {}
for key in prob:
    new_key = ''
    for i in range(len(key)):
        if key[i] == 'I' or key[i] =='X':
            new_key = new_key+'I'
        else:
            new_key = new_key+'Z'
    if new_key in new_prob:
        v0 = new_prob[new_key]
        v1 = prob[key]
        v2 = v0 + v1
        new_prob.update({new_key:v2})
    else:
        new_prob.update({new_key:prob[key]})
prob = list(new_prob.values())
prob_dic_max.update({'SP':prob})

# PT10 noise layer
qc = QuantumCircuit(4)
# qc.append(clix,[2])
# qc.x(0)
qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
qc.append(Depo_noise,[0,1,2,3])
qc.append(idle_SPAM_noise,[0])
qc.append(idle_SPAM_noise,[1])
qc.append(single_noise,[2])
qc.append(idle_SPAM_noise,[3])
qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_min.update({'PT1X':prob})
prob_dic_min.update({'PT1Y':prob})

qc = QuantumCircuit(4)
qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])

qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[0])
qc.append(idle_SPAM_noise1,[1])
qc.append(single_noise1,[2])
qc.append(idle_SPAM_noise1,[3])

qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_max.update({'PT1X':prob})
prob_dic_max.update({'PT1Y':prob})
# mat_max = SuperOp(qc)

# mat10= np.array(0.5*mat_min+0.5*mat_max)

# # PT11 noise layer
# qc = QuantumCircuit(4)
# qc.append(cliy,[2])
# qc.append(Depo_noise,[0,1,2,3])
# qc.append(idle_SPAM_noise,[0])
# qc.append(idle_SPAM_noise,[1])
# qc.append(single_noise,[2])
# qc.append(idle_SPAM_noise,[3])
# mat_min = SuperOp(qc)

# qc = QuantumCircuit(4)
# qc.append(cliy,[2])
# qc.append(Depo_noise1,[0,1,2,3])
# qc.append(idle_SPAM_noise1,[0])
# qc.append(idle_SPAM_noise1,[1])
# qc.append(single_noise1,[2])
# qc.append(idle_SPAM_noise1,[3])
# mat_max = SuperOp(qc)

# mat11= np.array(0.5*mat_min+0.5*mat_max)

# # first gate and noise layer
qc = QuantumCircuit(4)
qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
qc.append(Depo_noise,[0,1,2,3])
qc.append(two_noise,[0,1])
qc.append(single_noise,[2])
qc.append(single_noise,[3])
qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_min.update({'layer1':prob})

# mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])

qc.append(Depo_noise1,[0,1,2,3])
qc.append(two_noise1,[0,1])
qc.append(single_noise1,[2])
qc.append(single_noise1,[3])

qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_max.update({'layer1':prob})

# mat_max = SuperOp(qc)

# mat2= np.array(0.5*mat_min+0.5*mat_max)


# PT20 noise layer
qc = QuantumCircuit(4)
qc.append(Depo_noise,[0,1,2,3])
qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])

qc.append(idle_SPAM_noise,[2])
qc.append(idle_SPAM_noise,[1])
qc.append(single_noise,[0])
qc.append(idle_SPAM_noise,[3])

qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_min.update({'PT2X':prob})
prob_dic_min.update({'PT2Y':prob})

# mat_min = SuperOp(qc)

qc = QuantumCircuit(4)
qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])

qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[2])
qc.append(idle_SPAM_noise1,[1])
qc.append(single_noise1,[0])
qc.append(idle_SPAM_noise1,[3])

qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_max.update({'PT2X':prob})
prob_dic_max.update({'PT2Y':prob})

# mat_max = SuperOp(qc)

# mat30= np.array(0.5*mat_min+0.5*mat_max)

# # PT21 noise layer
# qc = QuantumCircuit(4)
# qc.append(cliy,[0])
# qc.append(Depo_noise,[0,1,2,3])
# qc.append(idle_SPAM_noise,[2])
# qc.append(idle_SPAM_noise,[1])
# qc.append(single_noise,[0])
# qc.append(idle_SPAM_noise,[3])
# mat_min = SuperOp(qc)

# qc = QuantumCircuit(4)
# qc.append(cliy,[0])
# qc.append(Depo_noise1,[0,1,2,3])
# qc.append(idle_SPAM_noise1,[2])
# qc.append(idle_SPAM_noise1,[1])
# qc.append(single_noise1,[0])
# qc.append(idle_SPAM_noise1,[3])
# mat_max = SuperOp(qc)

# mat31= np.array(0.5*mat_min+0.5*mat_max)

# second gate and noise layer
qc = QuantumCircuit(4)
qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])

qc.append(Depo_noise,[0,1,2,3])
qc.append(single_noise,[0])
qc.append(two_noise,[1,2])
qc.append(single_noise,[3])

qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_min.update({'layer2':prob})


qc = QuantumCircuit(4)
qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])

qc.append(Depo_noise1,[0,1,2,3])
qc.append(single_noise1,[0])
qc.append(two_noise1,[1,2])
qc.append(single_noise1,[3])

qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,4).values())
prob_dic_max.update({'layer2':prob})
# mat_max = SuperOp(qc)

# mat4= np.array(0.5*mat_min+0.5*mat_max)


# measure error
qc = QuantumCircuit(4)

qc.append(ende_noise,[0])
qc.append(ende_noise,[1])
qc.append(ende_noise,[2])
qc.append(ende_noise,[3])
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise,[0])
qc.append(idle_SPAM_noise,[1])
qc.append(idle_SPAM_noise,[2])
qc.append(idle_SPAM_noise,[3])
chi = Chi(qc)
prob = pauli_twirl_from_choi(chi,4)
# print(prob)
new_prob = {}
for key in prob:
    new_key = ''
    for i in range(len(key)):
        if key[i] == 'I' or key[i] =='X':
            new_key = new_key+'I'
        else:
            new_key = new_key+'Z'
    if new_key in new_prob:
        v0 = new_prob[new_key]
        v1 = prob[key]
        v2 = v0 + v1
        new_prob.update({new_key:v2})
    else:
        new_prob.update({new_key:prob[key]})
prob = list(new_prob.values())
prob_dic_min.update({'Mea':prob})


qc = QuantumCircuit(4)
qc.append(ende_noise1,[0])
qc.append(ende_noise1,[1])
qc.append(ende_noise1,[2])
qc.append(ende_noise1,[3])
qc.append(Depo_noise1,[0,1,2,3])
qc.append(idle_SPAM_noise1,[0])
qc.append(idle_SPAM_noise1,[1])
qc.append(idle_SPAM_noise1,[2])
qc.append(idle_SPAM_noise1,[3])
hi = Chi(qc)
prob = pauli_twirl_from_choi(chi,4)
# print(prob)
new_prob = {}
for key in prob:
    new_key = ''
    for i in range(len(key)):
        if key[i] == 'I' or key[i] =='X':
            new_key = new_key+'I'
        else:
            new_key = new_key+'Z'
    if new_key in new_prob:
        v0 = new_prob[new_key]
        v1 = prob[key]
        v2 = v0 + v1
        new_prob.update({new_key:v2})
    else:
        new_prob.update({new_key:prob[key]})
prob = list(new_prob.values())
prob_dic_max.update({'Mea':prob})
import pickle
with open('./TMP/Sample_max','wb')as f:
    pickle.dump(prob_dic_max,f)

with open('./TMP/Sample_min','wb')as f:
    pickle.dump(prob_dic_min,f)

print(len(prob_dic_max))
print(len(prob_dic_min))

P = 1
for key in prob_dic_max:
    print(prob_dic_max[key][0])
    P = P*(prob_dic_max[key][0])
print(1-P)

# mat_max = SuperOp(qc)
# mat5 = np.array(0.5*mat_min+0.5*mat_max)