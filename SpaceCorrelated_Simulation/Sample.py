from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator,SuperOp,DensityMatrix,Pauli,Chi
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
from scipy.linalg import block_diag
import itertools
import numpy as np


def cohe_noise(p,n):
    theta = np.sqrt(p/2)
    Z = Pauli(''.join(['Z' for i in range(n)])).to_matrix()
    U = np.cos(theta) * np.eye(2**n) - 1j * np.sin(theta) * Z
    return coherent_unitary_error(U)
def pauli_twirl_from_choi(choi: Chi, num_qubits: int):
    chi = choi.data
    diag_elements = np.diag(chi)
    d = 2 ** num_qubits

    # Pauli basis
    labels = [''.join(p) for p in itertools.product('IXYZ', repeat=num_qubits)]
    probs = {}
    for label, P in zip(labels, diag_elements):
        if np.abs(P) > 1e-10:
            probs[label] = float(np.real_if_close(P))

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




prob_dic = {}
# prob_dic_max = {}


qubits = 4

p0 = 0.01


cohe = cohe_noise(p0,qubits)
Depo_noise0 = depolarizing_error(p0/2,qubits)
Depo_noise = depolarizing_error(p0,qubits)
ende_noise = depolarizing_error(p0/2,qubits)
# Cz_noise = depolarizing_error(50*p0,2)
qc = QuantumCircuit(qubits)
qc.append(Depo_noise,range(qubits))
qc.append(ende_noise,range(qubits))
chi = Chi(qc)
print('w')
prob = pauli_twirl_from_choi(chi,qubits)
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
prob = list(new_prob.items())
prob_dic.update({'SP':prob})


qc = QuantumCircuit(qubits)
qc.append(ende_noise,range(qubits))
qc.append(Depo_noise,range(qubits))
chi = Chi(qc)
prob = pauli_twirl_from_choi(chi,qubits)
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
prob = list(new_prob.items())
prob_dic.update({'Mea':prob})


qc = QuantumCircuit(qubits)
qc.append(ende_noise,range(qubits))
qc.append(Depo_noise,range(qubits))
qc.append(ende_noise,range(qubits))
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,qubits).items())
prob_dic.update({'Stab':prob})
prob_dic.update({'cz1':prob})
prob_dic.update({'cz2':prob})


qc = QuantumCircuit(qubits)
qc.append(ende_noise,range(qubits))
qc.append(Depo_noise,range(qubits))
# for i in range(qubits//2):
#     qc.append(Cz_noise,[2*i,2*i+1])
qc.append(ende_noise,range(qubits))
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,qubits).items())
prob_dic.update({'cz1':prob})
qc = QuantumCircuit(qubits)
qc.append(ende_noise,range(qubits))
qc.append(Depo_noise,range(qubits))
# for i in range((qubits-1)//2):
#     qc.append(Cz_noise,[2*i+1,2*i+2])
qc.append(ende_noise,range(qubits))
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,qubits).items())
prob_dic.update({'cz2':prob})

qc = QuantumCircuit(qubits)
for i in range(qubits):
    qc.tdg(i)
qc.append(ende_noise,range(qubits))
for i in range(qubits):
    qc.t(i)
qc.append(Depo_noise0,range(qubits))
qc.append(cohe,range(qubits))
qc.append(ende_noise,range(qubits))
chi = Chi(qc)
prob = list(pauli_twirl_from_choi(chi,qubits).items())
prob_dic.update({'TLayer':prob})






import pickle


with open('./TMP/Sample_n='+str(qubits),'wb')as f:
    pickle.dump(prob_dic,f)


print(len(prob_dic))

# P = 1
# for key in prob_dic:
#     print(prob_dic[key][0])
#     P = P*(prob_dic[key][0])
# print(1-P)

# mat_max = SuperOp(qc)
# mat5 = np.array(0.5*mat_min+0.5*mat_max)