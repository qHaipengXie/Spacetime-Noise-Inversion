from qiskit.quantum_info import SuperOp,Operator
from qiskit_aer.noise import depolarizing_error,coherent_unitary_error
import numpy as np
import pickle

def DepoError(p,n):
    noise = depolarizing_error(p,n)
    return noise
L = 8

##----Construct Clifford gates for Twiling the T Gate
T = Operator.from_label('T')
a0=T.adjoint()@Operator.from_label('X')@T
Clifford_Gate_TX= SuperOp(a0)
a1=T.adjoint()@Operator.from_label('Y')@T
Clifford_Gate_TY= SuperOp(a1)

##----Assume the rotation direction of a coherent error.(Z)
n = np.array([0, 0, 1])  # 
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
n_dot_sigma = n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z

##----Error configration
p = 0.002
ped = 1/3*p*0  #----Without implementing the SNI/cPEC protocol, error boosting on the computational circuit is unnecessary, i.e., there is no need to introduce encoding and decoding errors.
pst = p

r_min= 0.5
theta = np.sqrt(r_min*p / 2)
U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * n_dot_sigma
T_Error_Coherent_min = coherent_unitary_error(U)
T_Error_Pauli_min = DepoError(1/2*p*r_min,1)
Single_qubits_Error_min = DepoError(pst*r_min,1)
Two_qubits_Error_min = DepoError(pst*r_min,2)
en_decoding_Error_min = DepoError(ped*r_min,1)

r_max= 1.5
theta = np.sqrt(r_max*p / 2)
U = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * n_dot_sigma
T_Error_Coherent_max = coherent_unitary_error(U)
T_Error_Pauli_max = DepoError(1/2*p*r_max,1)
Single_qubits_Error_max = DepoError(pst*r_max,1)
Two_qubits_Error_max = DepoError(pst*r_max,2)
en_decoding_Error_max = DepoError(ped*r_max,1)

##----Store the superoperator matrix of the operation and the error.
SuperMatdic_max = {}
SuperMatdic_max.update({'Gate_H':np.array(SuperOp(Operator.from_label('H')))})
SuperMatdic_max.update({'Gate_X':np.array(SuperOp(Operator.from_label('X')))})
SuperMatdic_max.update({'Gate_Y':np.array(SuperOp(Operator.from_label('Y')))})
SuperMatdic_max.update({'Gate_Z':np.array(SuperOp(Operator.from_label('Z')))})
SuperMatdic_max.update({'Gate_T':np.array(SuperOp(Operator.from_label('T')))})
SuperMatdic_max.update({'Gate_cx':np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]))})
SuperMatdic_max.update({'Gate_TX':np.array(Clifford_Gate_TX)})
SuperMatdic_max.update({'Gate_TY':np.array(Clifford_Gate_TY)})

SuperMatdic_min = SuperMatdic_max.copy()

SuperMatdic_max.update({'Single_qubits_Error':np.array(SuperOp(Single_qubits_Error_max))})
SuperMatdic_max.update({'Two_qubits_Error':np.array(SuperOp(Two_qubits_Error_max))})
SuperMatdic_max.update({'En_Decoding_Error':np.array(SuperOp(en_decoding_Error_max))})
SuperMatdic_max.update({'T_Error_Pauli':np.array(SuperOp(T_Error_Pauli_max))})
SuperMatdic_max.update({'T_Error_Coherent':np.array(SuperOp(T_Error_Coherent_max))})
a0 = SuperMatdic_max['T_Error_Coherent']@SuperMatdic_max['T_Error_Pauli']@SuperMatdic_max['Gate_T']
a1 = SuperMatdic_max['Gate_Z']@SuperMatdic_max['T_Error_Coherent']@SuperMatdic_max['T_Error_Pauli']@SuperMatdic_max['Gate_T']@SuperMatdic_max['Gate_Z']
a2 = SuperMatdic_max['Gate_X']@SuperMatdic_max['T_Error_Coherent']@SuperMatdic_max['T_Error_Pauli']@SuperMatdic_max['Gate_T']@SuperMatdic_max['Single_qubits_Error']@SuperMatdic_max['En_Decoding_Error']@SuperMatdic_max['En_Decoding_Error']@SuperMatdic_max['Gate_TX']
a3 = SuperMatdic_max['Gate_Y']@SuperMatdic_max['T_Error_Coherent']@SuperMatdic_max['T_Error_Pauli']@SuperMatdic_max['Gate_T']@SuperMatdic_max['Single_qubits_Error']@SuperMatdic_max['En_Decoding_Error']@SuperMatdic_max['En_Decoding_Error']@SuperMatdic_max['Gate_TY']
SuperMatdic_max.update({'TwirlingT':0.25*sum([a0,a1,a2,a3])})#--There are four circuits with equal probabilities for the twirled T gate.

SuperMatdic_min.update({'Single_qubits_Error':np.array(SuperOp(Single_qubits_Error_min))})
SuperMatdic_min.update({'Two_qubits_Error':np.array(SuperOp(Two_qubits_Error_min))})
SuperMatdic_min.update({'En_Decoding_Error':np.array(SuperOp(en_decoding_Error_min))})
SuperMatdic_min.update({'T_Error_Pauli':np.array(SuperOp(T_Error_Pauli_min))})
SuperMatdic_min.update({'T_Error_Coherent':np.array(SuperOp(T_Error_Coherent_min))})
a0 = SuperMatdic_min['T_Error_Coherent']@SuperMatdic_min['T_Error_Pauli']@SuperMatdic_min['Gate_T']
a1 = SuperMatdic_min['Gate_Z']@SuperMatdic_min['T_Error_Coherent']@SuperMatdic_min['T_Error_Pauli']@SuperMatdic_min['Gate_T']@SuperMatdic_min['Gate_Z']
a2 = SuperMatdic_min['Gate_X']@SuperMatdic_min['T_Error_Coherent']@SuperMatdic_min['T_Error_Pauli']@SuperMatdic_min['Gate_T']@SuperMatdic_min['Single_qubits_Error']@SuperMatdic_min['En_Decoding_Error']@SuperMatdic_min['En_Decoding_Error']@SuperMatdic_min['Gate_TX']
a3 = SuperMatdic_min['Gate_Y']@SuperMatdic_min['T_Error_Coherent']@SuperMatdic_min['T_Error_Pauli']@SuperMatdic_min['Gate_T']@SuperMatdic_min['Single_qubits_Error']@SuperMatdic_min['En_Decoding_Error']@SuperMatdic_min['En_Decoding_Error']@SuperMatdic_min['Gate_TY']
SuperMatdic_min.update({'TwirlingT':0.25*sum([a0,a1,a2,a3])})

##----When using the Kronecker product of superoperator matrices(1 qubit -> 2 qubit), a basis swap is required.
commute_mat = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
commute_mat0 = np.kron(np.eye(2),commute_mat)
commute_mat1 = np.kron(commute_mat,np.eye(2))

##----Observable (we insert a noiseless Hadamard gate in the circuit to transform X into Z)
O0 = Operator.from_label('ZI').to_matrix()
O = O0.flatten()


def FunRes():
    ##----Calculate the result of the maximum error circuit.
    mat = np.zeros((16,1))
    mat[0] = 1
    SuperMatdic = SuperMatdic_max
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
    ##----Calculate the result of the minimum error circuit
    mat = np.zeros((16,1))
    mat[0] = 1
    SuperMatdic = SuperMatdic_min
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


##----Building a circuit that includes errors.
def qc_make(L):
    qc = []
    qc.append(('Gate_H',0))
    qc.append(('Gate_H',1))
    qc.append(('Single_qubits_Error',0))
    qc.append(('Single_qubits_Error',1))
    for i in range(L):
        qc.append(('En_Decoding_Error',0))
        qc.append(('En_Decoding_Error',0))
        qc.append(('En_Decoding_Error',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('Gate_cx',[0,1]))
        qc.append(('Two_qubits_Error',[0,1]))
        qc.append(('En_Decoding_Error',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('TwirlingT',1))
        qc.append(('En_Decoding_Error',0))
        qc.append(('En_Decoding_Error',0))
        qc.append(('En_Decoding_Error',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('Gate_cx',[0,1]))
        qc.append(('Two_qubits_Error',[0,1]))
        qc.append(('En_Decoding_Error',0))
        qc.append(('En_Decoding_Error',0))     
        qc.append(('Gate_H',0))
        qc.append(('Single_qubits_Error',0))
        qc.append(('En_Decoding_Error',0))
        qc.append(('En_Decoding_Error',0))
        qc.append(('TwirlingT',0))
        qc.append(('En_Decoding_Error',0))
        qc.append(('En_Decoding_Error',0))
        qc.append(('Gate_H',0))
        qc.append(('Single_qubits_Error',0))
        qc.append(('En_Decoding_Error',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('Gate_H',1))
        qc.append(('Single_qubits_Error',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('TwirlingT',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('En_Decoding_Error',1))
        qc.append(('Gate_H',1))
        qc.append(('Single_qubits_Error',1))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',0))
    qc.append(('En_Decoding_Error',1))
    qc.append(('En_Decoding_Error',1))
    qc.append(('Single_qubits_Error',0))
    qc.append(('Single_qubits_Error',1))
    qc.append(('Gate_H',0))
    qc.append(('Gate_H',1))
    return qc



meanlist = []
for j in range(L):
    qc = qc_make(j+1)
    res = np.real(FunRes())
    meanlist.append(res[0])

print(meanlist)
fname = './Plot/No_mitigate'
with open(fname, 'wb') as fp:
    pickle.dump(meanlist, fp)


