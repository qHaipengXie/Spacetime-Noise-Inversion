from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix,Operator

Olist0 = 'XI'
Olist = Operator.from_label(Olist0)
L = 8
valuelist = []
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
for i in range(L):
    qc.cx(0,1)
    qc.t(1)
    qc.cx(0,1)
    qc.h(0)
    qc.t(0)
    qc.h(0)
    qc.h(1)
    qc.t(1)
    qc.h(1)
    rho = DensityMatrix.from_instruction(qc)
    Ovalue = rho.expectation_value(Olist)
    valuelist.append(Ovalue)

print(valuelist)
import pickle
with open('./Plot/ErrorFree','wb') as f1:
    pickle.dump(valuelist,f1)

