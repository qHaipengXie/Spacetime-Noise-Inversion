This repository contains the simulation source code for the paper "Unbiased Quantum Error Mitigation Without Reliance on an Accurate Error Model". 
It utilizes the Qiskit (version 1.3.1), Qiskit-aer (version:0.15.1) for computations and the MPI4PY library for parallel execution.


To simulate temporal correlations, you first need to run practical_ErrorSampler to generate the probability distribution of Pauli errors for each operation. This distribution is then used for sampling in the SNI.py and cPEC.py.


If you encounter any issues or have questions about the code, feel free to reach out via email: xiehaipeng22@gscaep.ac.cn
