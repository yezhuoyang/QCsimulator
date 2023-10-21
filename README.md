# QCsimulator

The quantum circuit simulator that use GPU/parrallel computation to accerate the simulation speed.
I wish we could support the fastest qubit simulation to support the future quantum algorithm design and analysis!





# Type Checking Should be strictly followed! 

https://realpython.com/python-type-checking/


# Convention

1. The qubit index is counted from left to right.
   For example, when we are using an integer 0b1101 to denote a 4 qubit state, the first qubit is 1, the second qubit is 1, and the third qubit is zero




py -m mypy .\State\State.py

# A basic two qubit gate simulator:

In the first step, we need to implement a simple circuit simulator using Numpy.


- [x] Quantum Gate Class
- [x] Pauli X,Y,Z gate
- [x] Phase gate
- [x] pi/8 gate
- [x] CNOT
- [x] swap
- [x] Controlled-Z
- [x] controlled-phase
- [x] Toffoli
- [x] Fredkin
- [x] Rx(\theta),Ry(\theta),Rz(\theta)
- [x] Single qubit gate act on single qubit state

# Practice Cuda

- [ ] Use cuda to calculate the Single qubit gate operation


# Tensor Product and Measurement
After we have implemented 


- [x] Quantum State class
- [x] Tensor Product
- [x] Measurement
- [x] Two/Three qubit gates

# Multi-qubits Simulation

- [x] QuantumCircuit Class
- [x] Tensor Product of Gates/States
- [x] Measurement


# TestCode for circuit

- [ ] Write test code that compare all single gate with qiskit
- [ ] Writes test code that compare the running speed and storage requirement between cuircuit computation with qiskit.
- [ ] Using circuit Identity to test circuit calculation.

# Using Cudu and GPU to accelerate simulation



# Using Parallel Computation to accelerate simulation




# Using TensorNetwork Contraction to accelerate simulation




# Compile to qasm and visualization

- [ ] Compile to qasm
- [ ] Visualization
- [ ] Support ZX-calculus Visulization


# Algorithms implementation

- [ ] Deutsch's algorithm
- [ ] Grover's algorithm
- [ ] QFFT
- [ ] Shor's algorithm
- [ ] HHL algorithm


# Quantum Optimization Algorithm

- [ ] Design Parameter Circuit class
- [ ] Gradient Calculation
- [ ] BackPropagation
- [ ] QAOA algorithm
- [ ] VQE algorithm



# Quantum Cryptography

- [ ] QKD



# Testcode for Algorithm
- [ ] TestCode for Deutsch
- [ ] TestCode for Grover
- [ ] TestCode for QFFT
- [ ] TestCode for Shor
- [ ] TestCode for QAOA
- [ ] TestCode for VQE




# Frontend website
- [ ] Write a frontend website for demonstration
- [ ] Wirte documentation


# Density Matrix

- [ ] Density Matrix Class



# Quantum noise

- [ ] Bitflip Noise
- [ ] Decoherence Noise



# Error Correction Code


- [ ] Stabilizer Codes
- [ ] Surface Code
- [ ] LDPC Code






