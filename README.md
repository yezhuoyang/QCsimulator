# QCsimulator

The quantum circuit simulator written in python that aims to use GPU/parrallel computation to accerate the simulation speed.
I wish we could support the fastest qubit simulation to support the future quantum algorithm design and analysis!




# Rules of developing and Conventions


1.Every function should obey the type checking rules [Blog of Type Checking for Python](https://realpython.com/python-type-checking/)



2. The qubit index is counted from left to right.
   For example, when we are using an integer 0b1101 to denote a 4 qubit state, the first qubit is 1, the second qubit is 1, and the third qubit is zero


# Plans for development

```bash
py -m mypy .\State\State.py
```


## Gate, State and Circuit


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
- [x] Quantum State class

## Tensor Product and Measurement
After we have implemented 



- [x] Tensor Product
- [x] Measurement
- [x] Two/Three qubit gates

Multi-qubits Simulation

- [x] QuantumCircuit Class
- [x] Tensor Product of Gates/States
- [x] Measurement


## TestCode

- [ ] Write test code that compare all single gate with qiskit
- [ ] Writes test code that compare the running speed and storage requirement between cuircuit computation with qiskit.
- [ ] Using circuit Identity to test circuit calculation.

## Algorithm

- [ ] Deutsch's algorithm
- [ ] Grover's algorithm
- [ ] QFFT
- [ ] Shor's algorithm
- [ ] HHL algorithm
- [ ] Design Parameter Circuit class
- [ ] Gradient Calculation
- [ ] BackPropagation
- [ ] QAOA algorithm
- [ ] VQE algorithm

## Compatibility

- [ ] Compile to qasm 2.0
- [x] Load qasm 2.0 assembly and simulate
- [ ] Compile to qasm 3.0
- [ ] Load qasm 3.0 assembly and simulate
- [ ] Visualization
- [ ] Support ZX-calculus Visulization

## Density Matrix and Quantum Noise


- [ ] Density Matrix Class
- [ ] Quantum Noise Class
- [ ] Bitflip Noise
- [ ] Decoherence Noise



## Error Correction Code


- [ ] Stabilizer Codes
- [ ] Surface Code
- [ ] LDPC Code






