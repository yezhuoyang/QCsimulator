# QCsimulator

The quantum circuit simulator written in python that aims to use GPU/parrallel computation to accerate the simulation speed.
I wish we could support the fastest qubit simulation to support the future quantum algorithm/architecture design and analysis!



# Rules of developing and Conventions


1.Every function should obey the type checking rules [Blog of Type Checking for Python](https://realpython.com/python-type-checking/)


```bash
py -m mypy filetobecheck.py
```


2. The qubit index is counted from left to right.
   For example, when we are using an integer 0b1101 to denote a 4 qubit state, the first qubit is 1, the second qubit is 1, and the third qubit is zero.
   It should be pointout that

# Plans and Goals for developing


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
- [x] Quantum Circuit class

Here is an example code of creating a QuantumCircuit class and add some gates to it.
```python
import Circuit
import Gate
circuit = Circuit.NumpyCircuit(2)
circuit.add_gate(Gate.CNOT(), [1, 0])
circuit.add_gate(Gate.Hadamard(), [1])
```



## Tensor Product and Measurement

- [x] Tensor Product
- [x] Measurement
- [x] Two/Three qubit gates
- [x] QuantumCircuit Class
- [x] Tensor Product of Gates/States
- [x] Measurement

Here is another example code of creating a QuantumCircuit class, add some gates to it, do the computation and measurement.

```python
import Circuit
circuit = Circuit.NumpyCircuit(4)
circuit.add_gate(Gate.Hadamard(), 0)
circuit.add_gate(Gate.Hadamard(), 1)
circuit.add_gate(Gate.Hadamard(), 2)
circuit.add_gate(Gate.Hadamard(), 3)
circuit.compute()
circuit.measureAll("0011")
```


## TestCode

- [x] Write test code that compare state vector result with qiskit
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






