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


3. Every important function should pass at least one unittest.
   This is a unittest example for DJ algorithm
```python
import random

sys.path.append('..')
import Algorithm
from Parameter import maximumqubit
import unittest


def generate_random_balance(qubit_num: int) -> np.ndarray:
    n = (1 << (qubit_num - 1))
    uf = [0] * n
    pos = 0
    for i in range(0, n):
        dice = 0
        while dice < 5:
            dice = random.randint(0, 5)
            if dice > 4:
                uf.insert(pos, 1)
            pos = (pos + 1) % len(uf)
    return uf


class TestDJ(unittest.TestCase):
    def test_constant(self):
        for i in range(2, maximumqubit + 1):
            inputsize = i - 1
            n = (1 << (inputsize))
            dice = random.randint(0, 1)
            uf = [dice] * n
            djalg = Algorithm.DuetchJosa(i)
            djalg.set_input(uf)
            djalg.construct_circuit()
            djalg.compute_result()
            self.assertEqual(djalg.is_balance(), False)

    def test_balance(self):
        for i in range(2, maximumqubit + 1):
            uf = generate_random_balance(i-1)
            djalg = Algorithm.DuetchJosa(i)
            djalg.set_input(uf)
            djalg.construct_circuit()
            djalg.compute_result()
            self.assertEqual(djalg.is_balance(), True)


def main():
    unittest.main()

'''
Example:
    alg = Algorithm.DuetchJosa(4)
    uf = [1, 1, 1, 1, 1, 1, 1, 1]
    alg.set_input(uf)
    alg.construct_circuit()
    alg.compute_result()
    alg.circuit.state.show_state_dirac()
'''


if __name__ == "__main__":
    main()
```



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

- [x] Deutsch's algorithm
- [ ] Grover's algorithm
- [ ] QFFT
- [ ] Shor's algorithm
- [ ] HHL algorithm
- [x] BerstainVazarani
- [ ] Simon
- [ ] Quantum phase Estimation
- [ ] Design Parameter Circuit class
- [ ] Gradient Calculation
- [ ] BackPropagation
- [ ] QAOA algorithm
- [ ] VQE algorithm


## Layout Synthesis

- [ ] Quantum chip class
- [ ] Transpile circuit to chips
- [ ] Layout synthesis benchmark


## Compatibility

- [ ] Compile to qasm 2.0
- [x] Load qasm 2.0 assembly and simulate
- [ ] Compile to qasm 3.0
- [ ] Load qasm 3.0 assembly and simulate
- [ ] Visualization
- [ ] Support ZX-calculus Visulization

Here is a qasm 2.0 code example
```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
```
We can load and simulate the above qasm 2.0 by calling the load_qasm method of quantum circuit

```python
import Circuit
simulator = Circuit.NumpyCircuit(8)
simulator.load_qasm(qasm_string)
simulator.compute()
print(simulator.state_vector())
```


## Density Matrix and Quantum Noise


- [ ] Density Matrix Class
- [ ] Quantum Noise Class
- [ ] Bitflip Noise
- [ ] Decoherence Noise

## Error Correction Code

- [ ] Stabilizer Codes
- [ ] Surface Code
- [ ] LDPC Code

## Pulse Simulation
Given a physical qubit model driven by pulse, automatically generate all pulse sequence


## BenchMark
- [ ] Compare the simulation speed with qiskit
- [ ] Compare the simulation speed with pennylane
- [ ] Compare the simulation speed with cirq
- [ ] Compare the simulation speed with Torchquantum


# Contact
I'm currently a Master student in UCLA studying quantum computation [MQST webpage](https://qst.ucla.edu/). I'm especially interested in quantum architecure and quantum algorithm and I'm looking for a PHD position in this year.
My email address is yezhuoyang98@g.ucla.edu


