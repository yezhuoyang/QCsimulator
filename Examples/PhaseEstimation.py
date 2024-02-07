import qiskit
from qiskit_aer import AerSimulator
from qiskit.circuit.library.standard_gates import SdgGate


if __name__ == "__main__":
    circuit = qiskit.QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.h(1)
    circuit.x(2)

    circuit.csdg(0,2)
    circuit.append(SdgGate.control(2), [0, 1, 2])
    simulator = AerSimulator()
