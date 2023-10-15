from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState


class QuantumCircuit:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")


'''
Quantum Circuit class simulated by Numpy
'''
class NumpyCircuit(QuantumCircuit):
    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")


'''
Quantum Circuit Class simulated by Pytorch
'''


class PytorchCircuit(QuantumCircuit):

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")


'''
Quantum Circuit Class simulated by GPU
'''


class GPUCircuit(QuantumCircuit):

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")
