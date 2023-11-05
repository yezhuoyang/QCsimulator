'''
Schedular class, that will assign and generate a qubit mapping given quantum circuit and quantum chips
'''
import networkx as nx


class Schedular:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def set_chip(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_chip method.")

    def set_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_circuit method.")

    def schedule(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement graph method.")

