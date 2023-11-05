'''
Quantum chip class for research in layout synthesis
'''
class QuantumChip:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits


    def graph(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement graph method.")

