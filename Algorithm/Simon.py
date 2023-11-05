from typing import List

from Algorithm import QuantumAlgorithm

'''
Simon's algorithm
The input is a function f and a number s,
such that f(x1)==f(x2) iff x1+x2=s of x1==x2
Our goal is to find such s
'''


class Simon(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits
        self.uf = []
        self._s = 0

    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, alginput: List) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_input method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")
