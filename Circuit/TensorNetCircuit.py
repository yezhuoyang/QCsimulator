import numpy as np

import Gate
import Parameter
from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState
from typing import List, Union
import re
from .Circuit import QuantumCircuit



'''
Quantum Circuit Class using stabilizer
'''


class TensorNetCircuit(QuantumCircuit):

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def load_qasm(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement load_qasm method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")


