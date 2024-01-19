import numpy as np
import Parameter
from typing import List, Union, Any


class QuantumGate:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self._dagger = False

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement matrix method.")

    def dagger(self) -> None:
        self._dagger = True

    def qasmstr(self) -> None:
        raise NotImplementedError("Subclasses must implement qasmstr method.")


class Identity(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        # Define the Hadamard matrix for a single qubit
        identity = np.array([[1, 0], [0, 1]], dtype=Parameter.qtype)
        return identity

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "Identity"

class Hadamard(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        # Define the Hadamard matrix for a single qubit
        hadamard = np.array([[1, 1], [1, -1]], dtype=Parameter.qtype) / np.sqrt(2)
        return hadamard if not self._dagger else np.matrix.getH(hadamard)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "Hadamard"


class PauliX(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        paulix = np.array([[0, 1], [1, 0]], dtype=Parameter.qtype)
        return paulix if not self._dagger else np.matrix.getH(paulix)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "X"


class PauliY(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        pauliy = np.array([[0, -1j], [1j, 0]], dtype=Parameter.qtype)
        return pauliy if not self._dagger else np.matrix.getH(pauliy)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "Y"


class PauliZ(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        pauliz = np.array([[1, 0], [0, -1]], dtype=Parameter.qtype)
        return pauliz if not self._dagger else np.matrix.getH(pauliz)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "Z"


class RotateX(QuantumGate):
    def __init__(self, theta: Parameter.qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatex = np.array([[np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
                            [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)]], dtype=Parameter.qtype)
        return rotatex if not self._dagger else np.matrix.getH(rotatex)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return f"Rx({self.theta})"


class RotateY(QuantumGate):
    def __init__(self, theta: Parameter.qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatey = np.array(
            [[np.cos(self.theta / 2), -np.sin(self.theta / 2)], [np.sin(self.theta / 2), np.cos(self.theta / 2)]],
            dtype=Parameter.qtype)
        return rotatey if not self._dagger else np.matrix.getH(rotatey)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return f"Ry({self.theta})"


class RotateZ(QuantumGate):
    def __init__(self, theta: Parameter.qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatez = np.array([[np.exp(-1j * self.theta / 2), 0], [0, np.exp(1j * self.theta / 2)]], dtype=Parameter.qtype)
        return rotatez if not self._dagger else np.matrix.getH(rotatez)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return f"Rz({self.theta})"


class Phase(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        phase = np.array([[1, 0], [0, 1j]])
        return phase if not self._dagger else np.matrix.getH(phase)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "S"


class TGate(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=Parameter.qtype)
        return t if not self._dagger else np.matrix.getH(t)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "T"


class CNOT(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=Parameter.qtype)
        return cnot if not self._dagger else np.matrix.getH(cnot)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "CNOT"


class CPhase(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        cphase = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]], dtype=Parameter.qtype)
        return cphase if not self._dagger else np.matrix.getH(cphase)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "CPhase"


class Swap(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=Parameter.qtype)
        return swap if not self._dagger else np.matrix.getH(swap)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "swap"


class ControlledZ(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        controlledz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=Parameter.qtype)
        return controlledz if not self._dagger else np.matrix.getH(controlledz)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "CZ"


class Toffoli(QuantumGate):

    def __init__(self) -> None:
        super().__init__(num_qubits=3)

    def matrix(self) -> np.ndarray:
        toffoli = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]],
            dtype=Parameter.qtype)
        return toffoli if not self._dagger else np.matrix.getH(toffoli)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "toffoli"


class Fredkin(QuantumGate):

    def __init__(self) -> None:
        super().__init__(num_qubits=3)

    def matrix(self) -> np.ndarray:
        fredkin = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
            dtype=Parameter.qtype)
        return fredkin if not self._dagger else np.matrix.getH(fredkin)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "fredkin"


'''
Use multi qubit to control another qubit.
param: act_condition is a list of 0 and 1. For example, for f(001)=1, the act_condition is [0,0,1]
When we create a MulticontrolX, the last qubit in the qubit_indices is the controlled one, while the first
n qubits represent the act_condition 
'''


class MultiControlX(QuantumGate):

    def __init__(self, numqubits: int, act_condition: List[int]) -> None:
        super().__init__(num_qubits=numqubits)
        self.act_condition = act_condition
        if not len(self.act_condition) == (numqubits - 1):
            raise ValueError("The number of act_condition must be equal to the number of qubits -1.")

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "MultiX"


'''
Use multi qubit to control another qubit.
param: act_condition is a list of 0 and 1. For example, for f(001)=1, the act_condition is [0,0,1]
When we create a MulticontrolZ, the last qubit in the qubit_indices is the controlled one, while the first
n qubits represent the act_condition 
'''


class MultiControlZ(QuantumGate):

    def __init__(self, numqubits: int, act_condition: List[int]) -> None:
        super().__init__(num_qubits=numqubits)
        self.act_condition = act_condition
        if not len(self.act_condition) == (numqubits - 1):
            raise ValueError("The number of act_condition must be equal to the number of qubits -1.")

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return "MultiZ"


'''
Add a layer of all hadamard gate    
'''


class AllHadamard(QuantumGate):
    def __init__(self, numqubits: int) -> None:
        super().__init__(num_qubits=numqubits)

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def __str__(self) -> str:
        return f"Allhadmard({self.num_qubits})"


'''
Quantum gate that has already been merged, but does not belong to any
normal gate class.
This class is used in optimization process
'''


class MergedGate(QuantumGate):
    def __init__(self, numqubits: int, matrix: np.ndarray) -> None:
        super().__init__(num_qubits=numqubits)
        N = 1 << numqubits
        if matrix.shape != (N, N):
            raise ValueError("Dimension of matrix is not consistent with qubit number")
        self._matrix = matrix

    def qasmstr(self) -> str:
        raise NotImplementedError("Subclasses must implement qasmstr method.")

    def matrix(self) -> np.ndarray:
        return self._matrix if not self._dagger else np.matrix.getH(self._matrix)

    def __str__(self) -> str:
        return "Merge"


'''
Merge the single qubit gate given the list of all gates
'''


def merge_single(gatelist):
    return


'''
Merge the doubli qubits gate given the list of all gates
'''


def merge_double(gatelist):
    return
