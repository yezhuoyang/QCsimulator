import numpy as np
import Parameter


class QuantumGate:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement apply method.")


class Hadamard(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        # Define the Hadamard matrix for a single qubit
        hadamard = np.array([[1, 1], [1, -1]], dtype=Parameter.qtype) / np.sqrt(2)
        return hadamard


class PauliX(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        paulix = np.array([[0, 1], [1, 0]], dtype=Parameter.qtype)
        return paulix


class PauliY(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        pauliy = np.array([[0, -1j], [1j, 0]], dtype=Parameter.qtype)
        return pauliy


class PauliZ(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        pauliz = np.array([[1, 0], [0, -1]], dtype=Parameter.qtype)
        return pauliz


class RotateX(QuantumGate):
    def __init__(self, theta: Parameter.qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatex = np.array([[np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
                            [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)]], dtype=Parameter.qtype)
        return rotatex


class RotateY(QuantumGate):
    def __init__(self, theta: Parameter.qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatey = np.array(
            [[np.cos(self.theta / 2), -np.sin(self.theta / 2)], [np.sin(self.theta / 2), np.cos(self.theta / 2)]],
            dtype=Parameter.qtype)
        return rotatey


class RotateZ(QuantumGate):
    def __init__(self, theta: Parameter.qtype) -> None:
        super().__init__(num_qubits=1)
        self.theta = theta

    def matrix(self) -> np.ndarray:
        rotatez = np.array([[np.exp(-1j * self.theta / 2), 0], [0, np.exp(1j * self.theta / 2)]], dtype=Parameter.qtype)
        return rotatez


class Phase(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        phase = np.array([[1, 0], [0, 1j]])
        return phase


class TGate(QuantumGate):
    def __init__(self, num_qubits) -> None:
        super().__init__(num_qubits=1)

    def matrix(self) -> np.ndarray:
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=Parameter.qtype)
        return T


class CNOT(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=Parameter.qtype)
        return cnot


class CPhase(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        cphase = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=Parameter.qtype)
        return cphase


class Swap(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=Parameter.qtype)
        return swap


class ControlledZ(QuantumGate):
    def __init__(self) -> None:
        super().__init__(num_qubits=2)

    def matrix(self) -> np.ndarray:
        controlledZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=Parameter.qtype)
        return controlledZ


class Toffoli(QuantumGate):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits=3)

    def matrix(self) -> np.ndarray:
        toffoli = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]],
            dtype=Parameter.qtype)
        return toffoli


class Fredkin(QuantumGate):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits=3)

    def matrix(self) -> np.ndarray:
        fredkin = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
            dtype=Parameter.qtype)
        return fredkin
