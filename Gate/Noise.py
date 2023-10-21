import numpy as np
import Parameter


class QuantumNoise:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement apply method.")


class Depolarizing(QuantumNoise):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement apply method.")

class AmplitudeDamping(QuantumNoise):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement apply method.")

class PhaseDamping(QuantumNoise):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement apply method.")

class PhaseFlip(QuantumNoise):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement apply method.")

class BitFlip(QuantumNoise):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement apply method.")

class BitPhaseFlip(QuantumNoise):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def matrix(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement apply method.")
