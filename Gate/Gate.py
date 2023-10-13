import numpy as np



class QuantumGate:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def matrix(self):
        raise NotImplementedError("Subclasses must implement apply method.")

    

class Hadamard(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)

    def matrix(self):
        # Define the Hadamard matrix for a single qubit
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return hadamard 
    
    
    
class PauliX(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)
        
    def matrix(self):
        paulix=np.array([[0,1],[1,0]])
        return paulix
    
    
    
class PauliY(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)
        
    def matrix(self):
        pauliy=np.array([[0,-1j],[1j,0]])       
        return super().matrix(pauliy)
    
    
    
class PauliZ(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)
        
    def matrix(self):
        pauliz=np.array([[1,0],[0,-1]])    
        return pauliz
    

class Phase(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=1)
        
    def matrix(self):
        phase=np.array([[1,0],[0,1j]])    
        return phase 
    
    
class TGate(QuantumGate):
    def __init__(self, num_qubits):
        super().__init__(num_qubits)
        
    def matrix(self):
        T=np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
        return T
    
    
class CNOT(QuantumGate):
    def __init__(self):
        super().__init__(num_qubits=2)
    
    def matrix(self):
        cnot=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        return cnot 
