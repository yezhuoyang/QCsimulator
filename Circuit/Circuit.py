from ..Gate import QuantumGate


    
class QuantumCircuit:
    def __init__(self, num_qubits:int)->None:
        self.num_qubits = num_qubits
        
        
   
       
    def add_gate(self, gate: QuantumGate, qubit_indices: list[int])->None:
        raise NotImplementedError("Subclasses must implement add_gate method.")
    
  
  
    
    def compute(self)->None:
        return NotImplementedError("Subclasses must implement compute method.")
    
    
    
    
    def measure(self,qubit_index: int)->None:
        return NotImplementedError("Subclasses must implement measure method.")
    
    
    
    
    def visulize(self)->None:
        return NotImplementedError("Subclasses must implement visulize method.")




    def transpile_qase(self,qasmStr:str)->None:
        return NotImplementedError("Subclasses must implement transpile_qase method.")



    def to_qasm(self)->str:
        return NotImplementedError("Subclasses must implement to_qasm method.")
    
    


    
