import numpy as np

import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import re
from Algorithm import QuantumAlgorithm



class DuetchJosa(QuantumAlgorithm):
    
    
    def __init__(self, num_qubits) -> None:
        self.num_qubits = num_qubits 
        self.circuit= Circuit.Circuit(num_qubits)

    def set_input(self,Uf:List)->None:
        raise None
    
    
    def construct_circuit(self)->NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")
    
    
    def compute_result(self)->NotImplementedError:
         raise NotImplementedError("Subclasses must implement compute_result method.")
        

    '''
    Compiler the Uf gate given the List of Uf input
    '''
    def compile_Uf(self):
        return
    
    
    