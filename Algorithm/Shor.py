'''
Algorithmic framework of shor algorithm
'''
from typing import List

from Algorithm import QuantumAlgorithm


class Shor(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits
        self._N=0


    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, N:int) -> None:
        self._N=N
        


    '''
    Use classical algorithm to find the order of a.
    The result is the order r, such that a^r mod N = 1
    '''
    def order_finding_classical(self,a:int) -> int:
        
        return

    '''
    In this implmentation, we only need to return 1 non trivial factor of N
    '''
    def compute_result(self) -> None:
        '''
        First determine if N is a even number
        '''
        if self._N%2==0:
            self.solution=2
            return
        
        
        
        
        
        
