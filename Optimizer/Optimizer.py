import numpy as np
import Parameter
from Circuit import QuantumCircuit
from typing import List, Union, Any

'''
Calculate the dependency graph of a given gatelist
With the dependency graph, we can derive the topological order of the graph
'''


def calc_circuit_dag(gitelist):
    return


'''
Calculate the topological order of a give dependency graph
'''
def topoorder(dag):
    return



'''
Circuit Optimizer Class
The optimizer can only change the circuit.calc_sequence
'''


class CircuitOptimizer:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit

    def optimize(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement optimize method.")

    '''
    Print method for debug
    '''

    def print(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement print method.")


'''
Align all gates into different columns.
Fill the column as much as possible.
For example, the input circuit sequence of length 9 looks like:
        ---H---X---Y---------------
        --------------Z-T------H---
        -H-----------------Tdg-----H
After optimization, it should look like:
        ---H---X---Y---
        ---Z---T---H---
        ---H---Tdg-H---   
Now, the input circuit sequence only has length 3. This reduce the time to do
kroneck product   
'''


class LayerAligner(CircuitOptimizer):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit

    def optimize(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement optimize method.")


'''
Merge all single qubit gates next to each other as much as possible 
For example, the input circuit sequence of length 9 looks like:
        ---H---X---Y---------------
        --------------Z-T------H---
        -H-----------------Tdg-----H
After optimization, it should look like:
        ---G1=H*X*Y---
        ---G2Z*T*H---
        ---G3=H*Tdg*H---   
'''


class SingleQubitGateMerger(CircuitOptimizer):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit

    def optimize(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement optimize method.")


'''
Merge all two qubit gates next to each other as much as possible 
For example, the input circuit sequence of length 9 looks like:
        ---C---X--C--
           |   |  |  
        ---X---C--X--
After optimization, it should look like:
        ---S---
           |
        ---S---
'''
class TwoQubitGateMerger(CircuitOptimizer):
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit

    def optimize(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement optimize method.")


