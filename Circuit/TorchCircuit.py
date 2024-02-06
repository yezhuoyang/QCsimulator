'''
Quantum Circuit Class simulated by Pytorch
'''

import torch
import string
import numpy as np

import Gate
import Parameter
from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState
from typing import List, Union
import re
from .Circuit import QuantumCircuit


class PytorchCircuit(QuantumCircuit):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        '''
        By default, initial the input state as |000...0>
        '''
        self.state = QuantumState(qubit_number=num_qubits)
        self.gate_list = []
        '''
        The optimized calculation sequence, which should be a list of gate list
        For example:
              self.calc_sequence=[[Hadmard(0),Hadamard(1)],[CNOT(0,1)],[Hadmard(0),Hadamard(1)]]
              The calc_sequence can be optimized by the optimizer class before the actual calculation
        '''
        self.calc_sequence = []
        self.gate_num = 0
        self.calc_step = 0
        self.Debug = False
        '''
        The alphabet list we used to do the torch einsum
        string.ascii_lowercase='abcdefghijklmnopqrstuvwxyz',
        '''
        self._alphabet = list(string.ascii_lowercase)
        if self.num_qubits > len(self._alphabet):
            raise ValueError("Number of qubit exceeds the size of alphabet.")
        '''
        Initialize the state we actually use in torch.einsum calculation
        '''
        self._tfstate = torch.zeros(2 ** self.num_qubits, dtype=Parameter.qtype)
        '''
        Reshape self._tfstate to n dimensional tensor of shape [[2],[2],[2],..[2]]
        '''
        self._tfstate = self._tfstate.view([2] * self.num_qubits)

    def print_state(self) -> None:
        self.state.show_state()

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        '''
        :param gate: A quantum gate, for example: Hadmard
        :param qubit_indices: The indices of qubit that the gate act on, for example
        If the hardmard is acted on the third qubit, it should be [3]
        If the CNOT gate acted on the forth and fifth qubits, it should be [4,5]
        If the multi controlled Z gate, controlled by the first and second qubit, add on the fifth qubit, it should be [[1,2],5]
        :return:
        '''
        '''
        First, check the dimension of qubit_indices
        '''
        if isinstance(qubit_indices, int):
            if gate.num_qubits != 1:
                raise ValueError("qubit_indices for multi-qubit gate has to be a List!")
            if qubit_indices < 0 or qubit_indices >= self.num_qubits:
                raise ValueError(f"qubit_indices {qubit_indices} out of range!")
            '''
            The step is to make sure it can be converted to tuple in the fowllowing steps.
            '''
            qubit_indices = [qubit_indices]
        elif isinstance(qubit_indices, List):
            if isinstance(gate, MultiControlX | MultiControlZ):
                if not isinstance(qubit_indices[0], List):
                    raise ValueError(
                        "The first element of qubit_indices for multi control qubit gate has to be a List!")
                if not len(qubit_indices[0]) == (gate.num_qubits - 1):
                    raise ValueError("The qubit number of multi control qubit doesn't match with the qubit_indices")
                for index in qubit_indices[0]:
                    if index < 0 or index >= self.num_qubits:
                        raise ValueError(f"{index} in qubit_indices {qubit_indices} out of range!")
                    if qubit_indices[1] < 0 or qubit_indices[1] > self.num_qubits:
                        raise ValueError(f"{qubit_indices[1]} in qubit_indices {qubit_indices} out of range!")
            else:
                if gate.num_qubits != len(qubit_indices):
                    raise ValueError(
                        "The length of the qubit_indices list has to match with the qubit number of the gate!")
                for index in qubit_indices:
                    if index < 0 or index >= self.num_qubits:
                        raise ValueError(f"{index} in qubit_indices {qubit_indices} out of range!")
            '''Make sure that the qubit indices of single qubit gate be just an integer'''
            if gate.num_qubits == 1:
                qubit_indices = qubit_indices[0]
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]
        self.gate_list.append((gate, tuple(qubit_indices), self.gate_num))
        self.calc_sequence.append([(gate, tuple(qubit_indices), self.gate_num)])
        self.gate_num += 1

    def compute(self) -> None:
        while self._compute_step():
            continue
        return

    '''
    Take on step in computation. Only do one matrix multiplication between a gate and 
    the quantum state.
    Return value: Return True if the function caclulate a gate. 
    If it does nothing, return False
    '''

    def _compute_step(self) -> bool:
        if self.calc_step == len(self.calc_sequence):
            return False
        if self.Debug:
            print(f"Step :{self.calc_step}")
        '''
        TODO:
        '''
        self.calc_step += 1
        return True

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def load_qasm(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement load_qasm method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")