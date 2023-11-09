'''
Algorithmic framework of shor algorithm
'''
from typing import List

from Algorithm import QuantumAlgorithm
import fractions
import random
import math

class Shor(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits
        self._N = 0
        self._solution = -1

    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, N: int) -> None:
        self._N = N

    '''
    Use classical algorithm to find the order of a.
    The result is the order r, such that a^r mod N = 1
    '''

    def order_finding_classical(self, a: int) -> int:
        if a == 1:
            raise ValueError("The basis for the order can not be 1")
        r = 2
        while True:
            if (a ** r % self._N) == 1:
                return r
            else:
                r += 1


    def order_finding_quantum(self,a:int)->int:
        return




    '''
    In this implmentation, we only need to return 1 non trivial factor of N
    '''

    def compute_result(self) -> None:
        '''
        First determine if N is a even number
        '''
        if self._N % 2 == 0:
            self._solution = 2
            return
        succeed = False
        while not succeed:
            a = random.randint(2, self._N - 1)
            d = math.gcd(a, self._N)
            if d > 1:
                self._solution = d
                return
            r = self.order_finding_classical(a)
            if (r % 2) == 1:
                continue
            d = math.gcd(a ** (r // 2) - 1, self._N)
            if d == 1:
                continue
            else:
                succeed = True
                self._solution=d

    @property
    def solution(self):
        return self._solution


