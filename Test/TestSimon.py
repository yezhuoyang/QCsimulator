import random
import sys
import unittest

sys.path.append('..')
import Algorithm
from Parameter import maximumqubit

alg = Algorithm.Simon(6)

'''
Generate a ramdom simon's algorithm 
imput given s.
f(x)=0 or 1
f(x1)=f(x2) if x1+x2=s,
x1,x2,s are all n-bit integer and + means bitwise add
For example, x1=010, x2=100, x1+x2=110 (x1^x2=s)

'''


def generate_random_simon_func(qubit_num: int, s: int) -> list:
    N = 1 << qubit_num
    if s >= N:
        raise ValueError("s outside the possible range!")
    uf = [0] * N
    '''
    Scan through all index and randomly flip function value
    of key pair (x1,x2)
    '''
    used_value = []
    calculated_key = []

    for x1 in range(N):
        '''
        If x1 already has a value, continue
        '''
        if x1 in calculated_key:
            continue
        '''
        Generate a value that haven't been used 
        '''
        while True:
            value = random.getrandbits(qubit_num)
            if not value in used_value:
                x2 = x1 ^ s
                uf[x1] = value
                uf[x2] = value
                used_value.append(value)
                calculated_key.append(x1)
                calculated_key.append(x2)
                break
    return uf


class TestSimon(unittest.TestCase):
    def test_result(self):
        for i in range(4, maximumqubit + 1, 2):
            inputsize = i // 2
            n = (1 << (inputsize))
            s = 0
            while s == 0:
                s = random.getrandbits(inputsize)
            print(f"Testing s={s}")
            uf = generate_random_simon_func(inputsize, s)
            simonalg = Algorithm.Simon(i)
            simonalg.set_input(uf)
            simonalg.construct_circuit()
            simonalg.compute_result()
            self.assertEqual(simonalg.solution, s)



def main():
    unittest.main()


'''
Example:
    alg = Algorithm.DuetchJosa(4)
    uf = [1, 1, 1, 1, 1, 1, 1, 1]
    alg.set_input(uf)
    alg.construct_circuit()
    alg.compute_result()
    alg.circuit.state.show_state_dirac()
'''

if __name__ == "__main__":
    main()
