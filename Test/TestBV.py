import sys
import unittest

from Parameter import maximumqubit
import random

sys.path.append('..')
import Algorithm


class TestBV(unittest.TestCase):
    def test_BV_accuracy(self):
        for i in range(2, maximumqubit + 1):
            a = random.getrandbits(i - 1)
            b = random.getrandbits(1)
            bvalg = Algorithm.BVAlgorithm(i)
            bvalg.set_input([a, b])
            bvalg.construct_circuit()
            bvalg.compute_result()
            self.assertEqual(bvalg.computed_a_value, a)



'''
Example of usage
    alg = Algorithm.BVAlgorithm(9)
    a=0b10101001
    b=1
    alg.set_input([a,b])
    alg.construct_circuit()
    alg.compute_result()
    alg.circuit.state.show_state_dirac()
'''

def main():
    unittest.main()


if __name__ == "__main__":
    main()
