import sys

import numpy as np
import random

sys.path.append('..')
import Algorithm
from Parameter import maximumqubit
import unittest

'''
Generate a random database, portion is the portion of keys in the 
database with value 1.
'''


def generate_random_database(qubit_num: int, portion: float) -> list:
    n = (1 << (qubit_num))
    A = int(n * portion) + 1
    B = n - A
    uf = [0] * B
    pos = 0
    for i in range(0, A):
        dice = 0
        while dice < 5:
            dice = random.randint(0, 5)
            if dice > 4:
                uf.insert(pos, 1)
            pos = (pos + 1) % len(uf)
    return uf


class TestGrover(unittest.TestCase):
    def test_grover_success(self):
        for i in range(2, maximumqubit + 1):
            gvalg = Algorithm.Grover(i)
            db_uf = generate_random_database(i - 1, 0.1)
            gvalg.set_input(db_uf)
            gvalg.construct_circuit()
            gvalg.compute_result()
            result = gvalg.solution
            self.assertEqual(db_uf[result], 1)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
