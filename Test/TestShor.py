import unittest
import random
import Algorithm
from primePy import primes

class TestShor(unittest.TestCase):
    def test_classical(self):
        shoralg = Algorithm.Shor(4)
        for i in range(0, 10):
            while True:
                N = random.randint(5, 100)
                if not primes.check(N):
                    break
            print(N)
            shoralg.set_input(N)
            shoralg.compute_result()
            print(f"Solution:{shoralg.solution}")
            self.assertEqual((N%shoralg.solution)==0, True)
        return


if __name__ == '__main__':
    unittest.main()
