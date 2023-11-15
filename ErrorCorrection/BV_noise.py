import Algorithm
import random





maximumqubit=14

def test_BV_qiskit_accuracy(self):
    for i in range(2, maximumqubit + 1):
        a = random.getrandbits(i - 1)
        b = random.getrandbits(1)
        bvalg = Algorithm.BVAlgorithm_qiskit(i)
        bvalg.set_input([a, b])
        bvalg.construct_circuit()
        bvalg.compute_result()
        self.assertEqual(bvalg.computed_a_value, a)



if __name__==







