import sys

sys.path.append('..')
import Circuit
import Gate


def read_qasmfile(path):
    with open(path, 'r') as f:
        return f.read()

qasmStr = read_qasmfile("C:/Users/yezhu/PycharmProjects/QCsimulator/Test/qasmBenchMark/alu-bdd_288.qasm")

C=Circuit.NumpyCircuit(10)
C.load_qasm(qasmStr)
