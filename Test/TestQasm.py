import sys
from qiskit import execute
from qiskit import QuantumCircuit, Aer
import numpy as np

import Parameter

sys.path.append('..')
import Circuit
from pathlib import Path
from colorama import Fore


def read_qasmfile(path):
    with open(path, 'r') as f:
        return f.read()


'''
Because qutip order bit reversely, we need to 
reverse it back before comparing with our result
'''


def reorder(state_vector, num_qubits):
    new_state = np.zeros(1 << num_qubits,dtype=Parameter.qtype)
    for i in range(0, 1 << num_qubits):
        binary_string = "{0:b}".format(i)
        if len(binary_string)<num_qubits:
            binary_string='0'*(num_qubits-len(binary_string))+binary_string
        newString = reversed(binary_string)
        newString = "".join(newString)
        new_state[int(newString, 2)] = state_vector[i]
    return new_state


def simulate_by_qiskit(qasmstr):
    circuit = QuantumCircuit.from_qasm_str(qasmstr)
    num_qubits = circuit.num_qubits
    # Do the simulation, return the result and get the state vector
    # Let's simulate our circuit in order to get the final state vector!
    svsim = Aer.get_backend('statevector_simulator')
    result = execute(circuit, svsim).result().get_statevector()
    return reorder(np.array(result.data), num_qubits)


def simulate(qasm_string) -> np.ndarray:
    simulator = Circuit.NumpyCircuit(1)
    simulator.load_qasm(qasm_string)
    simulator.compute()
    return simulator.state_vector()


def compare(state_vector, qiskit_state_vector):
    """Our comparison function for your grade

    Args:
        state_vector: your state vector amplitude list
        cirq_state_vector: cirq's state vector amplitude list

    Returns:
        Some value influencing your grade, subject to change :)
    """
    fidelity = np.inner(state_vector, qiskit_state_vector)
    fidelity=fidelity*fidelity.conjugate()
    return fidelity, np.all(np.isclose(state_vector, qiskit_state_vector))


# get the directory of qasm files and make sure it's a directory
qasm_dir = Path("C:/Users/yezhu/PycharmProjects/QCsimulator/Test/naiveqasm")
assert qasm_dir.is_dir()

# iterate the qasm files in the directory
for qasm_file in qasm_dir.glob("**/*.qasm"):
    # read the qasm file


    with open(qasm_file, "r") as f:
        qasm_string = f.read()

    # run your simulate function on the qasm string
    state_vector = simulate(qasm_string)
    # run cirq's simulator on the qasm string
    qiskit_state_vector = simulate_by_qiskit(qasm_string)
    # compare the results!
    fidelity, passed = compare(state_vector, qiskit_state_vector)
    if passed:
        print(Fore.GREEN + f"Congradulations! You pass the test file \"{qasm_file.name}\" with fidelity {fidelity}")
    else:
        print(Fore.RED + f"Sorry, you fail to pass the test file \"{qasm_file.name}\" with fidelity {fidelity}")
