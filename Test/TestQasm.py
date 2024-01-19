import sys
from qiskit import execute
from qiskit import QuantumCircuit, Aer
import numpy as np
import time
from memory_profiler import memory_usage

sys.path.append('..')
import Circuit
from pathlib import Path
from colorama import Fore
import tracemalloc
import Parameter
import linecache
from threading import Thread

def profile_simulation(func, *args):
    """Function to profile memory usage of a simulation function."""
    mem_usage = memory_usage((func, args), max_usage=True)
    return mem_usage


def read_qasmfile(path):
    with open(path, 'r') as f:
        return f.read()


'''
Because qutip order bit reversely, we need to 
reverse it back before comparing with our result
'''


def reorder(state_vector, num_qubits):
    new_state = np.zeros(1 << num_qubits, dtype=Parameter.qtype)
    for i in range(0, 1 << num_qubits):
        binary_string = "{0:b}".format(i)
        if len(binary_string) < num_qubits:
            binary_string = '0' * (num_qubits - len(binary_string)) + binary_string
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
    #simulator = Circuit.NumpyCircuit(1)
    simulator = Circuit.StateDictCircuit(1)
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
    fidelity = fidelity * fidelity.conjugate()
    return fidelity, np.all(np.isclose(state_vector, qiskit_state_vector))


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "< frozen importlib._bootstrap_external >"),
        tracemalloc.Filter(False, "<unknown>"),
        tracemalloc.Filter(False, "<frozen abc>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def main():
    # get the directory of qasm files and make sure it's a directory
    qasm_dir = Path("C:/Users/yezhu/PycharmProjects/QCsimulator/Test/singleqasm")
    assert qasm_dir.is_dir()

    # iterate the qasm files in the directory
    for qasm_file in qasm_dir.glob("**/*.qasm"):
        # read the qasm file

        with open(qasm_file, "r") as f:
            qasm_string = f.read()


        #tracemalloc.reset_peak()
        time_start = time.perf_counter()
        tracemalloc.start()
        state_vector = simulate(qasm_string)
        #first_size, first_peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        #display_top(snapshot)
        time_elapsed1 = (time.perf_counter() - time_start)
        tracemalloc.stop()


        time_start = time.perf_counter()
        qiskit_state_vector = simulate_by_qiskit(qasm_string)
        time_elapsed2 = (time.perf_counter() - time_start)
        print("(Your simulator)%5.5f secs, (Qiskit)%5.5f secs" % (time_elapsed1,time_elapsed2))

        # compare the results!
        fidelity, passed = compare(state_vector, qiskit_state_vector)
        if passed:
            print(Fore.GREEN + f"Congradulations! You pass the test file \"{qasm_file.name}\" with fidelity {fidelity}")
        else:
            print(Fore.RED + f"Sorry, you fail to pass the test file \"{qasm_file.name}\" with fidelity {fidelity}")
        print(Fore.WHITE)

if __name__ == '__main__':
    sys.path.append('..')
    main()
