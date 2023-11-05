from typing import List


def convert_int_to_list(num_qubits: int, alginput: int):
    controllist = []
    k = alginput
    for i in range(0, num_qubits):
        controllist.insert(0, k % 2)
        k = (k >> 1)
    return controllist


def convert_list_to_int(num_qubits: int, bitlist: List):
    result = 0
    for i in range(num_qubits):
        result = result + (bitlist[i] << (num_qubits - 1 - i))
    return result
