import numpy as np

import Parameter

'''
Implementation of quantum pure state of n qubit
'''


class QuantumState:
    '''
    First way of initialization: By
    '''

    def __init__(self, state_vector: np.ndarray[Parameter.qtype] = None, qubit_number: int = 1) -> None:
        '''
        If the user doesn't specify the state vecter, we will create a default state
        |000...000> for him.
        '''
        if state_vector is None:
            state_vector = np.array([0] * (2 ** qubit_number), dtype=Parameter.qtype)
            state_vector[0] = 1
        # The input dimension of state vector must be (n,)
        elif len(list(state_vector.shape)) > 1:
            raise ValueError(f"Invalid dimension: {state_vector} is not a vector.")
        self.qubit_number = qubit_number
        if 2 ** qubit_number != state_vector.shape[0]:
            raise ValueError("Qubit number doesn't match the shape of the state vector")
        self.state_vector = state_vector
        self.normalize()

    '''
    Reset the state.
    '''
    def reset_state(self, state_vector: np.ndarray[Parameter.qtype] = None) -> None:
        # The input dimension of state vector must be (n,)
        if len(list(state_vector.shape)) > 1:
            raise ValueError(f"Invalid dimension: {state_vector} is not a vector.")
        self.qubit_number = self.qubit_number
        '''
        If the user doesn't specify the state vecter, we will create a default state
        |000...000> for him.
        '''
        if state_vector is None:
            state_vector = np.array([0] * (2 ** self.qubit_number), dtype=Parameter.qtype)
            state_vector[0] = 1
        if 2 ** self.qubit_number != state_vector.shape[0]:
            raise ValueError("Qubit number doesn't match the shape of the state vector")
        self.state_vector = state_vector
        #self.normalize()

    def normalize(self) -> None:
        norm = np.sqrt(sum([abs(x) ** 2 for x in self.state_vector]))
        if norm == 0:
            raise ValueError(f"Invalid state vector: {self.state_vector} has norm 0")
        self.state_vector = np.array([x / norm for x in self.state_vector], Parameter.qtype)

    def inner_product(self, other_state: "QuantumState") -> np.ndarray:
        if self.state_vector.shape != other_state.state_vector.shape:
            raise ValueError("States must have the same dimension")
        return np.inner(self.state_vector, other_state.state_vector)

    def tensor_product(self, other_state: "QuantumState") -> "QuantumState":
        tensor_product_vector = np.kron(self.state_vector, other_state.state_vector)
        return QuantumState(np.array(tensor_product_vector, dtype=Parameter.qtype),qubit_number=self.qubit_number+other_state.qubit_number)

    def show_state(self) -> None:
        print(self.state_vector)
