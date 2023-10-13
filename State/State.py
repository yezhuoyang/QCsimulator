import numpy as np
import Parameter






class QuantumState:
    def __init__(self, state_vector:np.ndarray[Parameter.qtype]):
        self.state_vector = state_vector
    
    
    
    def normalize(self)->None:
        norm = sum([abs(x)**2 for x in self.state_vector])
        self.state_vector = [x/norm for x in self.state_vector]
    
    
    
    def inner_product(self, other_state : "QuantumState"):
        if len(self.state_vector) != len(other_state.state_vector):
            raise ValueError("States must have the same dimension")
        return sum([self.state_vector[i].conjugate() * other_state.state_vector[i] for i in range(len(self.state_vector))])
    
    
    
    def tensor_product(self, other_state: "QuantumState") :
        tensor_product_vector = []
        for i in range(len(self.state_vector)):
            for j in range(len(other_state.state_vector)):
                tensor_product_vector.append(self.state_vector[i] * other_state.state_vector[j])
        return QuantumState(np.array(tensor_product_vector))

