import sys

sys.path.append('..')

'''
Implement important circuit identity
'''
import Circuit
import Gate

if __name__ == "__main__":
    C1 = Circuit.NumpyCircuit(2)
    C2 = Circuit.NumpyCircuit(2)
    C1.set_store(True)
    C2.set_store(True)
    PauliNameDict = ["I", "X", "Y", "Z"]
    GateDist = {"I": Gate.Identity, "X": Gate.PauliX, "Y": Gate.PauliY, "Z": Gate.PauliZ}
    for i in range(0, len(PauliNameDict)):
        for j in range(0, len(PauliNameDict)):
            if j == i:
                continue
            """
            Test CNOT(i,j)P_i P_J CNOT(i,j)=?
            """
            C1.clear_all()
            C1.add_gate(Gate.CNOT(), [0, 1])
            C1.add_gate(GateDist[PauliNameDict[i]](), 0)
            C1.add_gate(GateDist[PauliNameDict[j]](), 1)
            C1.add_gate(Gate.CNOT(), [0, 1])
            C1.compute()
            #print("CNOT(0,1){A}{B}CNOT(0,1)".format(A=PauliNameDict[i], B=PauliNameDict[j]))
            #print(C1.matrix)
            for p in range(0, len(PauliNameDict)):
                for k in range(0, len(PauliNameDict)):
                    C2.clear_all()
                    C2.add_gate(GateDist[PauliNameDict[p]](), 0)
                    C2.add_gate(GateDist[PauliNameDict[k]](), 1)
                    C2.compute()
                    #print("{C}{D}".format(C=PauliNameDict[p], D=PauliNameDict[k]))
                    #print(C2.matrix)
                    if (C1.equal_result(C2)):
                        print("CNOT(0,1){A}{B}CNOT(0,1)={C}{D}".format(A=PauliNameDict[i], B=PauliNameDict[j],
                                                                       C=PauliNameDict[p], D=PauliNameDict[k]))
