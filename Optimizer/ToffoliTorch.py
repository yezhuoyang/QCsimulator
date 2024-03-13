import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import argparse

import torchquantum as tq
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np


class QModelFiveGeneral(tq.QuantumModule):
    def __init__(self, q0control, q1control, q2control, q3control, q4control):
        super().__init__()
        self.n_wires = 3
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)
        self.u3_6 = tq.U3(has_params=True, trainable=True)
        self.u3_7 = tq.U3(has_params=True, trainable=True)
        self.u3_8 = tq.U3(has_params=True, trainable=True)
        self.u3_9 = tq.U3(has_params=True, trainable=True)
        self.u3_10 = tq.U3(has_params=True, trainable=True)
        self.u3_11 = tq.U3(has_params=True, trainable=True)
        self.u3_12 = tq.U3(has_params=True, trainable=True)
        self.u3_13 = tq.U3(has_params=True, trainable=True)
        self.u3_14 = tq.U3(has_params=True, trainable=True)
        self.u3_15 = tq.U3(has_params=True, trainable=True)
        self.u3_16 = tq.U3(has_params=True, trainable=True)
        self.u3_17 = tq.U3(has_params=True, trainable=True)

        self.cu3_0 = tq.CU3(has_params=True, trainable=True)
        self.q0control = q0control
        self.cu3_1 = tq.CU3(has_params=True, trainable=True)
        self.q1control = q1control
        self.cu3_2 = tq.CU3(has_params=True, trainable=True)
        self.q2control = q2control
        self.cu3_3 = tq.CU3(has_params=True, trainable=True)
        self.q3control = q3control
        self.cu3_4 = tq.CU3(has_params=True, trainable=True)
        self.q4control = q4control

    def forward(self, q_device: tq.QuantumDevice):
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.u3_2(q_device, wires=2)

        self.cu3_0(q_device, wires=self.q0control)

        self.u3_3(q_device, wires=0)
        self.u3_4(q_device, wires=1)
        self.u3_5(q_device, wires=1)

        self.cu3_1(q_device, wires=self.q1control)

        self.u3_6(q_device, wires=2)
        self.u3_7(q_device, wires=0)
        self.u3_8(q_device, wires=1)

        self.cu3_2(q_device, wires=self.q2control)

        self.u3_9(q_device, wires=1)
        self.u3_10(q_device, wires=2)
        self.u3_11(q_device, wires=0)

        self.cu3_3(q_device, wires=self.q3control)

        self.u3_12(q_device, wires=1)
        self.u3_13(q_device, wires=2)
        self.u3_14(q_device, wires=2)

        self.cu3_4(q_device, wires=self.q4control)

        self.u3_15(q_device, wires=1)
        self.u3_16(q_device, wires=2)
        self.u3_17(q_device, wires=2)


class QModelFive1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 3
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)
        self.u3_6 = tq.U3(has_params=True, trainable=True)
        self.u3_7 = tq.U3(has_params=True, trainable=True)
        self.u3_8 = tq.U3(has_params=True, trainable=True)
        self.u3_9 = tq.U3(has_params=True, trainable=True)
        self.u3_10 = tq.U3(has_params=True, trainable=True)
        self.u3_11 = tq.U3(has_params=True, trainable=True)
        self.u3_12 = tq.U3(has_params=True, trainable=True)
        self.u3_13 = tq.U3(has_params=True, trainable=True)

        self.cu3_0 = tq.CU3(has_params=True, trainable=True)

        self.cu3_1 = tq.CU3(has_params=True, trainable=True)

        self.cu3_2 = tq.CU3(has_params=True, trainable=True)

        self.cu3_3 = tq.CU3(has_params=True, trainable=True)

        self.cu3_4 = tq.CU3(has_params=True, trainable=True)

    def forward(self, q_device: tq.QuantumDevice):
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.u3_2(q_device, wires=2)

        self.cu3_0(q_device, wires=[0, 1])

        self.u3_3(q_device, wires=0)
        self.u3_4(q_device, wires=1)

        self.cu3_1(q_device, wires=[1, 2])

        self.u3_5(q_device, wires=1)
        self.u3_6(q_device, wires=2)

        self.cu3_2(q_device, wires=[0, 1])

        self.u3_7(q_device, wires=0)
        self.u3_8(q_device, wires=1)

        self.cu3_3(q_device, wires=[1, 2])

        self.u3_9(q_device, wires=1)
        self.u3_10(q_device, wires=2)

        self.cu3_4(q_device, wires=[0, 1])

        self.u3_11(q_device, wires=0)
        self.u3_12(q_device, wires=1)
        self.u3_13(q_device, wires=2)


class QModelFive2(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 3
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)
        self.u3_6 = tq.U3(has_params=True, trainable=True)
        self.u3_7 = tq.U3(has_params=True, trainable=True)
        self.u3_8 = tq.U3(has_params=True, trainable=True)
        self.u3_9 = tq.U3(has_params=True, trainable=True)
        self.u3_10 = tq.U3(has_params=True, trainable=True)
        self.u3_11 = tq.U3(has_params=True, trainable=True)
        self.u3_12 = tq.U3(has_params=True, trainable=True)
        self.u3_13 = tq.U3(has_params=True, trainable=True)

        self.cu3_0 = tq.CU3(has_params=True, trainable=True)

        self.cu3_1 = tq.CU3(has_params=True, trainable=True)

        self.cu3_2 = tq.CU3(has_params=True, trainable=True)

        self.cu3_3 = tq.CU3(has_params=True, trainable=True)

        self.cu3_4 = tq.CU3(has_params=True, trainable=True)

    def forward(self, q_device: tq.QuantumDevice):
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.u3_2(q_device, wires=2)

        self.cu3_0(q_device, wires=[1, 2])

        self.u3_3(q_device, wires=0)
        self.u3_4(q_device, wires=1)

        self.cu3_1(q_device, wires=[0, 1])

        self.u3_5(q_device, wires=1)
        self.u3_6(q_device, wires=2)

        self.cu3_2(q_device, wires=[1, 2])

        self.u3_7(q_device, wires=0)
        self.u3_8(q_device, wires=1)

        self.cu3_3(q_device, wires=[0, 1])

        self.u3_9(q_device, wires=1)
        self.u3_10(q_device, wires=2)

        self.cu3_4(q_device, wires=[1, 2])

        self.u3_11(q_device, wires=0)
        self.u3_12(q_device, wires=1)
        self.u3_13(q_device, wires=2)


class QModelFive3(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 3
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)
        self.u3_6 = tq.U3(has_params=True, trainable=True)
        self.u3_7 = tq.U3(has_params=True, trainable=True)
        self.u3_8 = tq.U3(has_params=True, trainable=True)
        self.u3_9 = tq.U3(has_params=True, trainable=True)
        self.u3_10 = tq.U3(has_params=True, trainable=True)
        self.u3_11 = tq.U3(has_params=True, trainable=True)
        self.u3_12 = tq.U3(has_params=True, trainable=True)
        self.u3_13 = tq.U3(has_params=True, trainable=True)

        self.cu3_0 = tq.CU3(has_params=True, trainable=True)

        self.cu3_1 = tq.CU3(has_params=True, trainable=True)

        self.cu3_2 = tq.CU3(has_params=True, trainable=True)

        self.cu3_3 = tq.CU3(has_params=True, trainable=True)

        self.cu3_4 = tq.CU3(has_params=True, trainable=True)

    def forward(self, q_device: tq.QuantumDevice):
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.u3_2(q_device, wires=2)

        self.cu3_0(q_device, wires=[1, 0])

        self.u3_3(q_device, wires=0)
        self.u3_4(q_device, wires=1)

        self.cu3_1(q_device, wires=[1, 2])

        self.u3_5(q_device, wires=1)
        self.u3_6(q_device, wires=2)

        self.cu3_2(q_device, wires=[1, 0])

        self.u3_7(q_device, wires=0)
        self.u3_8(q_device, wires=1)

        self.cu3_3(q_device, wires=[1, 2])

        self.u3_9(q_device, wires=1)
        self.u3_10(q_device, wires=2)

        self.cu3_4(q_device, wires=[1, 0])

        self.u3_11(q_device, wires=0)
        self.u3_12(q_device, wires=1)
        self.u3_13(q_device, wires=2)


class QModelFour1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 3
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)
        self.u3_6 = tq.U3(has_params=True, trainable=True)
        self.u3_7 = tq.U3(has_params=True, trainable=True)
        self.u3_8 = tq.U3(has_params=True, trainable=True)
        self.u3_9 = tq.U3(has_params=True, trainable=True)
        self.u3_10 = tq.U3(has_params=True, trainable=True)

        self.cu3_0 = tq.CU3(has_params=True, trainable=True)

        self.cu3_1 = tq.CU3(has_params=True, trainable=True)

        self.cu3_2 = tq.CU3(has_params=True, trainable=True)

        self.cu3_3 = tq.CU3(has_params=True, trainable=True)

    def forward(self, q_device: tq.QuantumDevice):
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.u3_2(q_device, wires=2)

        self.cu3_0(q_device, wires=[0, 1])

        self.u3_3(q_device, wires=0)
        self.u3_4(q_device, wires=1)

        self.cu3_1(q_device, wires=[1, 2])

        self.u3_5(q_device, wires=1)
        self.u3_6(q_device, wires=2)

        self.cu3_2(q_device, wires=[0, 1])

        self.u3_7(q_device, wires=0)
        self.u3_8(q_device, wires=1)

        self.cu3_3(q_device, wires=[1, 2])

        self.u3_9(q_device, wires=1)
        self.u3_10(q_device, wires=2)


class QModelFour2(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 3
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)
        self.u3_6 = tq.U3(has_params=True, trainable=True)
        self.u3_7 = tq.U3(has_params=True, trainable=True)
        self.u3_8 = tq.U3(has_params=True, trainable=True)
        self.u3_9 = tq.U3(has_params=True, trainable=True)
        self.u3_10 = tq.U3(has_params=True, trainable=True)

        self.cu3_0 = tq.CU3(has_params=True, trainable=True)

        self.cu3_1 = tq.CU3(has_params=True, trainable=True)

        self.cu3_2 = tq.CU3(has_params=True, trainable=True)

        self.cu3_3 = tq.CU3(has_params=True, trainable=True)

    def forward(self, q_device: tq.QuantumDevice):
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.u3_2(q_device, wires=2)

        self.cu3_0(q_device, wires=[1, 2])

        self.u3_3(q_device, wires=0)
        self.u3_4(q_device, wires=1)

        self.cu3_1(q_device, wires=[0, 1])

        self.u3_5(q_device, wires=1)
        self.u3_6(q_device, wires=2)

        self.cu3_2(q_device, wires=[1, 2])

        self.u3_7(q_device, wires=0)
        self.u3_8(q_device, wires=1)

        self.cu3_3(q_device, wires=[0, 1])

        self.u3_9(q_device, wires=1)
        self.u3_10(q_device, wires=2)


def parse_text_and_plot_histogram(text):
    import matplotlib.pyplot as plt
    import numpy as np

    # Split the text into lines
    lines = text.split('\n')

    # Lists to store accuracies
    accuracies = []

    # Iterate over the lines and extract accuracies
    for i, line in enumerate(lines):
        if line.startswith('Step:'):
            continue  # We're not using steps for the histogram
        elif i % 3 == 2:  # The accuracies are on every third line starting with the second
            accuracies.append(float(line))

    # Convert accuracies to a numpy array for easier processing
    accuracies = np.array(accuracies)

    # Count occurrences of the three specified values using a tolerance
    tolerance = 0.001
    counts = {
        "~0.382": np.sum(np.abs(accuracies - 0.382) < tolerance),
        "~0.521": np.sum(np.abs(accuracies - 0.521) < tolerance),
        "~0.6123": np.sum(np.abs(accuracies - 0.6123) < tolerance)
    }

    print(counts)

    # Plotting
    plt.figure(figsize=(10, 6))
    counts_hist, bins, patches = plt.hist(accuracies, bins=50, color='skyblue', edgecolor='black', label='Occurrences')

    # Find the center of the bin to which the specific accuracies belong
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Annotation
    for key, count in counts.items():
        # Find the bin center for the accuracy we're annotating
        accuracy_value = float(key.strip('~'))
        bin_index = (np.abs(bin_centers - accuracy_value)).argmin()
        plt.annotate(f'{key}: {count}', xy=(bin_centers[bin_index], counts_hist[bin_index]),
                     xytext=(bin_centers[bin_index], counts_hist[bin_index] + 5),
                     arrowprops=dict(facecolor='red', shrink=0.05), ha='left')

    plt.title('Distribution of Training Accuracies')
    plt.xlabel('Accuracy', labelpad=15)
    plt.ylabel('Frequency', labelpad=15)
    plt.legend()  # This now references the labeled histogram
    plt.tight_layout()  # Ensure everything fits without being cut off
    plt.show()

    return accuracies


def hilbert_schmidt_distance(V: np.ndarray, U: np.ndarray):
    L = V.shape[0]
    Vdag = V.transpose()
    Vdag = Vdag.conjugate()
    return np.sqrt(1 - np.abs(np.abs(np.trace(np.matmul(Vdag, U))) ** 2) / (L ** 2))


def train(target_unitary, model, optimizer):
    result_unitary = model.get_unitary()

    # https://link.aps.org/accepted/10.1103/PhysRevA.95.042318 unitary fidelity according to table 1

    # compute the unitary infidelity
    loss = torch.sqrt(1 - torch.abs(torch.abs(torch.trace(target_unitary.T.conj() @ result_unitary)) ** 2) / (
            target_unitary.shape[0] ** 2))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    '''
    print(
        f"infidelity (loss): {loss.item()}, \n target unitary : "
        f"{target_unitary.detach().cpu().numpy()}, \n "
        f"result unitary : {result_unitary.detach().cpu().numpy()}\n"
    )
    '''
    return loss.item()


def main():
    f = open("C:/Users/yezhu/OneDrive/Desktop/torchquantum/output.txt", "a")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )

    parser.add_argument("--pdb", action="store_true", help="debug with pdb")

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    possible_control = [[0, 1], [1, 0], [1, 2], [2, 1]]
    step = 0
    minimum_loss = 1000000
    best_setup = []
    for q0control in possible_control:
        for q1control in possible_control:
            for q2control in possible_control:
                for q3control in possible_control:
                    for q4control in possible_control:

                        model = QModelFiveGeneral(q0control, q1control, q2control, q3control, q4control).to(device)

                        n_epochs = args.epochs
                        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
                        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

                        target_unitary = torch.tensor(
                            [
                                [1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 1, 0],
                            ]
                            , dtype=torch.complex64)

                        loss_list = []

                        for epoch in range(1, n_epochs + 1):
                            # print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
                            loss_list.append(train(target_unitary, model, optimizer))
                            scheduler.step()
                        print("Step:" + str(step))
                        print("Step:" + str(step), file=f)
                        print(q0control + q1control + q2control + q3control + q4control)
                        print(q0control + q1control + q2control + q3control + q4control, file=f)
                        final_loss = loss_list[-1]

                        if final_loss < minimum_loss:
                            minimum_loss = final_loss
                            best_setup = [q0control, q1control, q2control, q3control, q4control]
                        print(loss_list[-1])
                        print(loss_list[-1], file=f)
                        step += 1

    f.close()
    return


def read_file_and_plot_histogram(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Now use the previously defined function to parse and plot
    return parse_text_and_plot_histogram(file_content)


def plot_three_training_loss():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )

    parser.add_argument("--pdb", action="store_true", help="debug with pdb")

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    step = 0

    model = QModelFiveGeneral([0, 1], [1, 2], [0, 1], [1, 2], [0, 1]).to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    target_unitary = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
        , dtype=torch.complex64)

    loss_list1 = []

    for epoch in range(1, n_epochs + 1):
        # print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        loss_list1.append(train(target_unitary, model, optimizer))
        scheduler.step()
    print("Step:" + str(step))
    print("Step:" + str(step))

    plt.plot(loss_list1)

    model = QModelFiveGeneral([0, 1], [0, 1], [0, 1], [1, 2], [2, 1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_list2 = []

    for epoch in range(1, n_epochs + 1):
        # print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        loss_list2.append(train(target_unitary, model, optimizer))
        scheduler.step()
    print("Step:" + str(step))
    print("Step:" + str(step))

    plt.plot(loss_list2)

    model = QModelFiveGeneral([0, 1], [0, 1], [1, 0], [0, 1], [1, 0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_list3 = []

    for epoch in range(1, n_epochs + 1):
        # print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        loss_list3.append(train(target_unitary, model, optimizer))
        scheduler.step()
    print("Step:" + str(step))
    print("Step:" + str(step))

    plt.plot(loss_list3)

    plt.show()
    return


if __name__ == "__main__":
    # read_file_and_plot_histogram("C:/Users/yezhu/OneDrive/Desktop/torchquantum/output.txt")
    plot_three_training_loss()
