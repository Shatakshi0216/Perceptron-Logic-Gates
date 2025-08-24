import numpy as np

# Input data (shared by all gates)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Target outputs for each logic gate
LOGIC_GATES = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR": np.array([0, 1, 1, 0])
}