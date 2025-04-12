import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate


def dict_to_unitary(state_dict, n_qubits):
    # Initialize unitary matrix
    dim = 2**n_qubits
    U = np.zeros((dim, dim), dtype=complex)

    # Fill columns based on dictionary
    for x in range(dim):
        if x in state_dict:
            # Build the column vector from the dictionary entry
            column = np.zeros(dim, dtype=complex)
            for key, amp in state_dict[x].items():
                column[key] = amp
            U[:, x] = column
        else:
            # Default to identity for unspecified states (caution: may break unitarity)
            U[x, x] = 1

    # Verify unitarity (optional but recommended)
    op = Operator(U)
    if not op.is_unitary():
        raise ValueError("The matrix is not unitary. Revise your dictionary.")

    return U


# Define your dictionary (example: partial Hadamard-like behavior)
my_dict = {
    0: {0: 1 / np.sqrt(2), 1: 1 / np.sqrt(2)},  # |0> → (|0> + |1>)/√2
    1: {
        0: 1 / np.sqrt(2),
        1: -1 / np.sqrt(2),
    },  # |1> → (|0> - |1>)/√2 (to ensure unitarity)
}

# Example 2: More complex superposition example for 2 qubits
complex_dict = {
    0: {
        0: 1 / 2,
        1: 1 / 2,
        2: 1 / 2,
        3: 1 / 2,
    },  # |00⟩ → uniform superposition of all states
    1: {
        0: 1 / 2,
        1: -1 / 2,
        2: 1 / 2,
        3: -1 / 2,
    },  # |01⟩ → superposition with phase
    2: {
        0: 0,
        1: 0,
        2: 1 / np.sqrt(2),
        3: 1 / np.sqrt(2),
    },  # |10⟩ → 1/√2|10⟩ + 1/√2|11⟩
    3: {
        0: 1 / np.sqrt(2),
        1: np.sqrt(3) / 2,
        2: 0,
        3: 0,
    },  # |10⟩ → 1/√2|00⟩ + 1/√2|01⟩
}

# Example usage for 1 qubit
n_qubits = 2
custom_unitary = dict_to_unitary(complex_dict, n_qubits)
custom_gate = UnitaryGate(custom_unitary, label="DictGate")

# Test the gate in a circuit
qc = QuantumCircuit(1)
qc.append(custom_gate, [0])
print(qc.draw())

# Verify behavior (output should match my_dict)
print("\nUnitary matrix:\n", Operator(qc).data)
