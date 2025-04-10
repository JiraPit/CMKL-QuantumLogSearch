"""
Utility function to create a quantum state.
"""

import numpy as np
from qiskit.quantum_info import Statevector


def create_state(num_qubits, state_spec="random"):
    """
    Creates a state.

    Args:
        num_qubits (int): Number of qubits (default 3)
        state_spec: State specification:
                    - "random" for a random state
                    - String of 0s and 1s (e.g., "101") for a specific basis state
                    - List/array of complex amplitudes (length 2^num_qubits)

    Returns:
        Statevector: The created quantum state
    """
    # Case 1: Random state
    if state_spec == "random":
        # Create a random complex vector
        dim = 2**num_qubits
        real_parts = np.random.normal(0, 1, dim)
        imag_parts = np.random.normal(0, 1, dim)
        vector = real_parts + 1j * imag_parts

        # Normalize the vector
        norm = np.linalg.norm(vector)
        vector = vector / norm

        return Statevector(vector)

    # Case 2: Bit string
    elif isinstance(state_spec, str) and all(bit in "01" for bit in state_spec):
        # Check length
        if len(state_spec) != num_qubits:
            raise ValueError(
                f"Bit string length ({len(state_spec)}) doesn't match qubit count ({num_qubits})"
            )

        # Convert bit string to integer
        index = int(state_spec, 2)

        # Create a basis state
        vector = np.zeros(2**num_qubits, dtype=complex)
        vector[index] = 1.0

        return Statevector(vector)

    # Case 3: Direct amplitude specification
    elif isinstance(state_spec, (list, np.ndarray)):
        # Check dimension
        if len(state_spec) != 2**num_qubits:
            raise ValueError(
                f"Amplitude vector length ({len(state_spec)}) doesn't match required dimension ({2**num_qubits})"
            )

        return Statevector(state_spec)

    else:
        raise ValueError("Invalid state specification")
