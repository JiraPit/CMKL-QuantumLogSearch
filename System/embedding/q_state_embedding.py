from qiskit.quantum_info import Statevector
import numpy as np
import random


class QStateEmbedding:

    @staticmethod
    def amplitude_encode(vector):
        """
        Embed a classical vector into quantum amplitudes.
        Args:
            vector (list): A list of values to encode as amplitudes
        Returns:
            QuantumCircuit: Circuit that prepares the quantum state
        """
        # Normalize the vector (required for quantum state)
        vector = np.array(vector, dtype=complex)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Calculate required number of qubits
        num_qubits = int(np.ceil(np.log2(len(vector))))

        # Pad the vector with zeros to make it a power of 2
        state = np.zeros(2**num_qubits, dtype=complex)
        state[: len(vector)] = vector

        # Transform to a statevector
        state = Statevector(state)
        return state


# Example usage
if __name__ == "__main__":
    # Test with a list of length 4
    vector = [1 + 1j, 2 - 2j, 0, 3 + 0j]
    state = QStateEmbedding.amplitude_encode(vector)
    print(state)

    # Test with a list of length 10
    vector = [random.uniform(0, 50) for _ in range(10)]
    state = QStateEmbedding.amplitude_encode(vector)
    print(state)
