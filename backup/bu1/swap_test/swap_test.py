import pennylane as qml
import numpy as np


class SwapTest:
    def __init__(self):
        """
        Initialize the Swap Test circuit.
        The Swap Test measures similarity between two quantum states.
        When two states are identical, the ancilla qubit will be measured as 0 with probability 1.
        When two states are orthogonal, the ancilla will be measured as 0 or 1 with equal probability.
        """
        pass

    def build_circuit(self, state1_wires, state2_wires, ancilla_wire):
        """
        Build a quantum circuit for the Swap Test

        Args:
            state1_wires: Wires holding the first quantum state
            state2_wires: Wires holding the second quantum state
            ancilla_wire: Wire for the ancilla qubit that will hold the result

        Note:
            - For the oracle function in Grover's algorithm, we want to mark states
              that are similar to our target state (omega).
            - When states are similar, the ancilla will be measured as 0 with high probability.
            - When states are different, the ancilla will be measured as 1 with higher probability.
        """
        # Apply Hadamard to ancilla
        qml.Hadamard(wires=ancilla_wire)

        # Apply controlled SWAP operations between corresponding qubits
        # The two quantum states must have the same number of qubits
        if len(state1_wires) != len(state2_wires):
            raise ValueError("Both quantum states must have the same number of qubits")

        for i in range(len(state1_wires)):
            qml.CSWAP(wires=[ancilla_wire, state1_wires[i], state2_wires[i]])

        # Apply Hadamard to ancilla again
        qml.Hadamard(wires=ancilla_wire)

