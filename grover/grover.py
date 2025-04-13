import pennylane as qml
import numpy as np
from qram.qram import QRAM
from swap_test.swap_test import SwapTest
from collections import Counter


class GroverSearch:
    def __init__(self, num_index_qubits, num_data_qubits):
        """
        Initialize Grover's search algorithm for similarity search

        Args:
            num_index_qubits: Number of qubits for index addressing
            num_data_qubits: Number of qubits for data representation
        """
        self.num_index_qubits = num_index_qubits
        self.num_data_qubits = num_data_qubits

        # Calculate the total number of qubits needed
        # index qubits + data qubits + omega qubits + ancilla qubit
        self.total_qubits = num_index_qubits + (2 * num_data_qubits) + 1

        # Create distinct wire mappings with no overlaps
        wire_idx = 0

        # Index register
        self.index_wires = list(range(wire_idx, wire_idx + num_index_qubits))
        wire_idx += num_index_qubits

        # Data register
        self.data_wires = list(range(wire_idx, wire_idx + num_data_qubits))
        wire_idx += num_data_qubits

        # Omega register (separate from data register)
        self.omega_wires = list(range(wire_idx, wire_idx + num_data_qubits))
        wire_idx += num_data_qubits

        # Ancilla qubit (for swap test)
        self.ancilla_wire = wire_idx

        # Initialize QRAM and Swap Test
        self.qram = QRAM(num_index_qubits, num_data_qubits)
        self.swap_test = SwapTest()

        # Calculate optimal number of Grover iterations
        self.num_iterations = int(np.pi / 4 * np.sqrt(2**num_index_qubits))

    def _amplitude_encoding(self, vector, wires):
        """
        Encode a classical vector into qubit amplitudes

        Args:
            vector: Classical vector to encode
            wires: Qubits to use for encoding
        """
        # Normalize the vector
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1.0):
            vector = vector / norm

        # Use PennyLane's AmplitudeEmbedding
        qml.AmplitudeEmbedding(vector, wires=wires, normalize=True)

    def _initialize_omega(self, omega):
        """
        Initialize the target state omega

        Args:
            omega: Classical vector to encode as target state
        """
        self._amplitude_encoding(omega, self.omega_wires)

    def _initialize_index_register(self):
        """
        Initialize the index register to equal superposition
        """
        for i in self.index_wires:
            qml.Hadamard(wires=i)

    def _oracle(self):
        """
        Oracle for Grover's algorithm - marks states that are similar to omega

        This is a simplified oracle for demonstration purposes that directly
        marks the state that should match the target vector.
        In a real quantum system, this would use the ancilla bit from the swap test.
        """
        # Lookup data from QRAM based on index register
        self.qram.build_lookup_circuit()

        # Perform swap test to compare retrieved data with target omega
        self.swap_test.build_circuit(
            self.data_wires, self.omega_wires, self.ancilla_wire
        )

        # Mark states where the data matches the target (ancilla = |0⟩)
        # First apply X to the ancilla to convert |0⟩ to |1⟩
        qml.PauliX(wires=self.ancilla_wire)

        # Apply controlled-Z to index register when ancilla is |1⟩
        # This marks states where data is similar to the target
        for wire in self.index_wires:
            qml.ctrl(qml.PauliZ, control=self.ancilla_wire, control_values=1)(
                wires=wire
            )

        # Uncompute ancilla
        qml.PauliX(wires=self.ancilla_wire)
        self.swap_test.build_circuit(
            self.data_wires, self.omega_wires, self.ancilla_wire
        )

    def _grover_iteration(self):
        """
        Perform one iteration of Grover's algorithm
        """
        # Apply oracle to mark states similar to target
        self._oracle()

        # Step 4: Diffusion operator
        # Apply Hadamard to all index qubits
        for i in self.index_wires:
            qml.Hadamard(wires=i)

        # Apply Z to |0⟩ state
        for i in self.index_wires:
            qml.PauliX(wires=i)

        # Apply multi-controlled Z
        if len(self.index_wires) > 1:
            # Multi-controlled-Z with all controls=1
            qml.ctrl(
                qml.PauliZ,
                control=self.index_wires[:-1],
                control_values=[1] * len(self.index_wires[:-1]),
            )(self.index_wires[-1])
        else:
            # With just one qubit, simply apply Z
            qml.PauliZ(wires=self.index_wires[0])

        # Restore original states
        for i in self.index_wires:
            qml.PauliX(wires=i)

        # Apply Hadamard again
        for i in self.index_wires:
            qml.Hadamard(wires=i)

    def build_circuit(self, database, omega, iterations=None):
        """
        Build the complete Grover search circuit

        Args:
            database: Dictionary mapping indices to data vectors
            omega: Target vector to search for similar entries
            iterations: Optional number of Grover iterations (default is optimal)
        """
        # Load the database into QRAM
        self.qram.load_database(database)

        # Initialize omega
        self._initialize_omega(omega)

        # Initialize index register to equal superposition
        self._initialize_index_register()

        # Set number of iterations if provided
        if iterations is not None:
            self.num_iterations = iterations
        else:
            # Use just a single iteration for this small example
            self.num_iterations = 1

        # Apply Grover iterations
        for _ in range(self.num_iterations):
            self._grover_iteration()

        # Measure the index register
        return qml.sample(wires=self.index_wires)

    def search(self, database, omega, num_shots=1000, iterations=None, dev=None):
        """
        Perform Grover's search to find database entries similar to omega

        Args:
            database: Dictionary mapping indices to data vectors
            omega: Target vector to search for similar entries
            num_shots: Number of measurements to perform
            iterations: Optional number of Grover iterations (default is optimal)
            dev: Optional PennyLane device (default is lightning.qubit)

        Returns:
            Dictionary of results with indices and their counts
        """
        # If no device provided, use default
        if dev is None:
            dev = qml.device(
                "lightning.qubit", wires=self.total_qubits, shots=num_shots
            )

        # Define the quantum circuit
        @qml.qnode(dev)
        def circuit():
            return self.build_circuit(database, omega, iterations)

        # Run the circuit
        results = circuit()

        # Convert binary samples to integers for easier interpretation
        int_results = [int("".join(str(b) for b in sample), 2) for sample in results]

        counts = Counter(int_results)
        print(f"Counts: {counts}")
        print(f"Counted {len(int_results)} samples")

        return {idx: count / num_shots for idx, count in counts.items()}
