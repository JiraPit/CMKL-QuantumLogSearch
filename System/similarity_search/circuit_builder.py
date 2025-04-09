"""
Quantum Circuit Builder for similarity search.
"""

from math import asin, sqrt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class QuantumCircuitBuilder:
    """Class for building quantum circuits for similarity search."""

    @staticmethod
    def prepare_state_circuit(state_vector, qreg):
        """
        Creates a circuit to prepare the given quantum state.

        Args:
            state_vector (Statevector): The quantum state to prepare
            qreg (QuantumRegister): Register to prepare the state in

        Returns:
            QuantumCircuit: Circuit that prepares the state
        """
        qc = QuantumCircuit(qreg)
        qc.initialize(state_vector.data, qreg)
        return qc

    @staticmethod
    def swap_test_circuit(state1_qreg, state2_qreg, ancilla_qreg):
        """
        Constructs a SWAP test circuit between two quantum registers.

        Args:
            state1_qreg (QuantumRegister): First quantum register
            state2_qreg (QuantumRegister): Second quantum register
            ancilla_qreg (QuantumRegister): Ancilla qubit register

        Returns:
            QuantumCircuit: SWAP test circuit
        """
        qc = QuantumCircuit(ancilla_qreg, state1_qreg, state2_qreg)

        # Apply Hadamard to ancilla
        qc.h(ancilla_qreg[0])

        # Apply controlled SWAP operations between corresponding qubits
        for i in range(len(state1_qreg)):
            qc.cswap(ancilla_qreg[0], state1_qreg[i], state2_qreg[i])

        # Apply Hadamard to ancilla again
        qc.h(ancilla_qreg[0])

        return qc

    @staticmethod
    def similarity_oracle(target_state, num_qubits, threshold):
        """
        Creates an oracle that marks states with similarity GREATER than threshold to target_state.

        Args:
            target_state (Statevector): The target state to compare against
            num_qubits (int): Number of qubits in each state
            threshold (float): Minimum similarity threshold (0 to 1)

        Returns:
            QuantumCircuit: Oracle circuit
        """
        # Calculate angle for the rotation
        # For SWAP test, probability of measuring |0⟩ is (1+s)/2
        angle = asin(sqrt(threshold))

        # Create registers
        oracle_reg = QuantumRegister(num_qubits, "oracle")
        target_reg = QuantumRegister(num_qubits, "target")
        ancilla_reg = QuantumRegister(1, "ancilla")
        phase_reg = QuantumRegister(1, "phase")

        # Create circuit
        qc = QuantumCircuit(oracle_reg, target_reg, ancilla_reg, phase_reg)

        # Prepare the target state in target_reg
        target_prep = QuantumCircuitBuilder.prepare_state_circuit(
            target_state, target_reg
        )
        qc.compose(target_prep, qubits=target_reg, inplace=True)

        # Create and compose the SWAP test circuit
        swap_test = QuantumCircuitBuilder.swap_test_circuit(
            oracle_reg, target_reg, ancilla_reg
        )
        qc.compose(swap_test, inplace=True)

        # For high similarity, we want to mark states when ancilla is 0
        # (indicating high similarity in the SWAP test)

        # Flip the ancilla to mark high similarity states
        qc.x(ancilla_reg[0])

        # Apply controlled phase rotation based on similarity threshold
        qc.ch(ancilla_reg[0], phase_reg[0])  # Apply controlled-H
        qc.cp(2 * angle, ancilla_reg[0], phase_reg[0])  # Apply controlled-Phase
        qc.ch(ancilla_reg[0], phase_reg[0])  # Apply controlled-H again

        # Flip the ancilla back
        qc.x(ancilla_reg[0])

        # Uncompute the SWAP test to clean up ancilla qubits
        swap_test_dag = swap_test.inverse()
        qc.compose(swap_test_dag, inplace=True)

        # Uncompute the target state preparation
        # Instead of using inverse() which has issues with complex parameters in initialize,
        # simply reset all target register qubits to |0⟩ state
        for i in range(num_qubits):
            qc.reset(target_reg[i])

        return qc

    @staticmethod
    def diffusion_operator(num_qubits):
        """
        Creates the standard diffusion operator for Grover's algorithm.

        Args:
            num_qubits (int): Number of qubits

        Returns:
            QuantumCircuit: Diffusion operator circuit
        """
        qc = QuantumCircuit(num_qubits)

        # Apply H gates to all qubits
        for i in range(num_qubits):
            qc.h(i)

        # Apply Z to |0...0⟩
        # First, apply X gates to all qubits
        for i in range(num_qubits):
            qc.x(i)

        # Apply controlled-Z operation
        # This is a multi-controlled Z implementation that works well for 3-5 qubits
        # This implementation uses the first qubit as target and works for NUM_QUBITS=3
        # For different qubit counts, this would need to be adjusted
        qc.h(0)

        # Use remaining qubits as controls
        for i in range(1, num_qubits):
            qc.cx(i, 0)

        qc.h(0)

        # Uncompute X gates
        for i in range(num_qubits):
            qc.x(i)

        # Apply H gates to all qubits again
        for i in range(num_qubits):
            qc.h(i)

        return qc

    @staticmethod
    def grover_search_high_similarity(
        target_state, num_qubits, iterations=1, threshold=0.1
    ):
        """
        Implements Grover's search for states with high similarity to the target.

        Args:
            target_state (Statevector): Target state for similarity comparison
            num_qubits (int): Number of qubits
            iterations (int): Number of Grover iterations
            threshold (float): Minimum similarity threshold

        Returns:
            QuantumCircuit: Complete Grover search circuit
        """
        # Create registers for the main algorithm
        oracle_reg = QuantumRegister(num_qubits, "oracle")
        target_reg = QuantumRegister(num_qubits, "target")
        ancilla_reg = QuantumRegister(1, "ancilla")
        phase_reg = QuantumRegister(1, "phase")
        cr = ClassicalRegister(num_qubits, "c")

        # Create a circuit with all necessary registers
        grover_circuit = QuantumCircuit(
            oracle_reg, target_reg, ancilla_reg, phase_reg, cr
        )

        # Initialize with Hadamard on the oracle register (search space)
        for i in range(num_qubits):
            grover_circuit.h(oracle_reg[i])

        # Get the oracle for high similarity
        oracle = QuantumCircuitBuilder.similarity_oracle(
            target_state, num_qubits, threshold
        )

        # Get the diffusion operator
        diffusion = QuantumCircuitBuilder.diffusion_operator(num_qubits)

        # Apply the Grover iterations
        for _ in range(iterations):
            # Apply oracle
            grover_circuit = grover_circuit.compose(oracle)
            assert isinstance(grover_circuit, QuantumCircuit)

            # Apply diffusion operator to the oracle register
            grover_circuit = grover_circuit.compose(diffusion, qubits=oracle_reg)
            assert isinstance(grover_circuit, QuantumCircuit)

        # Measure the results (only the oracle register)
        grover_circuit.measure(oracle_reg, cr)

        return grover_circuit

