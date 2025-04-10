"""
Random State Swap Test Demo

This demo generates two random quantum states and performs a swap test
to measure their similarity.
"""

# Import required modules
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from similarity_search.circuit_builder import QuantumCircuitBuilder


def create_random_state(num_qubits):
    """Create a truly random quantum state vector with 2^num_qubits dimensions."""
    # Generate random components with both positive and negative values
    real_parts = np.random.normal(0, 1, 2**num_qubits)
    imag_parts = np.random.normal(0, 1, 2**num_qubits)

    # Combine into complex vector
    vec = real_parts + 1j * imag_parts

    # Normalize the vector
    vec = vec / np.linalg.norm(vec)
    return Statevector(vec)


def main():
    """Main function to run the random state swap test demo."""
    print("Random State Swap Test Demo")
    print("==========================")

    # Define number of qubits for our states
    num_qubits = 8
    print(f"Using {num_qubits} qubits for quantum state representation")

    simulator = AerSimulator()
    shots = 1024

    # Create registers for swap test
    ancilla_reg = QuantumRegister(1, "ancilla")
    state1_reg = QuantumRegister(num_qubits, "state1")
    state2_reg = QuantumRegister(num_qubits, "state2")
    cr = ClassicalRegister(1, "result")

    # Run multiple tests with different random states
    print("\nRunning multiple tests with different random states...")
    num_tests = 5

    for i in range(num_tests):
        state_1 = create_random_state(num_qubits)
        state_2 = create_random_state(num_qubits)

        # Classical fidelity
        fidelity = np.abs(np.dot(state_1, np.conj(state_2))) ** 2

        # Create quantum circuit
        qc = QuantumCircuitBuilder.create_circuit([state1_reg, state2_reg])
        qc.initialize(state_1, state1_reg)
        qc.initialize(state_2, state2_reg)
        qc.add_register(ancilla_reg, cr)

        swap_test = QuantumCircuitBuilder.swap_test_circuit(
            state1_reg,
            state2_reg,
            ancilla_reg,
        )
        qc = qc.compose(swap_test)
        assert isinstance(qc, QuantumCircuit)

        # Measure the ancilla qubit to classical bit
        qc.measure(ancilla_reg[0], cr[0])

        # Run simulation
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        p_zero = counts.get("0", 0) / shots
        quantum_similarity = 2 * p_zero - 1

        print(f"\nTest {i+1}/{num_tests}:")
        print(f"Classical fidelity: {fidelity:.4f}")
        print(f"Quantum similarity: {quantum_similarity:.4f}")
        print(f"Measurement counts: {counts}")


if __name__ == "__main__":
    main()
