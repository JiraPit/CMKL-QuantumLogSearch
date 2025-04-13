import pennylane as qml
import numpy as np
from swap_test.swap_test import SwapTest


def test_swap_test_identical():
    """Test swap test with identical states - should result in |0⟩ with high probability"""
    print("Testing swap test with identical states...")

    # Create a quantum device with 5 qubits:
    # - 2 qubits for first state
    # - 2 qubits for second state
    # - 1 qubit for ancilla
    dev = qml.device("lightning.qubit", wires=5, shots=2000)

    # Initialize SwapTest
    swap_test = SwapTest()

    @qml.qnode(dev)
    def circuit():
        # Prepare two identical states |01⟩
        qml.PauliX(wires=1)
        qml.PauliX(wires=3)

        # Apply swap test
        swap_test.build_circuit(
            state1_wires=[0, 1], state2_wires=[2, 3], ancilla_wire=4
        )

        # Measure the ancilla
        return qml.sample(wires=4)

    # Run the circuit
    results = circuit()

    # Count occurrences of 0 and 1
    zeros = np.count_nonzero(results == 0)
    ones = np.count_nonzero(results == 1)

    print(f"Results: {zeros} zeros, {ones} ones")
    print(f"Probability of |0⟩: {zeros/2000:.4f}")

    # For identical states, we expect probability of |0⟩ to be close to 1
    assert (
        zeros / 2000 > 0.9
    ), "Expected probability of |0⟩ to be close to 1 for identical states"

    print("Swap test with identical states passed.")


def test_swap_test_orthogonal():
    """Test swap test with orthogonal states - should result in |0⟩ with ~0.5 probability"""
    print("\nTesting swap test with orthogonal states...")

    # Create a quantum device with 5 qubits
    dev = qml.device("lightning.qubit", wires=5, shots=2000)

    # Initialize SwapTest
    swap_test = SwapTest()

    @qml.qnode(dev)
    def circuit():
        # Prepare orthogonal states |00⟩ and |11⟩
        qml.PauliX(wires=2)
        qml.PauliX(wires=3)

        # Apply swap test
        swap_test.build_circuit(
            state1_wires=[0, 1], state2_wires=[2, 3], ancilla_wire=4
        )

        # Measure the ancilla
        return qml.sample(wires=4)

    # Run the circuit
    results = circuit()

    # Count occurrences of 0 and 1
    zeros = np.count_nonzero(results == 0)
    ones = np.count_nonzero(results == 1)

    print(f"Results: {zeros} zeros, {ones} ones")
    print(f"Probability of |0⟩: {zeros/2000:.4f}")

    # For orthogonal states, we expect probability of |0⟩ to be close to 0.5
    assert (
        0.4 < zeros / 2000 < 0.6
    ), "Expected probability of |0⟩ to be close to 0.5 for orthogonal states"

    print("Swap test with orthogonal states passed.")


def test_swap_test_partial_overlap():
    """Test swap test with partially overlapping states"""
    print("\nTesting swap test with partially overlapping states...")

    # Create a quantum device with 3 qubits:
    # - 1 qubit for first state
    # - 1 qubit for second state
    # - 1 qubit for ancilla
    dev = qml.device("lightning.qubit", wires=3, shots=2000)

    # Initialize SwapTest
    swap_test = SwapTest()

    @qml.qnode(dev)
    def circuit(theta):
        # Prepare state |0⟩ for first qubit
        # Prepare state cos(theta)|0⟩ + sin(theta)|1⟩ for second qubit
        qml.RY(theta, wires=1)

        # Apply swap test
        swap_test.build_circuit(state1_wires=[0], state2_wires=[1], ancilla_wire=2)

        # Measure the ancilla
        return qml.sample(wires=2)

    # Test with various angles
    angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]

    for theta in angles:
        # Calculate expected overlap
        overlap = np.cos(theta / 2) ** 2
        expected_prob_0 = 0.5 * (1 + overlap**2)

        # Run circuit
        results = circuit(theta)

        # Count zeros
        zeros = np.count_nonzero(results == 0)
        prob_0 = zeros / 2000

        print(f"Angle: {theta:.4f}, Overlap: {overlap:.4f}")
        print(f"Expected probability of |0⟩: {expected_prob_0:.4f}")
        print(f"Measured probability of |0⟩: {prob_0:.4f}")

        # Check if measurement is within acceptable range
        assert (
            abs(prob_0 - expected_prob_0) < 0.2
        ), f"Probability {prob_0} too far from expected {expected_prob_0}"

        print(f"Test with angle {theta:.4f} passed.")

    print("Swap test with partially overlapping states passed.")


if __name__ == "__main__":
    print("=== Swap Test Tests ===")
    test_swap_test_identical()
    test_swap_test_orthogonal()
    test_swap_test_partial_overlap()
    print("\nAll swap test tests passed successfully!")
