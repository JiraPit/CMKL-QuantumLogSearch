import pennylane as qml
import numpy as np
from qram.qram import QRAM


def test_qram_initialization():
    """Test if QRAM can be correctly initialized"""
    print("Testing QRAM initialization...")

    num_index_qubits = 2
    num_data_qubits = 2

    qram = QRAM(num_index_qubits, num_data_qubits)

    print(
        f"QRAM initialized with {num_index_qubits} index qubits and {num_data_qubits} data qubits"
    )
    print(f"Index wires: {qram.index_wires}")
    print(f"Data wires: {qram.data_wires}")
    print("QRAM initialization test passed.")


def test_qram_database_loading():
    """Test if QRAM can correctly load database"""
    print("\nTesting QRAM database loading...")

    num_index_qubits = 2
    num_data_qubits = 2

    qram = QRAM(num_index_qubits, num_data_qubits)

    database = {
        "00": [0.3, 0.1, 0.1, 0.5],
        "01": [0.1, 0.2, 0.3, 0.4],
        "10": [0.5, 0.2, 0.1, 0.2],
        "11": [0.2, 0.3, 0.4, 0.1],
    }

    # Load database
    qram.load_database(database)

    print(f"Database loaded successfully with {len(qram.database)} entries")
    print("QRAM database loading test passed.")


def test_qram_lookup():
    """Test if QRAM can correctly lookup data"""
    print("\nTesting QRAM lookup functionality...")

    num_index_qubits = 2
    num_data_qubits = 2
    total_qubits = num_index_qubits + num_data_qubits

    qram = QRAM(num_index_qubits, num_data_qubits)

    # Simple database with orthogonal states for easy verification
    database = {
        "00": [1.0, 0.0, 0.0, 0.0],  # |00⟩
        "01": [0.0, 1.0, 0.0, 0.0],  # |01⟩
        "10": [0.0, 0.0, 1.0, 0.0],  # |10⟩
        "11": [0.0, 0.0, 0.0, 1.0],  # |11⟩
    }

    qram.load_database(database)

    # Test lookup for each index
    dev = qml.device("lightning.qubit", wires=total_qubits)

    @qml.qnode(dev)
    def circuit(index_bits):
        # Initialize index register to specific state
        for i, bit in enumerate(index_bits):
            if bit == 1:
                qml.PauliX(wires=i)

        # Perform QRAM lookup
        qram.build_lookup_circuit()

        # Measure data register
        return qml.probs(wires=qram.data_wires)

    # Test for |00⟩ index
    print("Testing lookup for index |00⟩...")
    probs_00 = circuit([0, 0])
    print(f"Probabilities: {probs_00}")
    expected_00 = np.zeros(2**num_data_qubits)
    expected_00[0] = 1.0  # |00⟩
    np.testing.assert_allclose(probs_00, expected_00, atol=1e-5)
    print("Lookup for |00⟩ passed.")

    # Test for |01⟩ index
    print("\nTesting lookup for index |01⟩...")
    probs_01 = circuit([0, 1])
    print(f"Probabilities: {probs_01}")
    expected_01 = np.zeros(2**num_data_qubits)
    expected_01[1] = 1.0  # |01⟩
    np.testing.assert_allclose(probs_01, expected_01, atol=1e-5)
    print("Lookup for |01⟩ passed.")

    # Test for |10⟩ index
    print("\nTesting lookup for index |10⟩...")
    probs_10 = circuit([1, 0])
    print(f"Probabilities: {probs_10}")
    expected_10 = np.zeros(2**num_data_qubits)
    expected_10[2] = 1.0  # |10⟩
    np.testing.assert_allclose(probs_10, expected_10, atol=1e-5)
    print("Lookup for |10⟩ passed.")

    # Test for |11⟩ index
    print("\nTesting lookup for index |11⟩...")
    probs_11 = circuit([1, 1])
    print(f"Probabilities: {probs_11}")
    expected_11 = np.zeros(2**num_data_qubits)
    expected_11[3] = 1.0  # |11⟩
    np.testing.assert_allclose(probs_11, expected_11, atol=1e-5)
    print("Lookup for |11⟩ passed.")

    print("QRAM lookup test passed for all indices.")


if __name__ == "__main__":
    print("=== QRAM Tests ===")
    test_qram_initialization()
    test_qram_database_loading()
    test_qram_lookup()
    print("\nAll QRAM tests passed successfully!")

