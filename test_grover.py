from grover.grover import GroverSearch


def test_full_grover():
    """
    Test the full Grover search with our actual implementation
    """
    print("=== Testing Full Grover Similarity Search ===")

    # Create a simple test database with orthogonal vectors for clear testing
    database = {
        "00": [1.0, 0.0, 0.0, 0.0],  # |00⟩
        "01": [0.0, 1.0, 0.0, 0.0],  # |01⟩
        "10": [0.0, 0.0, 1.0, 0.0],  # |10⟩
        "11": [0.0, 0.0, 0.0, 1.0],  # |11⟩
    }

    # Setup the Grover search
    num_index_qubits = 2  # For 4 database entries (2^2)
    num_data_qubits = 2  # For orthogonal basis states

    # Create the Grover search instance
    grover = GroverSearch(num_index_qubits, num_data_qubits)

    # Test with different target vectors
    test_cases = [
        ([1.0, 0.0, 0.0, 0.0], "00"),  # Matches index 00
        ([0.0, 1.0, 0.0, 0.0], "01"),  # Matches index 01
        ([0.0, 0.0, 1.0, 0.0], "10"),  # Matches index 10
        ([0.0, 0.0, 0.0, 1.0], "11"),  # Matches index 11
    ]

    for target_vector, expected_index in test_cases:
        print(
            f"\nSearching for vector {target_vector} (should match index {expected_index})"
        )

        # Run the search with 1 iteration and 1000 shots
        results = grover.search(database, target_vector, num_shots=1000, iterations=1)

        # Display results
        print("Results:")
        for idx, prob in sorted(results.items()):
            binary_idx = format(idx, f"0{num_index_qubits}b")
            print(
                f"  Index {binary_idx} (matches vector {database[binary_idx]}): {prob:.4f}"
            )

        # Find the most likely result
        best_match = max(results.items(), key=lambda x: x[1])
        best_idx = format(best_match[0], f"0{num_index_qubits}b")

        print(f"Best match: Index {best_idx} with probability {best_match[1]:.4f}")

        # Verify the result is what we expected
        assert (
            best_idx == expected_index
        ), f"Expected {expected_index} but got {best_idx}"
        print(f"Test with target vector {target_vector} passed!")

    print("\nAll full Grover search tests passed successfully!")


if __name__ == "__main__":
    test_full_grover()

