import pennylane as qml
import numpy as np
from qram.qram import QRAM
from swap_test.swap_test import SwapTest

def test_integrated_similarity_search():
    """
    Test an integrated similarity search circuit that combines QRAM and swap test
    """
    print("Testing integrated similarity search...")
    
    # Create quantum device with sufficient qubits:
    # - 2 qubits for index
    # - 2 qubits for data
    # - 2 qubits for target vector
    # - 1 qubit for ancilla
    dev = qml.device("lightning.qubit", wires=7, shots=1000)
    
    # Define wire assignments
    index_wires = [0, 1]
    data_wires = [2, 3]
    target_wires = [4, 5]
    ancilla_wire = 6
    
    # Initialize QRAM
    qram = QRAM(num_index_qubits=2, num_data_qubits=2)
    
    # Initialize swap test
    swap_test = SwapTest()
    
    # Simple database with basis states
    database = {
        "00": [1.0, 0.0, 0.0, 0.0],  # |00⟩
        "01": [0.0, 1.0, 0.0, 0.0],  # |01⟩
        "10": [0.0, 0.0, 1.0, 0.0],  # |10⟩
        "11": [0.0, 0.0, 0.0, 1.0]   # |11⟩
    }
    
    # Load database
    qram.load_database(database)
    
    @qml.qnode(dev)
    def circuit(target_vector, index_to_test):
        """
        Circuit that:
        1. Selects a specific index from the QRAM
        2. Compares the data at that index with a target vector
        3. Measures the swap test result
        
        Args:
            target_vector: Vector to compare against
            index_to_test: Index to look up (binary string)
        """
        # Prepare index register with specified index
        for i, bit in enumerate(index_to_test):
            if bit == '1':
                qml.PauliX(wires=index_wires[i])
        
        # Prepare target register with target vector
        qml.AmplitudeEmbedding(target_vector, wires=target_wires, normalize=True)
        
        # Perform QRAM lookup
        qram.build_lookup_circuit()
        
        # Perform swap test between data and target
        swap_test.build_circuit(data_wires, target_wires, ancilla_wire)
        
        # Measure the ancilla (0 means similar, 1 means different)
        return qml.sample(wires=ancilla_wire)
    
    # Test with different target vectors
    test_cases = [
        ([1.0, 0.0, 0.0, 0.0], "00"),  # Same as index 00
        ([0.0, 1.0, 0.0, 0.0], "01"),  # Same as index 01
        ([0.0, 0.0, 1.0, 0.0], "10"),  # Same as index 10
        ([0.0, 0.0, 0.0, 1.0], "11"),  # Same as index 11
    ]
    
    for target_vector, matching_index in test_cases:
        print(f"\nTesting target vector {target_vector} which should match index {matching_index}")
        
        # Test for each possible index
        for index in ["00", "01", "10", "11"]:
            # Run circuit
            results = circuit(target_vector, index)
            
            # Count zeros (similar) and ones (different)
            zeros = np.count_nonzero(results == 0)
            ones = np.count_nonzero(results == 1)
            
            similarity = zeros / 1000
            print(f"  Index {index}: {similarity:.4f} similarity (zeros: {zeros}, ones: {ones})")
            
            # The similarity should be higher for the matching index
            if index == matching_index:
                assert similarity > 0.9, f"Expected high similarity for matching index {index}, got {similarity}"
            else:
                assert similarity < 0.6, f"Expected low similarity for non-matching index {index}, got {similarity}"
        
        print(f"Test with target vector {target_vector} passed!")
    
    print("\nIntegrated similarity search test passed!")

if __name__ == "__main__":
    print("=== Integrated Similarity Search Test ===")
    test_integrated_similarity_search()
    print("\nAll integrated tests passed successfully!")