import pennylane as qml
import numpy as np
from qram.qram import QRAM
from swap_test.swap_test import SwapTest
from grover.grover import GroverSearch

def main():
    """
    Test the quantum similarity search using Grover's algorithm
    """
    # Example database with binary keys and 4D vectors
    database = {
        "00": [0.3, 0.1, 0.1, 0.5],
        "01": [0.1, 0.2, 0.3, 0.4],
        "10": [0.5, 0.2, 0.1, 0.2],
        "11": [0.2, 0.3, 0.4, 0.1]
    }
    
    # Target vector to search for
    omega = [0.1, 0.2, 0.3, 0.4]
    
    print("Database:")
    for key, value in database.items():
        print(f"  {key}: {value}")
    
    print("\nTarget vector (omega):")
    print(f"  {omega}")
    
    # Setup the Grover search
    num_index_qubits = 2  # For 4 database entries (2^2)
    num_vector_dim = 4    # Dimension of the vectors
    
    # Calculate required data qubits for amplitude encoding
    num_data_qubits = int(np.ceil(np.log2(num_vector_dim)))
    
    print(f"\nInitializing Grover search with {num_index_qubits} index qubits and {num_data_qubits} data qubits")
    
    # Create the Grover search instance
    grover = GroverSearch(num_index_qubits, num_data_qubits)
    
    # Run the search with default parameters
    print("\nRunning Grover's search algorithm...")
    results = grover.search(database, omega, num_shots=1000)
    
    print("\nResults (probability of measuring each index):")
    for idx, prob in results.items():
        binary_idx = format(idx, f'0{num_index_qubits}b')
        print(f"  Index {binary_idx} (matches vector {database[binary_idx]}): {prob:.4f}")
    
    # Find the most likely result
    best_match = max(results.items(), key=lambda x: x[1])
    best_idx = format(best_match[0], f'0{num_index_qubits}b')
    
    print(f"\nBest match: Index {best_idx} with probability {best_match[1]:.4f}")
    print(f"Vector at best match: {database[best_idx]}")
    print(f"Target vector (omega): {omega}")
    
if __name__ == "__main__":
    main()