import pennylane as qml
import numpy as np
import sys
import argparse
from qram.qram import QRAM
from swap_test.swap_test import SwapTest
from grover.grover import GroverSearch

def main():
    """
    Quantum similarity search using Grover's algorithm with a QRAM and swap test
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quantum similarity search")
    parser.add_argument("--vector", type=float, nargs=4, 
                        help="Target vector to search for (4 space-separated values)")
    parser.add_argument("--iterations", type=int, default=1, choices=[1, 2, 3], 
                        help="Number of Grover iterations (1-3)")
    parser.add_argument("--shots", type=int, default=1000,
                        help="Number of measurement shots")
    
    args = parser.parse_args()
    
    # Example database with binary keys and 4D vectors
    database = {
        "00": [0.3, 0.1, 0.1, 0.5],
        "01": [0.1, 0.2, 0.3, 0.4],
        "10": [0.5, 0.2, 0.1, 0.2],
        "11": [0.2, 0.3, 0.4, 0.1]
    }
    
    print("Database:")
    for key, value in database.items():
        print(f"  {key}: {value}")
    
    # Setup the Grover search
    num_index_qubits = 2  # For 4 database entries (2^2)
    num_vector_dim = 4    # Dimension of the vectors
    
    # Get target vector from args or use default
    if args.vector:
        omega = args.vector
    else:
        # Default vector (same as entry "01")
        omega = [0.1, 0.2, 0.3, 0.4]
    
    # Normalize vector
    norm = np.linalg.norm(omega)
    omega = [v/norm for v in omega]
    
    print("\nTarget vector (omega):")
    print(f"  {[float(v) for v in omega]}")
    
    # Calculate required data qubits for amplitude encoding
    num_data_qubits = int(np.ceil(np.log2(num_vector_dim)))
    
    print(f"\nInitializing Grover search with {num_index_qubits} index qubits and {num_data_qubits} data qubits")
    
    # Create the Grover search instance
    grover = GroverSearch(num_index_qubits, num_data_qubits)
    
    # Run the search with specified parameters
    print(f"\nRunning Grover's search algorithm with {args.iterations} iteration(s) and {args.shots} shots...")
    results = grover.search(database, omega, num_shots=args.shots, iterations=args.iterations)
    
    print("\nResults (probability of measuring each index):")
    for idx, prob in results.items():
        binary_idx = format(idx, f'0{num_index_qubits}b')
        print(f"  Index {binary_idx} (matches vector {database[binary_idx]}): {prob:.4f}")
    
    # Find the most likely result
    best_match = max(results.items(), key=lambda x: x[1])
    best_idx = format(best_match[0], f'0{num_index_qubits}b')
    
    print(f"\nBest match: Index {best_idx} with probability {best_match[1]:.4f}")
    print(f"Vector at best match: {database[best_idx]}")
    print(f"Target vector (omega): {[float(v) for v in omega]}")
    
    # Calculate classical similarity for comparison
    print("\nClassical cosine similarities:")
    for key, vector in database.items():
        similarity = np.dot(vector, omega) / (np.linalg.norm(vector) * np.linalg.norm(omega))
        print(f"  Index {key} similarity: {similarity:.4f}")
    
if __name__ == "__main__":
    main()