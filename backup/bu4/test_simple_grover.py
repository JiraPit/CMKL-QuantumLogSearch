import pennylane as qml
import numpy as np

def test_simple_grover():
    """
    Test a simple Grover search implementation
    This is a minimal implementation to test if the core concept works
    """
    print("Testing a simple Grover search implementation...")
    
    # Create a quantum device with 3 qubits:
    # - 2 qubits for index register (4 possible indices)
    # - 1 qubit for oracle result
    dev = qml.device("lightning.qubit", wires=3, shots=1000)
    
    @qml.qnode(dev)
    def circuit(marked_index, iterations=1):
        """
        Simple Grover search circuit that marks a specific index
        
        Args:
            marked_index: Index to mark (0-3)
            iterations: Number of Grover iterations
        """
        # Initialize index register to equal superposition
        for i in range(2):
            qml.Hadamard(wires=i)
        
        # Apply Grover iterations
        for _ in range(iterations):
            # Oracle - mark the specified index state
            # Convert marked_index to binary and apply phase flip
            oracle_func(marked_index, wires=[0, 1])
            
            # Diffusion operator
            # Apply H to all qubits
            for i in range(2):
                qml.Hadamard(wires=i)
            
            # Apply -I + 2|0⟩⟨0| reflection
            # First, apply X to all qubits
            for i in range(2):
                qml.PauliX(wires=i)
            
            # Apply Z on the first qubit, controlled by the second
            # If there's only 1 qubit, just apply Z
            if 2 > 1:
                qml.ctrl(qml.PauliZ, control=[0], control_values=[1])(wires=1)
            else:
                qml.PauliZ(wires=0)
            
            # Apply X to all qubits again
            for i in range(2):
                qml.PauliX(wires=i)
            
            # Apply H to all qubits again
            for i in range(2):
                qml.Hadamard(wires=i)
        
        # Sample from the index register
        return qml.sample(wires=range(2))
    
    def oracle_func(marked_idx, wires):
        """Mark the specified index with a phase flip"""
        # Convert marked_idx to binary
        binary = format(marked_idx, f'0{len(wires)}b')
        
        # Apply X gates where binary digit is 0
        for i, bit in enumerate(binary):
            if bit == '0':
                qml.PauliX(wires=wires[i])
        
        # Apply Z to the last qubit, controlled by all others
        if len(wires) > 1:
            controls = wires[:-1]
            target = wires[-1]
            qml.ctrl(qml.PauliZ, control=controls, control_values=[1] * len(controls))(target)
        else:
            qml.PauliZ(wires=wires[0])
        
        # Uncompute - apply X gates again
        for i, bit in enumerate(binary):
            if bit == '0':
                qml.PauliX(wires=wires[i])
    
    # Test with each possible index (0-3)
    for marked_idx in range(4):
        print(f"\nMarking index {marked_idx} (binary {format(marked_idx, '02b')})")
        
        # Try with 1 iteration
        results_1 = circuit(marked_idx, iterations=1)
        
        # Convert binary samples to integers
        int_results_1 = [int("".join(str(b) for b in result), 2) for result in results_1]
        
        # Count occurrences
        from collections import Counter
        counts_1 = Counter(int_results_1)
        
        # Display results
        print(f"Results after 1 iteration:")
        for idx, count in sorted(counts_1.items()):
            print(f"  Index {idx} (binary {format(idx, '02b')}): {count/1000:.4f}")
        
        # Check that the marked index has the highest probability after 1 iteration
        best_match_1 = max(counts_1.items(), key=lambda x: x[1])
        print(f"Best match after 1 iteration: Index {best_match_1[0]} with prob {best_match_1[1]/1000:.4f}")
        
        # Verify the best match after 1 iteration is the marked index
        assert best_match_1[0] == marked_idx, f"Expected {marked_idx} but got {best_match_1[0]}"
        print(f"Test with marked index {marked_idx} passed!")
    
    print("\nSimple Grover search implementation test passed!")

if __name__ == "__main__":
    print("=== Simple Grover Search Test ===")
    test_simple_grover()
    print("\nAll Grover tests passed successfully!")