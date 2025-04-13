import pennylane as qml
import numpy as np
from ..qram.qram import QRAM
from ..swap_test.swap_test import SwapTest

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
        self.total_qubits = num_index_qubits + num_data_qubits + 1  # +1 for ancilla qubit
        
        # Create wire mappings
        self.index_wires = list(range(num_index_qubits))
        self.data_wires = list(range(num_index_qubits, num_index_qubits + num_data_qubits))
        self.omega_wires = list(range(num_index_qubits + num_data_qubits, 
                                     num_index_qubits + 2 * num_data_qubits))
        self.ancilla_wire = self.total_qubits - 1
        
        # Initialize QRAM and Swap Test
        self.qram = QRAM(num_index_qubits, num_data_qubits)
        self.swap_test = SwapTest()
        
        # Calculate optimal number of Grover iterations
        self.num_iterations = int(np.pi/4 * np.sqrt(2**num_index_qubits))
        
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
        Oracle for Grover's algorithm
        
        The oracle marks states that are similar to omega using the swap test:
        1. Performs QRAM lookup based on the index register
        2. Applies swap test between the retrieved data and omega
        3. Flips the phase of states where the swap test result is 0 (similar)
        """
        # Perform QRAM lookup to retrieve data based on index
        self.qram.build_lookup_circuit()
        
        # Perform swap test between retrieved data and omega
        self.swap_test.build_circuit(self.data_wires, self.omega_wires, self.ancilla_wire)
        
        # Oracle marks states where ancilla is |0⟩ (similar states)
        # Apply phase flip when ancilla is |0⟩
        qml.PauliZ(wires=self.ancilla_wire)
    
    def _diffusion_operator(self):
        """
        Diffusion operator (Grover's diffusion) to amplify marked states
        """
        # Apply Hadamard to index qubits
        for i in self.index_wires:
            qml.Hadamard(wires=i)
            
        # Apply phase flip to all states except |0⟩
        qml.PauliZ(wires=self.index_wires[0])
        qml.ctrl(qml.PauliZ, control=self.index_wires[0:-1])(wires=self.index_wires[-1])
        qml.PauliZ(wires=self.index_wires[0])
        
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
            
        # Apply Grover iterations
        for _ in range(self.num_iterations):
            # Oracle to mark similar states
            self._oracle()
            
            # Diffusion operator to amplify marked states
            self._diffusion_operator()
            
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
            dev = qml.device("lightning.qubit", wires=self.total_qubits, shots=num_shots)
            
        # Define the quantum circuit
        @qml.qnode(dev)
        def circuit():
            return self.build_circuit(database, omega, iterations)
        
        # Run the circuit
        results = circuit()
        
        # Convert binary samples to integers for easier interpretation
        int_results = [int("".join(str(b) for b in sample), 2) for sample in results]
        
        # Count occurrences of each result
        from collections import Counter
        counts = Counter(int_results)
        
        # Sort by count in descending order
        sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        return {idx: count/num_shots for idx, count in sorted_results}