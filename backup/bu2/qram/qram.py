import pennylane as qml
import numpy as np

class QRAM:
    def __init__(self, num_index_qubits, num_data_qubits):
        """
        Initialize QRAM with specific dimensions
        
        Args:
            num_index_qubits: Number of qubits for index addressing
            num_data_qubits: Number of qubits for data representation
        """
        self.num_index_qubits = num_index_qubits
        self.num_data_qubits = num_data_qubits
        self.total_qubits = num_index_qubits + num_data_qubits
        self.database = None
        self.index_wires = list(range(num_index_qubits))
        self.data_wires = list(range(num_index_qubits, self.total_qubits))
    
    def _int_to_binary(self, n, bits):
        """Convert integer to binary string with specified number of bits"""
        return format(n, f'0{bits}b')
    
    def load_database(self, database):
        """
        Load a classical database into QRAM
        
        Args:
            database: Dictionary mapping integer indices to data vectors
                     Example: {0: [0.3, 0.1, 0.1, 0.5], 1: [0.1, 0.2, 0.3, 0.4]}
        """
        # Convert integer keys to binary strings
        binary_database = {}
        for idx, vector in database.items():
            # If the key is already a binary string, use it directly
            if isinstance(idx, str) and all(bit in '01' for bit in idx):
                binary_key = idx
                if len(binary_key) != self.num_index_qubits:
                    raise ValueError(f"Binary key {binary_key} must have length {self.num_index_qubits}")
            else:
                # Convert integer to binary
                binary_key = self._int_to_binary(int(idx), self.num_index_qubits)
            
            binary_database[binary_key] = vector
        
        self.database = binary_database
        self.num_entries = len(binary_database)
        
        # Verify all vectors have the same dimension
        vector_lens = [len(vector) for vector in self.database.values()]
        if not all(l == vector_lens[0] for l in vector_lens):
            raise ValueError("All vectors in database must have the same dimension")
            
        # Normalize vectors for amplitude encoding
        for idx, vec in self.database.items():
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0):
                self.database[idx] = np.array(vec) / norm
                
        return self
    
    def _amplitude_encoding(self, vector, wires):
        """
        Encode a classical vector into qubit amplitudes
        
        Args:
            vector: Classical vector to encode
            wires: Qubits to use for encoding
        """
        # Normalize the vector if it's not already normalized
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1.0):
            vector = vector / norm
            
        # Use PennyLane's AmplitudeEmbedding operation
        qml.AmplitudeEmbedding(vector, wires=wires, normalize=True)
    
    def build_lookup_circuit(self):
        """
        Build a quantum circuit that performs data lookup based on index
        This circuit loads the data from the database based on the index register state
        """
        if self.database is None:
            raise ValueError("Database must be loaded before lookup")
            
        # For each possible index value in the database
        for binary_idx, vector in self.database.items():
            # Convert binary string to list of 0s and 1s for control values
            control_values = [int(bit) for bit in binary_idx]
            
            # Create a state preparation for this data entry
            index_state_ops = []
            for i, bit in enumerate(binary_idx):
                if bit == '0':
                    index_state_ops.append(qml.PauliX(wires=self.index_wires[i]))
            
            # Create a function that applies the amplitude encoding
            def load_data():
                self._amplitude_encoding(vector, self.data_wires)
            
            # Apply X gates for 0 bits to create control pattern
            for op in index_state_ops:
                op
                
            # Apply controlled operation to load data when index matches
            qml.ctrl(load_data, control=self.index_wires)()
            
            # Uncompute the X gates
            for op in reversed(index_state_ops):
                op