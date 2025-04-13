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
    
    def _selective_state_preparation(self, target_idx, vector):
        """
        Prepare a specific state conditional on index register matching target_idx
        
        Args:
            target_idx: Binary string index to match
            vector: Vector to load when index matches
        """
        # Convert binary string to control values (0 or 1 for each index qubit)
        control_values = [int(bit) for bit in target_idx]
        
        # Create a state preparation function for the data register
        def prepare_state():
            self._amplitude_encoding(vector, self.data_wires)
        
        # Apply controlled state preparation
        # This loads the vector data only when index register equals target_idx
        qml.ctrl(prepare_state, control=self.index_wires, control_values=control_values)()
    
    def build_lookup_circuit(self):
        """
        Build a quantum circuit that performs data lookup based on index
        This circuit loads the data from the database based on the index register state
        """
        if self.database is None:
            raise ValueError("Database must be loaded before lookup")
            
        # Initialize data register to |0⟩
        # This ensures it's in a known state before applying selective operations
        for wire in self.data_wires:
            qml.RX(0, wires=wire)
        
        # For each index in the database, selectively prepare the corresponding state
        for idx, vector in self.database.items():
            self._selective_state_preparation(idx, vector)