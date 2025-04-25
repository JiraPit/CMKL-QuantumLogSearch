# CMKL-QuantumLogSearch

A quantum computing-based log search system that leverages quantum algorithms for unsorted log data retrieval.

## Project Structure

```
CMKL-LogSearch/
- database[x]/             # Log database with [x] entries
- grover_module/           # Quantum search implementation
- hash_module/             # Classical hash generation
- qrom_module/             # Quantum ROM implementation
- main.py                  # Application entry point
- preprocess_data.py       # Data preparation utilities
```

## Module Integration

The system processes and searches log data through the following workflow:

1. `preprocess_data.py` reads raw log files from a specified database directory and generates hash representations using `hash_module`.
2. `hash_module` converts text strings into binary hash representations using SHA-3 256 cryptographic hash function.
3. `qrom_module` implements a Quantum Read-Only Memory that maps address qubits to value qubits using multi-controlled X gates.
4. `grover_module` implements Grover's search algorithm with an oracle that uses the QROM to mark states where the target value matches the search criteria.
5. `main.py` integrates these components, providing a command interface for searching, adding, and displaying log entries.

## Technical Details

### hash_module

Implements SHA-3 256 hashing algorithm to convert string inputs into fixed-length binary representations. The module encodes strings using UTF-8, applies the hash function, and converts bytes to binary strings with configurable truncation length.

### qrom_module

Implements Quantum Read-Only Memory using multi-controlled X gates to map binary address states to binary target states. The module applies controlled operations based on a dictionary mapping address strings to target strings, enabling quantum superposition-based data retrieval.

### grover_module

Implements Grover's quantum search algorithm with a QROM-based oracle. The module includes components for:

- Multi-controlled Z gate implementation
- Oracle construction using QROM operations and phase marking
- Diffusion operator for amplitude amplification
- Circuit construction with optimal iteration calculation based on search space size

### preprocess_data.py

Converts log data CSV files into hash-based representations through column-wise processing. The module reads from log_data.csv, applies hash functions to each value, and writes the results to hash_data.csv with configurable hash length.

### main.py

Provides a command-line interface with three primary functions:

- Search: Uses Grover's algorithm with QROM to find log entries that match search criteria
- Add: Inserts new log entries with automatic hash generation
- Show: Displays the most recent log entries in both original and hashed form

## Usage

```
# Preprocess the database
python preprocess_data.py --db <database_directory> --len <hash_length>

# Start the command-line interface
python main.py --db <database_directory> --len <hash_length>
```
