import numpy as np
from grover.grover import GroverSearch

# Constants
SHOTS = 1000

# Generate random vectors and normalize them
np.random.seed(42)  # For reproducibility
DATABASE = {}
for i in range(16):
    # Convert to 4-bit binary string
    key = format(i, "04b")
    # Generate random components with both positive and negative values
    vec = np.random.normal(0, 1, 2**4)
    # Normalize the vector
    vec = vec / np.linalg.norm(vec)
    DATABASE[key] = vec


def main():
    print("Database:")
    for key, value in DATABASE.items():
        print(f"  {key}: {np.round(value, 1)}")

    # Setup the Grover search
    vector_dim = len(list(DATABASE.values())[0])
    num_index_qubits = int(np.ceil(np.log2(len(DATABASE))))
    num_data_qubits = int(np.ceil(np.log2(vector_dim)))

    # Calculate the number of iterations based on the database size
    num_entries = len(DATABASE)
    iterations = int(np.floor(np.pi / 4 * np.sqrt(num_entries)))

    print(f"\nUsing {num_index_qubits} index qubits and {num_data_qubits} data qubits")
    print(f"Using {SHOTS} measurement shots")

    # Ask for user input for the target vector
    print(
        "\nEnter your target vector (comma-separated values) or 'c' followed by a number to use a database entry:"
    )
    try:
        user_input = input(">> ")
        if user_input.startswith("c"):
            try:
                db_index = int(user_input[1:])
                binary_idx = format(db_index, "04b")
                if binary_idx not in DATABASE:
                    print(
                        f"Error: Database entry with index {db_index} ({binary_idx}) does not exist."
                    )
                    return
                omega = DATABASE[binary_idx]
                print(
                    f"Using database entry at index {db_index} ({binary_idx}) as omega."
                )
            except ValueError:
                print("Error: Please enter a valid database index after 'c'.")
                return
        else:
            omega = [float(x) for x in user_input.strip().split(",")]
            if len(omega) != 16:
                print(f"Error: Expected 16 values, but got {len(omega)}.")
                return
    except ValueError:
        print("Error: Please enter valid numeric values or a valid database index.")
        return

    # Normalize vector
    norm = np.linalg.norm(omega)
    omega = [v / norm for v in omega]

    print("\nTarget vector (omega):")
    print(f"  {[float(v) for v in omega]}")

    print(
        f"\nInitializing Grover search with {num_index_qubits} index qubits and {num_data_qubits} data qubits"
    )

    # Create the Grover search instance
    grover = GroverSearch(num_index_qubits, num_data_qubits)

    # Run the search with specified parameters
    print(
        f"\nRunning Grover's search algorithm with {iterations} iteration(s) and {SHOTS} shots..."
    )
    results = grover.search(
        DATABASE,
        omega,
        num_shots=SHOTS,
        iterations=iterations,
    )

    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("\nResults (probability of measuring each index):")
    for idx, prob in results:
        binary_idx = format(idx, f"0{num_index_qubits}b")
        print(f"  Index {binary_idx}: {prob:.4f}")

    # Find the most likely result
    best_match = max(results, key=lambda x: x[1])
    best_idx = format(best_match[0], f"0{num_index_qubits}b")

    print(f"\nBest match: Index {best_idx} with probability {best_match[1]:.4f}")
    print(f"Vector at best match: {DATABASE[best_idx]}")
    print(f"Target vector (omega): {[float(v) for v in omega]}")

    # Calculate classical similarity for comparison
    print("\nClassical cosine similarities:")
    similarities = {}
    for key, vector in DATABASE.items():
        # Normalize both vectors
        norm_db_vector = np.array(vector) / np.linalg.norm(vector)
        norm_omega = np.array(omega) / np.linalg.norm(omega)

        # Calculate cosine similarity
        similarity = np.dot(norm_db_vector, norm_omega)
        similarities[key] = similarity
        print(f"  Index {key} similarity: {similarity:.4f}")

    # Find the highest classical similarity
    best_classical = max(similarities.items(), key=lambda x: x[1])
    print(
        f"\nBest classical match: Index {best_classical[0]} with similarity {best_classical[1]:.4f}"
    )


if __name__ == "__main__":
    main()
