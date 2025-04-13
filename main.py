"""
Quantum Recommender System

A recommender system that uses quantum computing for similarity search.
The system loads pre-computed article embeddings and performs similarity-based 
recommendations using Grover's algorithm.
"""

import pickle
import numpy as np
from pathlib import Path
from grover.grover import GroverSearch

SHOTS = 100


def main():
    """
    Main function that runs the interactive quantum recommender system
    """
    # Load pre-computed embeddings
    pre_computed_path = Path(__file__).parent / "pre_computed" / "pre_embedded_data.pkl"

    # Check if pre-computed embeddings exist
    if not pre_computed_path.exists():
        print("Pre-computed embeddings not found. Run embed.py first.")
        return

    # Load pre-computed embeddings
    print("Loading pre-computed embeddings...")
    with open(pre_computed_path, "rb") as f:
        pre_computed_data = pickle.load(f)

    # Extract data from the pickle file
    embeddings = pre_computed_data["embeddings"]
    filtered_df = pre_computed_data["filtered_df"]
    vector_size = pre_computed_data["vector_size"]

    # Calculate required data qubits for amplitude encoding
    num_data_qubits = int(np.ceil(np.log2(vector_size)))

    # Number of qubits needed for index addressing (number of articles)
    num_articles = len(embeddings)
    num_index_qubits = int(np.ceil(np.log2(num_articles)))

    print(f"Database contains {num_articles} articles with {vector_size} features each")
    print(f"Using {num_index_qubits} index qubits and {num_data_qubits} data qubits")

    # Create database dictionary mapping binary indices to embedding vectors
    database = {}
    for i in range(num_articles):
        binary_idx = format(i, f"0{num_index_qubits}b")
        database[binary_idx] = embeddings[i].tolist()

    # Create the Grover search instance
    grover = GroverSearch(num_index_qubits, num_data_qubits)

    # Calculate the number of iterations based on the number of articles
    iterations = int(np.floor(np.pi / 4 * np.sqrt(num_articles)))

    # Main interaction loop
    while True:
        try:
            # Get article index from user
            print("\nEnter an article index to start or 'q' to quit:")
            user_input = input().strip()

            if user_input.lower() == "q":
                break

            article_idx = int(user_input)

            # Check if the index is valid
            if article_idx < 0 or article_idx >= num_articles:
                raise ValueError(
                    f"Article index must be between 0 and {num_articles-1}"
                )

            # Display selected article
            article = filtered_df.iloc[article_idx]
            print(f"\nSelected Article [{article_idx}]:")
            print(f"Title: {article['doc_full_name']}")
            if "doc_description" in article:
                print(f"Description: {article['doc_description']}")

            # Get the target article's embedding as the search target
            omega = embeddings[article_idx].tolist()

            # Normalize the target vector
            norm = np.linalg.norm(omega)
            omega = [v / norm for v in omega]

            # Run the search with specified parameters
            print("\nFinding similar articles using Grover's algorithm...")
            print(f"Running with {iterations} iteration(s) and {SHOTS} shots...")

            results = grover.search(
                database,
                omega,
                num_shots=SHOTS,
                iterations=iterations,
            )

            # Find top similar articles (excluding the selected article itself)
            print("\nRecommended Articles:")
            similar_articles = []

            # Process and sort results
            for idx, prob in results.items():
                binary_idx = format(idx, f"0{num_index_qubits}b")
                result_idx = int(binary_idx, 2)

                # Skip if this is the same article or invalid index
                if result_idx == article_idx or result_idx >= num_articles:
                    continue

                similar_articles.append((result_idx, prob))

            # Sort by probability (descending) and take top 5
            similar_articles.sort(key=lambda x: x[1], reverse=True)
            top_similar = similar_articles[:5]

            # Display recommended articles
            for idx, similarity in top_similar:
                similar = filtered_df.iloc[idx]
                print(
                    f"[{idx}] {similar['doc_full_name']} (similarity: {similarity:.4f})"
                )

            # Also calculate classical similarity for comparison
            print("\nClassical similarity recommendations:")
            classical_similar = []

            for i in range(num_articles):
                if i == article_idx:
                    continue

                # Calculate cosine similarity
                vec1 = np.array(embeddings[article_idx])
                vec2 = np.array(embeddings[i])

                # Normalize vectors
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = vec2 / np.linalg.norm(vec2)

                similarity = np.dot(vec1, vec2)
                classical_similar.append((i, similarity))

            # Sort by similarity (descending) and take top 5
            classical_similar.sort(key=lambda x: x[1], reverse=True)
            top_classical = classical_similar[:5]

            # Display classical recommendations
            for idx, similarity in top_classical:
                similar = filtered_df.iloc[idx]
                print(
                    f"[{idx}] {similar['doc_full_name']} (similarity: {similarity:.4f})"
                )

        except ValueError as e:
            print(f"Error: {e}. Please enter a valid article index.")
        except IndexError:
            print(
                f"Error: Article index out of range. Dataset has {num_articles} articles."
            )
        except KeyboardInterrupt:
            break

    print("Exiting quantum recommender system.")


if __name__ == "__main__":
    main()
