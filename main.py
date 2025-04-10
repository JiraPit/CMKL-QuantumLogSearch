"""
Quantum Recommender System

A recommender system that uses quantum computing for similarity search.
The system embeds article text as quantum states and performs similarity-based recommendations.
"""

# Import required modules
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from embedding.sentence_embedding import SentenceEmbedder
from embedding.q_state_embedding import QStateEmbedding
from similarity_search.simulator import GroverSimulationManager


class QuantumRecommender:
    """Main class for quantum article recommendation system."""

    def __init__(self, embedding_model="all-MiniLM-L6-v2", threshold=0.1, shots=1024):
        """
        Initialize the quantum recommender system.

        Args:
            embedding_model (str): Name of the sentence embedding model
            threshold (float): Similarity threshold for search
            shots (int): Number of simulation shots
        """
        self.threshold = threshold
        self.shots = shots

        # Initialize embedders
        self.sentence_embedder = SentenceEmbedder(model=embedding_model)

        # Load and prepare data
        self.articles_df = None
        self.article_states = None
        self.num_qubits = None

    def load_dataset(self, dataset_path):
        """
        Load the article dataset.

        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.articles_df = pd.read_csv(dataset_path)

    def prepare_states(self, pca_components):
        """
        Prepare quantum states for all articles in the dataset.

        Args:
            pca_components (int): Number of components to keep after PCA
        """
        if self.articles_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")

        # Combine full name and description for embedding
        texts = [
            f"{row['doc_full_name']} {row['doc_description']}"
            for _, row in self.articles_df.iterrows()
        ]

        # Create sentence embeddings
        embeddings = self.sentence_embedder.embed(texts)

        # Apply PCA to reduce dimensionality
        reduced_embeddings = self.sentence_embedder.apply_pca(
            embeddings, n_components=pca_components
        )

        # Determine number of qubits needed (based on reduced embedding dimension)
        vector_size = reduced_embeddings.shape[1]
        self.num_qubits = int(np.ceil(np.log2(vector_size)))
        print(f"Using {self.num_qubits} qubits for quantum state representation")

        # Create quantum states for each embedding and store with original index
        self.article_states = {}
        for i, embedding in enumerate(reduced_embeddings):
            state = QStateEmbedding.amplitude_encode(embedding)
            self.article_states[i] = state

        return self.article_states

    def find_similar_articles(self, article_idx, max_results=5):
        """
        Find articles similar to the given article using quantum similarity search.

        Args:
            article_idx (int): Index of the article to find similarities for
            max_results (int): Maximum number of results to return

        Returns:
            list: List of (idx, similarity) tuples sorted by similarity
        """
        if self.article_states is None or len(self.article_states) == 0:
            raise ValueError("Article states not prepared. Call prepare_states first.")

        if article_idx not in self.article_states:
            raise ValueError(f"Article index {article_idx} not found in the dataset.")

        # Get the reference state and create reference tuple
        reference_state = self.article_states[article_idx]
        reference = (article_idx, reference_state)

        # Use the GroverSimulationManager to find similar states
        similar_articles = GroverSimulationManager.find_similar_states(
            reference=reference,
            states_dict=self.article_states,
            num_qubits=self.num_qubits,
            threshold=self.threshold,
            max_results=max_results,
            shots=self.shots,
        )

        return similar_articles


def main():
    # Initialize the recommender
    recommender = QuantumRecommender()

    # Check if pre-computed embeddings exist
    pre_computed_path = Path(__file__).parent / "pre_computed" / "pre_embedded_data.pkl"

    if not pre_computed_path.exists():
        print("Pre-computed embeddings not found. Run pre_embed.py first.")
        return

    # Load pre-computed embeddings
    print("Loading pre-computed embeddings...")
    with open(pre_computed_path, "rb") as f:
        pre_computed_data = pickle.load(f)

    # Extract data from the pickle file
    reduced_embeddings = pre_computed_data["embeddings"]
    recommender.articles_df = pre_computed_data["filtered_df"]
    vector_size = pre_computed_data["vector_size"]

    # Determine number of qubits needed
    recommender.num_qubits = int(np.ceil(np.log2(vector_size)))
    print(f"Using {recommender.num_qubits} qubits for quantum state representation")

    # Get number of entries
    num_entries = len(recommender.articles_df)
    selected_indices = list(range(num_entries))

    # Create quantum states for the pre-computed embeddings
    print(f"Preparing quantum states for {num_entries} articles...")
    recommender.article_states = {}
    for i in selected_indices:
        state = QStateEmbedding.amplitude_encode(reduced_embeddings[i])
        recommender.article_states[i] = state

    print(f"Prepared states for {len(recommender.article_states)} articles")

    # Main interaction loop
    while True:
        try:
            # Get article index from user
            print("\nEnter an article index to start (0-29) or 'q' to quit:")
            user_input = input().strip()

            if user_input.lower() == "q":
                break

            article_idx = int(user_input)

            # Check if the index is valid for our subset
            if article_idx not in recommender.article_states:
                raise ValueError(f"Article index must be between 0 and {num_entries-1}")

            # Display selected article
            article = recommender.articles_df.iloc[article_idx]
            print(f"\nSelected Article [{article_idx}]:")
            print(f"Title: {article['doc_full_name']}")
            print(f"Description: {article['doc_description']}")

            # Find similar articles
            print("\nRecommended Articles:")
            similar_articles = recommender.find_similar_articles(article_idx)

            for idx, similarity in similar_articles:
                similar = recommender.articles_df.iloc[idx]
                print(
                    f"[{idx}] {similar['doc_full_name']} (similarity: {similarity:.4f})"
                )

        except ValueError as e:
            print(f"Error: {e}. Please enter a valid article index.")
        except IndexError:
            print(
                f"Error: Article index out of range. Dataset has {num_entries} articles."
            )
        except KeyboardInterrupt:
            break

    print("Exiting quantum recommender system.")


if __name__ == "__main__":
    main()
