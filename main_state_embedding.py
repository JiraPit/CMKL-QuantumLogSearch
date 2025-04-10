"""
Demo of sentence embedding with PCA dimensionality reduction and quantum state encoding
"""

from embedding.sentence_embedding import SentenceEmbedder
from embedding.q_state_embedding import QStateEmbedding
import numpy as np

if __name__ == "__main__":
    # Example sentences
    sentences = [
        "Quantum Computing Cookbook",
        "How to bake a cake",
        "Neural networks and deep learning",
        "Python programming for beginners",
    ]

    # Initialize the sentence embedder
    sentence_embedder = SentenceEmbedder()

    # Step 1: Create sentence embeddings
    print("--- Step 1: Sentence Embedding ---")
    vectors = sentence_embedder.embed(sentences)
    print(f"Original embeddings shape: {vectors.shape}")

    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print(f"Original Embedding (first 5 values): {vectors[i][:5]}...")

    # Step 2: Apply PCA for dimensionality reduction
    print("\n--- Step 2: PCA Dimensionality Reduction ---")
    pca_components = 64
    reduced_vectors = sentence_embedder.apply_pca(vectors, n_components=pca_components)
    print(f"Reduced embeddings shape: {reduced_vectors.shape}")

    # Calculate number of qubits needed
    num_qubits = int(np.ceil(np.log2(reduced_vectors.shape[1])))
    print(f"Number of qubits required for quantum encoding: {num_qubits}")

    # Step 3: Quantum State Encoding
    print("\n--- Step 3: Quantum State Encoding ---")
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print(f"Reduced Embedding: {reduced_vectors[i]}")

        # Embed the sentence into quantum state
        state = QStateEmbedding.amplitude_encode(reduced_vectors[i])
        print(f"Quantum State: {state}")
