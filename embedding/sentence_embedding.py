"""
Embedding sentences into semantic vectors with dimensionality reduction
"""

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class SentenceEmbedder:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.pca = None

    def embed(self, sentences):
        """
        Embed a list of sentences into their respective embeddings.
        Args:
            sentences (list): List of sentences to be embedded.

        Returns:
            np.ndarray: Array of sentence embeddings.
        """
        return self.model.encode(sentences)

    def apply_pca(self, embeddings, n_components):
        """
        Apply PCA dimensionality reduction to a set of embeddings.

        Args:
            embeddings (np.ndarray): Array of embeddings to reduce
            n_components (int): Number of components to keep

        Returns:
            np.ndarray: Reduced embeddings
        """
        # Initialize PCA if not already done
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            # Fit PCA on the data
            self.pca.fit(embeddings)

        # Transform the data
        reduced_embeddings = self.pca.transform(embeddings)
        print(
            f"Reduced embedding dimensions from {embeddings.shape[1]} to {reduced_embeddings.shape[1]}"
        )
        return reduced_embeddings


if __name__ == "__main__":
    test_sentences = [
        "Quantum Computing Cookbook",
        "How to bake a cake",
    ]
    se = SentenceEmbedder()
    embeddings = se.embed(test_sentences)
    print(
        f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
    )
    print(f"First embedding: {embeddings[0][:5]}...")

    # Test PCA reduction
    reduced = se.apply_pca(embeddings, n_components=16)
    print(f"Reduced embedding: {reduced[0]}")

