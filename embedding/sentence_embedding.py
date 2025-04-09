"""
Embedding sentences into semantec vecotors
"""

from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    def __init__(self, model):
        self.model = SentenceTransformer(model)

    def embed(self, sentences):
        """
        Embed a list of sentences into their respective embeddings.
        Args:
            sentences (list): List of sentences to be embedded.

        Returns:
            np.ndarray: Array of sentence embeddings.
        """
        return self.model.encode(sentences)


if __name__ == "__main__":
    test_sentences = [
        "Quantum Computing Cookbook",
        "How to bake a cake",
    ]
    se = SentenceEmbedder(model="all-MiniLM-L6-v2")
    embeddings = se.embed(test_sentences)
    print(
        f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
    )
    print(f"First embedding: {embeddings[0][:5]}...")
