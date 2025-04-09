"""
Example usage of the Quantum Recommender system
"""

from main import QuantumRecommender, DEFAULT_CONFIG


def custom_config_example():
    """
    Example of using a custom configuration
    """
    # Create a custom configuration
    custom_config = DEFAULT_CONFIG.copy()
    custom_config["NUM_QUBITS"] = 3
    custom_config["THRESHOLD"] = 0.2
    custom_config["SHOTS"] = 512

    # Initialize the recommender with custom config
    recommender = QuantumRecommender(config=custom_config)

    # Run a similarity search
    results = recommender.run_high_similarity_search()

    # Print results
    print("\nCustom Configuration Results:")
    print(f"Number of qubits: {recommender.config['NUM_QUBITS']}")
    print(f"Threshold: {recommender.config['THRESHOLD']}")
    print(f"Total shots: {results['total_shots']}")
    print(f"Success rate: {results['success_rate']*100:.2f}%")


def sentence_embedding_example():
    """
    Example of using the sentence embedding functionality
    """
    # Initialize with default config
    recommender = QuantumRecommender()

    # Test sentences
    sentences = [
        "Quantum computing uses quantum bits",
        "Classical computing uses binary bits",
        "Machine learning models can make predictions",
        "Quantum machine learning combines quantum computing with ML",
    ]

    # Get embeddings
    embeddings = recommender.embed_sentences(sentences)

    # Print information about embeddings
    print("\nSentence Embedding Results:")
    print(f"Model used: {recommender.config['EMBEDDING_MODEL']}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Sample embedding: {embeddings[0][:5]}...")


if __name__ == "__main__":
    print("Quantum Recommender System Examples")
    print("==================================")

    # Run examples
    custom_config_example()
    sentence_embedding_example()

