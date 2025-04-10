"""
Pre-computes embeddings for the article dataset and saves them to a pickle file.
This allows the main application to load pre-computed embeddings instead of
generating them each time.
"""

import os
import pickle
import pandas as pd
from pathlib import Path

from embedding.sentence_embedding import SentenceEmbedder


def precompute_embeddings():
    # Initialize the sentence embedder
    sentence_embedder = SentenceEmbedder(model="all-MiniLM-L6-v2")

    # Load article dataset
    dataset_path = Path(__file__).parent / "dataset" / "articles_database.csv"
    articles_df = pd.read_csv(dataset_path)

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "pre_computed"
    os.makedirs(output_dir, exist_ok=True)

    # Encode all entries to vectors
    print("Generating sentence embeddings for all articles...")
    texts = [
        f"{row['doc_full_name']} {row['doc_description']}"
        for _, row in articles_df.iterrows()
    ]
    embeddings = sentence_embedder.embed(texts)

    # Apply PCA to reduce dimensionality to 64 components
    print("Applying PCA to reduce dimensionality...")
    reduced_embeddings = sentence_embedder.apply_pca(embeddings, n_components=64)

    # Select only 30 entries
    print("Selecting subset of entries for quantum state preparation...")
    num_entries = min(30, len(reduced_embeddings))
    selected_indices = list(range(num_entries))

    # Create a filtered dataframe with just the selected entries
    filtered_df = articles_df.iloc[selected_indices].copy()

    # Save the embeddings and filtered dataframe
    output = {
        "embeddings": reduced_embeddings[:num_entries],
        "filtered_df": filtered_df,
        "vector_size": reduced_embeddings.shape[1],
    }

    output_path = output_dir / "pre_embedded_data.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Pre-computed embeddings saved to {output_path}")
    print(
        f"Selected {num_entries} articles with {reduced_embeddings.shape[1]} features each"
    )


if __name__ == "__main__":
    precompute_embeddings()

