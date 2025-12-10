from typing import List, Tuple
import os
from glob import glob

import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY, EMBED_MODEL

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def split_into_chunks(text: str, max_words: int = 200, overlap: int = 40) -> List[str]:
    """
    Split a long paragraph into smaller chunks based on word count.

    Args:
        text      : original paragraph string
        max_words : maximum number of words in each chunk
        overlap   : number of overlapping words between consecutive chunks

    Returns:
        A list of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= max_words:
        return [text.strip()]

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words).strip())
        # Slide the window forward with overlap to preserve context
        start = end - overlap

    return chunks


def load_corpus(path: str = "data") -> Tuple[List[str], List[str]]:
    """
    Load all .txt files inside a directory and split them into chunks.

    Processing steps:
    1. For each .txt file in `path`:
        - Read the full text.
        - Split into paragraphs by blank lines ("\n\n").
        - For each paragraph, further split into word-based chunks
          using `split_into_chunks`.
    2. Assign each final chunk a source id "filename#chunk_index".

    Returns:
        texts   : list of chunk strings
        sources : list of source identifiers, e.g., "eu_ai_act.txt#5"
    """
    texts: List[str] = []
    sources: List[str] = []

    if not os.path.isdir(path):
        raise ValueError(f"Corpus path must be a directory, got: {path}")

    txt_files = sorted(glob(os.path.join(path, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {path}")

    chunk_index = 0  # global chunk counter across all files

    for filepath in txt_files:
        filename = os.path.basename(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # First split file into paragraphs by blank lines
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # Then split each paragraph into smaller chunks if it is too long
        for para in paragraphs:
            para_chunks = split_into_chunks(para, max_words=200, overlap=40)
            for chunk in para_chunks:
                if not chunk:
                    continue
                texts.append(chunk)
                sources.append(f"{filename}#{chunk_index}")
                chunk_index += 1

    return texts, sources


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of texts using the OpenAI embeddings API,
    in batches to avoid hitting API limits.

    Args:
        texts      : list of text strings
        batch_size : maximum number of texts per API call

    Returns:
        A 2D numpy array of shape (num_texts, embedding_dim).
    """
    if not texts:
        raise ValueError("embed_texts() received an empty list of texts.")

    all_vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch = texts[start:end]

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        batch_vectors = [item.embedding for item in response.data]
        all_vectors.extend(batch_vectors)

    vectors = np.array(all_vectors, dtype="float32")
    return vectors


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and document vectors.

    Args:
        query_vec : numpy array of shape (d,)
        doc_vecs  : numpy array of shape (n, d)

    Returns:
        A 1D numpy array of similarity scores of shape (n,).
    """
    q = query_vec / np.linalg.norm(query_vec)
    d = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    sims = d @ q
    return sims


class SimpleRAG:
    """
    A minimal Retrieval-Augmented Generation retriever using OpenAI embeddings.
    """

    def __init__(self, corpus_path: str = "data"):
        """
        Initialize the retriever by loading and embedding the corpus.

        Args:
            corpus_path : directory containing multiple .txt files.
        """
        # Load text chunks and their source identifiers
        self.texts, self.sources = load_corpus(corpus_path)

        # Pre-compute embeddings for all chunks
        self.embeddings = embed_texts(self.texts)

    def retrieve(self, query: str, k: int = 3):
        """
        Retrieve top-k relevant chunks for a given query.

        Args:
            query : user query string
            k     : number of chunks to return

        Returns:
            A list of dictionaries with keys:
            - "text"       : retrieved chunk
            - "similarity" : cosine similarity score
            - "source"     : source identifier "filename#chunk_index"
        """
        # Embed the query (single string -> one vector)
        query_vec = embed_texts([query])[0]

        # Compute similarities with all document embeddings
        sims = cosine_similarity(query_vec, self.embeddings)

        # Get indices of top-k highest scores
        top_k_indices = np.argsort(-sims)[:k]

        results = []
        for idx in top_k_indices:
            results.append(
                {
                    "text": self.texts[idx],
                    "similarity": float(sims[idx]),
                    "source": self.sources[idx],
                }
            )

        return results