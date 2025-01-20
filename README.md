# Word Embeddings Translation and Locality-Sensitive Hashing

This notebook implements word embeddings translation between languages and document similarity search using Locality-Sensitive Hashing (LSH).

## Overview

The project includes:
- English to French word embeddings translation
- Implementation of gradient descent for embedding alignment
- Document embedding generation from tweets
- Locality-Sensitive Hashing implementation
- Approximate K-nearest neighbors search

## Requirements

- Python 3.x
- NumPy
- NLTK
- Pickle

## Data Files

- `en_embeddings.p`: English word embeddings
- `fr_embeddings.p`: French word embeddings
- `en-fr.train.txt`: English-French translation pairs for training
- `en-fr.test.txt`: English-French translation pairs for testing
- Twitter samples dataset from NLTK

## Main Functions

### Embedding Translation
```python
# Get aligned matrices for translation
def get_matrices(en_fr_data, en_embed, fr_embed):
    # Returns paired word embedding matrices

# Train translation matrix
def align_embeddings(X, Y, train_steps=100, learning_rate=0.01):
    # Returns transformation matrix R

# Compute loss for training
def compute_loss(X, Y, R):
    # Returns Frobenius norm loss

# Compute gradient for optimization
def compute_gradient(X, Y, R):
    # Returns gradient for R matrix
```

### Document Embeddings
```python
# Generate document embedding from text
def get_document_embedding(tweet, en_embeddings):
    # Returns document vector

# Process multiple documents
def get_document_vecs(all_docs, en_embeddings):
    # Returns document embedding matrix
```

### Locality-Sensitive Hashing
```python
# Generate hash value for vector
def hash_value_of_vector(v, planes):
    # Returns hash value

# Create hash tables
def make_hash_table(vecs, planes):
    # Returns hash tables and ID tables

# Find approximate nearest neighbors
def approximate_knn(doc_id, v, planes_l, k=1):
    # Returns nearest neighbor IDs
```

## Usage Example

```python
# Load embeddings
en_embeddings_subset = pickle.load(open('en_embeddings.p','rb'))
fr_embeddings_subset = pickle.load(open('fr_embeddings.p','rb'))

# Train translation matrix
X, Y = get_matrices(en_fr_train, en_embeddings_subset, fr_embeddings_subset)
R = align_embeddings(X, Y)

# Create document embeddings
tweet_embedding = get_document_embedding(custom_tweet, en_embeddings_subset)

# Find similar documents using LSH
nearest_neighbors = approximate_knn(doc_id, vector, planes_l)
```

## Key Features

### Word Translation
- Gradient descent optimization
- Loss computation using Frobenius norm
- Nearest neighbor search for translation
- Accuracy evaluation

### Document Similarity
- Document vector generation
- Multiple hash table creation
- Approximate nearest neighbor search
- Efficient similarity computation

## Implementation Details

- Number of planes: 10
- Number of universes: 25
- Embedding dimension: 300
- Uses cosine similarity for vector comparison
- Implements efficient document lookup using LSH
- Handles out-of-vocabulary words

## Model Evaluation

The system evaluates:
- Translation accuracy on test set
- Efficiency of LSH vs. exact nearest neighbor search
- Quality of document similarity matches
