from sentence_transformers import SentenceTransformer, util

# 1. Load a lightweight, high-performance model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define your statements
sentences = [
    "The exciting new effort to make computers think, machines with minds, in the full and literal sense.",	"""Think Human"""
    "The art of creating machines that perform functions that require intelligence when performed by people",	"""Act Human"""
    "The study of mental faculties (abilities) through the use of computation models",				"""Think Rational"""
    "Computational Intelligence is the study of the design of intelligent agents"				"""Act Rational"""
]

# 3. Compute vector embeddings
embeddings = model.encode(sentences)

# 4. Calculate cosine similarity matrix
similarity_matrix = util.cos_sim(embeddings, embeddings)

# Display pairwise similarity scores
print(
    f"Similarity (1 vs 2): {similarity_matrix[0][1]:.4f}"
)  # High (similar meaning)
print(
    f"Similarity (1 vs 3): {similarity_matrix[0][2]:.4f}"
)  # Low (unrelated topic)