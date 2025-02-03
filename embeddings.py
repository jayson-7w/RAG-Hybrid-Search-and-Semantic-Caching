from fastembed import SparseTextEmbedding, TextEmbedding

dense_embedding_model = TextEmbedding("BAAI/bge-base-en-v1.5")
sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

# Function to create dense vectors
def dense_vectors_list(text, model=dense_embedding_model):
    embeddings_generator = model.embed(text)
    
    embeddings_list = list(embeddings_generator)
    
    return embeddings_list

# Function to create sparse vectors
def sparse_vectors_list(text, model=sparse_embedding_model):
    
    sparse_embeddings_list = list(model.embed(text))
    
    return sparse_embeddings_list
