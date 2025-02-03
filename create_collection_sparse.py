import os
import time
from qdrant_client.http.models import Distance, VectorParams, HnswConfig
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

# Set KEYS
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
LEN_EMBEDDINGS = int(os.environ["LEN_EMBEDDINGS"])  # Convert to integer

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

## Define common configurations
# High speed
hnsw_config = HnswConfig(
    m=8,                          # Number of connections for each vector
    ef_construct=50,              # Smaller list size during index construction for faster indexing
    full_scan_threshold=10000,    # Threshold for full scan
    max_indexing_threads=0,       # Use more threads for faster indexing
    on_disk=False,                # Keep the index in memory for faster access
)

quantization_config=models.ScalarQuantization(
   scalar=models.ScalarQuantizationConfig(
       type=models.ScalarType.INT8,
           quantile=0.99,
           always_ram=True,
       ),
   )

sparse_vectors_config={
    "sparse": models.SparseVectorParams(
        index=models.SparseIndexParams(
            on_disk=False,  # Set to True if you want to keep the index on disk
        )
    )
}

# High accuracy
# hnsw_config = HnswConfig(
#     m=64,                              
#     ef_construct=500,
#     full_scan_threshold=10000,   
#     max_indexing_threads=0,       
#     on_disk=True,
# )


# Create Collections for Knowledge Base and Q/A Pairs
def create_collections():
    try:
        # Knowledge base collection
        start_time = time.time()
        client.create_collection(
            collection_name="knowledge_base",
            vectors_config={
                "dense": models.VectorParams(size=LEN_EMBEDDINGS, distance=Distance.COSINE)
                },
            hnsw_config=hnsw_config.model_dump(),
            quantization_config=quantization_config,
            sparse_vectors_config=sparse_vectors_config
        )
        
        knowledge_base_time = time.time() - start_time
        print(f"Knowledge base collection created successfully in {knowledge_base_time:.2f} seconds.")


        # Question-Answer collection (Caching)
        start_time = time.time()
        client.create_collection(
            collection_name="qa_pairs",
            vectors_config={
                "dense": models.VectorParams(size=LEN_EMBEDDINGS, distance=Distance.COSINE)
                },            
            hnsw_config=hnsw_config.model_dump(),
            quantization_config=quantization_config,
            sparse_vectors_config=sparse_vectors_config
        )

        qa_pairs_time = time.time() - start_time
        print(f"Q/A pairs collection created successfully in {qa_pairs_time:.2f} seconds.")

    except Exception as e:
        print(f"Error creating collections: {e}")

if __name__ == "__main__":
    create_collections()

