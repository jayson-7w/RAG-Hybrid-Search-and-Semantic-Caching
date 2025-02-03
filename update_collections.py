import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models, HnswConfigDiff
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set KEYS
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
DATAPATH = os.environ["DATAPATH"]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

client.update_collection(
    collection_name="knowledge_base_big",

    quantization_config=models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X64,
            always_ram=True,
    )
  )
)

client.update_collection(
    collection_name="qa_pairs_big",

    quantization_config=models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X64,
            always_ram=True,
    )
  )
)

client.update_collection(
    collection_name="knowledge_base_big",

    # Example: Optimized for Fast Performance
    hnsw_config=HnswConfigDiff(
        m=64,                         
        ef_construct=500,             
        full_scan_threshold=10000,   
        max_indexing_threads=0,      
        on_disk=True,                
        # payload_m=None              
    )

)

client.update_collection(
    collection_name="qa_pairs_big",

    # Example: Optimized for Fast Performance
    hnsw_config=HnswConfigDiff(
        m=64,                         
        ef_construct=500,             
        full_scan_threshold=10000,   
        max_indexing_threads=0,      
        on_disk=True,               
        # payload_m=None               
    )

)