import os
import time
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from extract_ingredients import create_ingredient_list

from embeddings import dense_vectors_list, sparse_vectors_list
from dotenv import load_dotenv


load_dotenv()

# Set KEYS
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
DATAPATH = os.environ["DATAPATH"]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Insert knowledge base data
def insert_recipes_from_dataframe(df, ingredient_list):
    required_columns = ['name', 'diet', 'prep_time', 'cook_time', 'flavor_profile', 'course', 'state', 'region', 'ingredients']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Some required columns are missing from the DataFrame.")
    
    ids = []
    dense_vectors = []
    sparse_vectors = []
    payloads = []

    print(f"Inserting {len(df)} recipes into the knowledge base...")

    start_time = time.time()  

    for idx, row in df.iterrows():
        try:
            ingredients = row["ingredients"].lower()
 
            content_vector = dense_vectors_list([ingredients])[0].tolist()  # Convert to list
            sparse_embeddings = sparse_vectors_list(ingredients)


            ids.append(idx)
            dense_vectors.append(content_vector)
            
            for sparse_embedding in sparse_embeddings:
                sparse_vectors.append(models.SparseVector(
                    indices=list(sparse_embedding.indices),  
                    values=list(sparse_embedding.values)  
                ))
                
            payloads.append({
                "name": row['name'],
                "diet": row['diet'],
                "prep_time": row['prep_time'],
                "cook_time": row['cook_time'],
                "flavor_profile": row['flavor_profile'],
                "course": row['course'],
                "state": row['state'],
                "region": row['region'],
                "ingredients": ingredients
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    if ids:  
  
        # Insert in batches
        client.upsert(
            collection_name="knowledge_base_big",
            points=models.Batch(
                ids=ids,
                vectors={
                    "dense": dense_vectors,
                    "sparse": sparse_vectors
                },
                payloads=payloads
            )
        )
        
    total_time = time.time() - start_time
    print(f"Total time for inserting recipes: {total_time:.2f} seconds.")


if __name__ == "__main__":
    
    df = pd.read_csv(DATAPATH)

    ingredient_list = create_ingredient_list(df)

    insert_recipes_from_dataframe(df, ingredient_list)