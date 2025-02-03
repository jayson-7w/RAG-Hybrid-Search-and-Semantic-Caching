import time
import os
from uuid import uuid4
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embeddings import dense_vectors_list, sparse_vectors_list
from extract_words import extract_words_from_query
from extract_ingredients import create_ingredient_list
from generate_response import generate_llm_response
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set KEYS
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
DATAPATH = os.environ["DATAPATH"]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Record the start time
start_time = time.time()

# Step 1: Encode the query
step_1_start = time.time()
question = """Can you explain me a recipe the includes the following ingredients: 
                Fish roe, pumpkin flowers, mustard oil, turmeric, tomato
            Please provide the steps to make the recipe, like how to cook all and the time needed.            
            """
            
query_vector = dense_vectors_list([question])[0]
print(f"Step 1 - Query encoding time: {time.time() - step_1_start:.2f} seconds")

# Step 2: Load the data
step_2_start = time.time()
df = pd.read_csv(DATAPATH)
print(f"Step 2 - Data loading time: {time.time() - step_2_start:.2f} seconds")

# Step 3: Create a list of all possible ingredients
step_3_start = time.time()
ingredient_list = create_ingredient_list(df)
print(f"Step 3 - Ingredient list creation time: {time.time() - step_3_start:.2f} seconds")

# Step 4: Extract words from the query
step_4_start = time.time()
query_words = extract_words_from_query(question)
extracted_ingredients = [word for word in query_words if word in ingredient_list]
sparse_embeddings = sparse_vectors_list(', '.join(extracted_ingredients))
print(f"Step 4 - Query word extraction time: {time.time() - step_4_start:.2f} seconds")

# Step 5: Search in qa_collection for a similar query
step_5_start = time.time()
try:
    cached_results = client.query_points(
        collection_name="qa_pairs",
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(
                    indices=[idx for embedding in sparse_embeddings for idx in embedding.indices],
                    values=[val for embedding in sparse_embeddings for val in embedding.values]),
                using="sparse",
                limit=255,
            ),
            models.Prefetch(
                query=query_vector,
                using="dense",
                score_threshold=0.9,
                limit=255,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )
        ),
        limit=100
    )
    print(f"Step 5 - QA collection search time: {time.time() - step_5_start:.2f} seconds")
except Exception as e:
    print(f"Error searching in qa_pairs: {e}")
    cached_results = []  # Set to empty if there's an error

# Step 6: Check for cached results
step_6_start = time.time()
if cached_results.points and cached_results.points[0].score >= 0.9:
    cached_answer = cached_results.points[0].payload['llm_answer']
    print("Cached Answer:\n", cached_answer)
else:
    print("No cached result found, performing a new search in the knowledge base.")
    print(f"Step 6 - Cached result check time: {time.time() - step_6_start:.2f} seconds")

    # Step 7: Perform a new search in knowledge_base
    step_7_start = time.time()
    try:
        results = client.query_points(
            collection_name="knowledge_base",
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=[idx for embedding in sparse_embeddings for idx in embedding.indices],
                        values=[val for embedding in sparse_embeddings for val in embedding.values]),
                    using="sparse",
                    limit=255,
                ),
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    score_threshold=0.9,
                    limit=255,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            limit=100
        )
        print(f"Step 7 - Knowledge base search time: {time.time() - step_7_start:.2f} seconds")
    except Exception as e:
        print(f"Error searching in knowledge_base: {e}")
        results = []  # Set to empty if there's an error

    # Step 8: Process results
    step_8_start = time.time()
    if results.points[0].score:
        context = results.points[0].payload
        response_text, num_generated_tokens = generate_llm_response(context, question)
        qa_payload = {
            "question": question,
            "recipe": [context],
            "llm_answer": response_text
        }
    else:
        context = ""
        response_text, num_generated_tokens = generate_llm_response(context, question)
        qa_payload = {
            "question": question,
            "recipe": ["No results from knowledge base"],
            "llm_answer": response_text
        }
    print(f"Step 8 - Result processing time: {time.time() - step_8_start:.2f} seconds")

    # Step 9: Upsert the question-answer pair
    step_9_start = time.time()
    try:
        id = str(uuid4())
        sparse_vectors = [models.SparseVector(
            indices=[idx for embedding in sparse_embeddings for idx in embedding.indices],
            values=[val for embedding in sparse_embeddings for val in embedding.values]
        )]
        client.upsert(
            collection_name="qa_pairs",
            points=models.Batch(
                ids=[id],
                vectors={"dense": [query_vector], "sparse": sparse_vectors},
                payloads=[qa_payload]
            )
        )
        print(f"Step 9 - QA pair upsert time: {time.time() - step_9_start:.2f} seconds")
    except Exception as e:
        print(f"Error caching the question-answer pair: {e}")

# Calculate and print the total execution time
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")
