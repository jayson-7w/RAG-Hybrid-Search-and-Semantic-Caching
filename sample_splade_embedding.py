from fastembed import SparseTextEmbedding, SparseEmbedding
from typing import List
import json
import json
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained(SparseTextEmbedding.list_supported_models()[0]["sources"]["hf"])

def encode_text(text):

    model_name = "prithivida/Splade_PP_en_v1"
    # This triggers the model download
    model = SparseTextEmbedding(model_name=model_name)

    sparse_embeddings_list: List[SparseEmbedding] = list(
        model.embed(text, batch_size=6)
    ) 
    
    return sparse_embeddings_list


def get_tokens_and_weights(sparse_embedding, tokenizer):
    token_weight_dict = {}
    for i in range(len(sparse_embedding.indices)):
        token = tokenizer.decode([sparse_embedding.indices[i]])
        weight = sparse_embedding.values[i]
        token_weight_dict[token] = weight

    # Sort the dictionary by weights
    token_weight_dict = dict(sorted(token_weight_dict.items(), key=lambda item: item[1], reverse=True))
    return token_weight_dict



# Example usage
if __name__ == "__main__":

    sparse_embeddings_list = encode_text(["Maida flour, yogurt, oil, sugar",
                                        #   "Gram flour, ghee, sugar"
                                        ])
    
    print(f'Result: {sparse_embeddings_list}')
    
    for i in range(5):
    
        print(f"Token at index {sparse_embeddings_list[0].indices[i]} has weight {sparse_embeddings_list[0].values[i]}")
    
    index = 0
    
    # Test the function with the first SparseEmbedding
    print(json.dumps(get_tokens_and_weights(sparse_embeddings_list[index], tokenizer), indent=4))