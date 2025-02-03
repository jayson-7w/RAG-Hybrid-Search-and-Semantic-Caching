import os
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

def generate_llm_response(context, question):
    # Define a prompt template
    prompt_template = """
    ### [INST]
    You are a helpful assistant who provides cooking advice.

    Use the recipe information below as relevant context to help answer the user question.

    If there is no recipe in the context, answer the user question based on your general knowledge.

    ### CONTEXT:
    {context}

    ### QUESTION:
    {question}

    ### RESPONSE:
    
    [/INST]
    """
    
    # Model API endpoint (replace with the model ID)
    model_id = "microsoft/Phi-3.5-mini-instruct"

    client = InferenceClient(api_key=HUGGING_FACE_TOKEN)

    # Format the prompt using the template and provided context/question
    prompt = prompt_template.format(context=context, question=question)

    # Initialize variables to track tokens
    response_text = ""

    # Generate the response using the formatted prompt
    for message in client.chat_completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        stream=True,
    ):
        response_chunk = message.choices[0].delta.content
        response_text += response_chunk

        # Print each chunk immediately
        print(response_chunk, end="", flush=True)  # Add flush=True for immediate output

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Count the tokens using the tokenizer
    total_tokens = len(tokenizer.encode(response_text))

    return response_text, total_tokens

# # Example usage
# if __name__ == "__main__":
#     response_text, total_tokens = generate_llm_response(question="Can you provide me an indian recipe?", context=None)
#     print(response_text)
#     print(total_tokens)