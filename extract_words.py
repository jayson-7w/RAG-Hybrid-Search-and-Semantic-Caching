from dotenv import load_dotenv
import re

load_dotenv()

# Function to clean and extract words from the query
def extract_words_from_query(query):
    # Use regex to remove special characters and split by spaces
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    words = query.lower().split()
    return words
