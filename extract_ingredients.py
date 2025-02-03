import os
from dotenv import load_dotenv

load_dotenv()

# Set the path to the CSV file
DATAPATH = os.environ["DATAPATH"]

# Function to create a unique ingredient list from the DataFrame
def create_ingredient_list(df):
    unique_ingredients = set()  # Use a set to store unique ingredients
    
    for ingredients in df['ingredients']:
        # Split ingredients by commas and strip extra spaces
        ingredients_list = [ingredient.strip().lower() for ingredient in ingredients.split(',')]
        
        # Add the ingredients to the set (only unique values will be stored)
        unique_ingredients.update(ingredients_list)
    
    return sorted(unique_ingredients)  # Return a sorted list of unique ingredients

# if __name__ == "__main__":
#     # Load the dataset
#     df = pd.read_csv(DATAPATH)
    
#     # Create the unique ingredients list
#     ingredient_list = create_ingredient_list(df)
    
#     print(ingredient_list)
    
#     # Save the unique ingredients list to a file (optional)
#     with open("unique_ingredients.txt", "w") as f:
#         for ingredient in ingredient_list:
#             f.write(f"{ingredient}\n")
    
#     print("Unique ingredients list created and saved to 'unique_ingredients.txt'.")
