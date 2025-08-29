from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os 

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key and project ID from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

def configuration(api_key):
    """
    Initialize and return the Google Generative AI model object.
    """
    # Pass the project_id inside model_kwargs, not as a direct parameter
    model_kwargs = {
        "project_id": os.getenv("secrets-416414")
    }

    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model = "gemini-1.5-flash",
        model_kwargs=model_kwargs  # Include project_id here
    )
    return llm



#  TO use it 


# {{{
# app.py (or any other file)

# from config import configuration  # Import the function from config.py

# # Fetch the API key from environment variables
# import os
# api_key = os.getenv("GOOGLE_API_KEY")

# # Call the configuration function to get the model
# llm = configuration(api_key)

# # Use the llm object to invoke queries
# response = llm.invoke("What is the capital of France?")
# print(response)
# }}}