import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the LLM with Gemini-1.5-Flash using your Google API Key
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    api_key="AIzaSyARfPLU7dJ8x3OjwCqkB4qIAFyVjVKlXF8"  # Make sure to set your API key
)

# Ask the model a question
response = llm.invoke("Hello, how are you?")  # Use _chat for sending queries to the model

# Print the response
print(response)  # This will print the response from the model
