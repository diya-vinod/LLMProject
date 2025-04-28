from openai import OpenAI
from math import exp
import numpy as np
from IPython.display import display, HTML
import os
import getpass

# Prompt user to enter their API key
api_key = getpass.getpass("Enter your OpenAI API key: ")
class OpenAIClient:
	def __init__(self, api_key):
		self.client = OpenAI(api_key=api_key, model="gpt-4o")
		self.model = "gpt-4o"
		self.temperature = 0.1

# Instantiate the class with the API key
client_instance = OpenAIClient(api_key=api_key)

def get_best_answer(client_instance, question):
    """
    Get the best answer from the LLM for a given question.
    """
    response = client_instance.client.chat.completions.create(
        model="gpt-4o",  # Use the appropriate model
        messages=[
            {"role": "user", "content": f"Answer this question briefly: {question}"}
        ],
        temperature=0.1,  # Low temperature for confident answer
        max_tokens=100
    )

    # Extract the actual text of the answer
    return response.choices[0].message.content.strip()


question = input("Enter a question to ask the LLM: ")
print(f"You entered: {question}")

# Call the function and print the result
answer_text = get_best_answer(client_instance, question)
print(f"Best Answer: {answer_text}")
