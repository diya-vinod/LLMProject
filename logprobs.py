from openai import OpenAI
from math import exp
import numpy as np
#from IPython.display import display, HTML
import os
import getpass

# Prompt user to enter their API key
api_key = input("Enter your OpenAI API key: ")
class OpenAIClient:
	def __init__(self, api_key):
		self.client = OpenAI(api_key=api_key)
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

# get multiple answers from the LLM
def get_multiple_answers(client_instance, question, num_answers=10):
    """
    Get multiple answers from the LLM for a given question.
    """
    responses = []
    for _ in range(num_answers):
        response = client_instance.client.chat.completions.create(
            model="gpt-4o",  # Use the appropriate model
            messages=[
                {"role": "user", "content": f"Answer this question briefly: {question}"}
            ],
            temperature=1.0,  # Low temperature for confident answer
            max_tokens=100,
            top_p=0.9
        )
        responses.append(response.choices[0].message.content.strip())
    return responses

# Count frequency of each unique response
def count_frequencies(responses):
    """
    Count the frequency of each unique response.
    """
    frequencies = {}
    for response in responses:
        if response in frequencies:
            frequencies[response] += 1
        else:
            frequencies[response] = 1
    return frequencies_to_probabilities(frequencies)  # Return probabilities

# Convert frequencies to probabilities
def frequencies_to_probabilities(frequencies):
    """
    Convert frequencies to probabilities.
    """
    total = sum(frequencies.values())
    probabilities = {k: v / total for k, v in frequencies.items()}
    return probabilities

# Calculate entropy using standard formula
def calculate_entropy(probabilities):
    """
    Calculate entropy using the standard formula.
    """
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    return entropy

def cluster_answers(client_instance, responses, question):
    """
    Cluster semantically similar answers using bidirectional entailment.
    Returns a list of clusters, where each cluster is a list of similar responses.
    """
    used = set()
    clusters = []

    for i, ans1 in enumerate(responses):
        if i in used:
            continue
        cluster = [ans1]
        used.add(i)

        for j in range(i + 1, len(responses)):
            if j in used:
                continue
            ans2 = responses[j]

            # Check if ans1 semantically entails ans2 and vice versa
            entail1 = check_entailment(client_instance, ans1, ans2, question)
            entail2 = check_entailment(client_instance, ans2, ans1, question)

            if entail1 and entail2:
                cluster.append(ans2)
                used.add(j)

        clusters.append(cluster)

    return clusters

def check_entailment(client_instance, text1, text2, question):
    """
    Use the LLM to check if text1 semantically entails text2 in the context of the question.
    """
    try:
        response = client_instance.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content":
                    f"In the context of this question: '{question}', does the following answer 1 semantically entail answer 2?\n\n"
                    f"Answer 1: {text1}\n"
                    f"Answer 2: {text2}\n\n"
                    "Respond with only 'yes' or 'no'."}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == "yes"
    except Exception as e:
        print(f"Entailment check failed: {e}")
        return False

def calculate_semantic_entropy(clusters, probabilities):
    """
    Calculate semantic entropy based on clusters and their summed probabilities.
    
    Args:
        clusters: List of lists (each inner list is a cluster of equivalent answers)
        probabilities: Dictionary mapping answer text to its probability

    Returns:
        semantic_entropy: A float representing the semantic entropy
    """
    cluster_probs = []

    for cluster in clusters:
        cluster_prob = sum(probabilities.get(answer, 0) for answer in cluster)
        if cluster_prob > 0:
            cluster_probs.append(cluster_prob)

    semantic_entropy = -sum(p * np.log2(p) for p in cluster_probs if p > 0)
    return semantic_entropy


question = input("Enter a question to ask the LLM: ")
print(f"You entered: {question}")

# Call the function and print the result
answer_text = get_best_answer(client_instance, question)
print(f"Best Answer: {answer_text}")

# Get multiple answers
mult_answer_text = get_multiple_answers(client_instance, question, num_answers=5)
print(f"Multiple Answers: {mult_answer_text}")

# Count frequencies and convert to probabilities
probabilities = count_frequencies(mult_answer_text)

# calculate entropy
entropy = calculate_entropy(probabilities)
print(f"Entropy: {entropy}")

clusters = cluster_answers(client_instance, mult_answer_text, question)
for idx, cluster in enumerate(clusters):
    print(f"Cluster {idx + 1}:")
    for ans in cluster:
        print(f" - {ans}")
    print()

# Calculate semantic entropy
semantic_entropy = calculate_semantic_entropy(clusters, probabilities)
print(f"\nSemantic Entropy: {semantic_entropy:.4f}")
