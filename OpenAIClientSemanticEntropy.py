from typing import List
import numpy as np
from openai import OpenAI


# Prompt user to enter their API key
api_key = input("Enter your OpenAI API key: ")

class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = "deepseek-chat"
        

# Instantiate the class with the API key
client_instance = OpenAIClient(api_key=api_key)

def get_best_answer(client_instance, question):
    """
    Get the best answer from the LLM for a given question.
    """
    response = client_instance.client.chat.completions.create(
        model=client_instance.model,
        messages=[
            {"role": "user", "content": f"Answer this question briefly: {question}"}
        ],
        temperature=0.1,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def get_multiple_answers(client_instance, question, num_answers=10):
    """
    Get multiple answers from the LLM for a given question.
    """
    responses = []
    for _ in range(num_answers):
        response = client_instance.client.chat.completions.create(
            model=client_instance.model,
            messages=[
                {"role": "user", "content": f"Answer this question briefly: {question}"}
            ],
            temperature=1.0,
            top_p=0.9
        )
        responses.append(response.choices[0].message.content.strip())
    return responses

def calculate_answer_frequencies(answers):
    """
    Calculate the frequency of each answer.
    """
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    total = len(answers)
    probs = [(a, cnt/total) for a, cnt in counts.items()]
   
    return probs

def calculate_naive_entropy(answer_freq_tuple):
    """
    Calculate naive entropy based on answer frequencies.
    """
    raw_probs = [p for _, p in answer_freq_tuple]
    
    # sum of probs should be 1.0
    total = sum(raw_probs)
    if total == 0:
        return 0.0

    # normalize 
    normalized_probs = [p / total for p in raw_probs]

    # calculate individual entropies
    entropy_contributions = []

    for i, (answer_freq_tuple, _) in enumerate(answer_freq_tuple):
        p = normalized_probs[i]
        if p > 0:
            entropy_contributions.append(p * np.log(p))
        
    # calculate total entropy
    total_entropy = -sum(entropy_contributions)
    return float(total_entropy)

def _is_yes(response_text: str) -> bool:
    """
    Check if the response text indicates a "yes" answer.
    """
    return response_text.strip().lower().startswith("yes")

def _check_semantic_equivalence(client_instance, a, b, question):
    """
    Check if two answers are semantically equivalent using the LLM.
    """
    response = client_instance.client.chat.completions.create(
        model=client_instance.model,
        messages=[
            {"role": "user", "content": f"In the context of this question: {question}. Does the following answer 1 semantically entail answer 2?\n\nAnswer 1: {a}\nAnswer 2: {b}\n\nRespond with only 'yes' or 'no'."}
        ],
        temperature=0.0,
        
    )
    response2 = client_instance.client.chat.completions.create(
        model=client_instance.model,
        messages=[
            {"role": "user", "content": f"In the context of this question: {question}. Does the following answer 2 semantically entail answer 1?\n\nAnswer 1: {b}\nAnswer 2: {a}\n\nRespond with only 'yes' or 'no'."}
        ],
        temperature=0.0,
    )
    r1 = response.choices[0].message.content
    r2 = response2.choices[0].message.content

    print(f"Response 1: {r1}")
    print(f"Response 2: {r2}")
    return _is_yes(r1) and _is_yes(r2)


def calculate_semantic_entropy(client_instance, answers, question):
    """
    Calculate semantic entropy based on the answers.
    """
    total_probability = sum(p for answer, p in answers)
    normalized_probs = [(text, p / total_probability) for text, p in answers]

    clusters = {}
    used = set()

    for i, (a, p) in enumerate(normalized_probs):
        if i in used:
            continue
        cluster_texts = [a]
        cluster_prob = p
        used.add(i)
        for j, (b, p2) in enumerate(normalized_probs[i+1:], start=i+1):
            if j in used:
                continue
            if _check_semantic_equivalence(client_instance, a, b, question):
                cluster_texts.append(b)
                cluster_prob += p2
                used.add(j)
        
        cluster_id = len(clusters)
        clusters[cluster_id] = (cluster_texts, cluster_prob)

    cluster_probs = [prob for _, (_, prob) in clusters.items()]
    semantic_contributions = [-p * np.log(p) for p in cluster_probs]
    semantic_entropy = sum(semantic_contributions)

    print("\nFinal semantic entropy calculation:")
    for i, (prob, contribution) in enumerate(zip(cluster_probs, semantic_contributions)):
        print(
            f"Cluster {i + 1}: {prob:.3f} * log({prob:.3f}) = {contribution:.3f}"
        )
    print(f"\nTotal semantic entropy: {semantic_entropy:.3f}")
    return float(semantic_entropy)

def main():
    question = input("Enter a question to ask the LLM: ")
    print(f"You entered: {question}\n")

    # Best single answer
    best = get_best_answer(client_instance, question)
    print(f"Best Answer: {best}\n")

    # Multiple sampled answers
    samples = get_multiple_answers(client_instance, question, num_answers=10)
    
    # compute frequencies
    answer_freq_tuple = calculate_answer_frequencies(samples)
    print("Answer Frequencies:")
    for ans, freq in answer_freq_tuple:
        print(f"{ans}: freq = {freq:.4f}")
        print()
        print()
    naive_H = calculate_naive_entropy(answer_freq_tuple)
    semantic_H = calculate_semantic_entropy(client_instance, answer_freq_tuple, question)

    print(f"Naive Entropy: {naive_H:.4f}")
    print(f"Semantic Entropy: {semantic_H:.4f}")


if __name__ == "__main__":
    main()

