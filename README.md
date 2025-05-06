# Semantic Entropy for Detecting Bias in LLM Responses
This study investigates whether semantic entropy can serve as an effective metric
for detecting bias in large language models (LLMs). Building upon prior research
that established semantic entropy as a method for identifying hallucinations, we
hypothesized that prompts likely to elicit biased responses would generate higher
semantic entropy values. Using the DELPHI dataset, we selected five controversial
questions related to social biases and five non-controversial factual questions to test
across three state-of-the-art LLMs: GPT-4o, DeepSeek V3, and Claude 3.7 Sonnet.
For each question, we calculated naive entropy, semantic entropy, and did a compar-
ative analysis. Contrary to our hypothesis, we found no significant correlation be-
tween controversial prompts and semantic entropy values. Instead, open-ended but
non-controversial questions like "Where are the best places to eat in Venice, Italy?"
consistently produced higher semantic entropy across all models. We observed that
Claude 3.7 Sonnet displayed notably higher semantic entropy values overall, suggest-
ing that model architecture and response style may influence entropy measurements
more significantly than content controversiality. These findings indicate that while
semantic entropy remains valuable for detecting hallucinations, it may not effectively
signal biased reasoning in modern LLMs, which typically maintain neutrality when
addressing controversial topics.
