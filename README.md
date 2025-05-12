# Semantic Entropy for Detecting Bias in LLM Responses

View the [final paper](https://github.com/diya-vinod/LLMProject/blob/main/LLMProjectFinalPaper.pdf).

## Abstract

This study investigates whether semantic entropy can serve as an effective metric
for detecting bias in large language models (LLMs). Building upon prior research
that established semantic entropy as a method for identifying hallucinations, we
hypothesized that prompts likely to elicit biased responses would generate higher
semantic entropy values. Using the DELPHI dataset, we selected five controversial
questions related to social biases and five non-controversial factual questions to test
across three state-of-the-art LLMs: GPT-4o, DeepSeek V3, and Claude 3.7 Sonnet.
For each question, we calculated naive entropy, semantic entropy, and did a comparative analysis. Contrary to our hypothesis, we found no significant correlation between controversial prompts and semantic entropy values. Instead, open-ended but
non-controversial questions like "Where are the best places to eat in Venice, Italy?"
consistently produced higher semantic entropy across all models. We observed that
Claude 3.7 Sonnet displayed notably higher semantic entropy values overall, suggesting that model architecture and response style may influence entropy measurements
more significantly than content controversiality. These findings indicate that while
semantic entropy remains valuable for detecting hallucinations, it may not effectively
signal biased reasoning in modern LLMs, which typically maintain neutrality when
addressing controversial topics.

## Code

The datasets least_controversial_questions.csv and controversial_questions.csv contain the questions we used for our project. These questions come from the Quora Question Pairs Dataset and are assigned a controversial score in the DELPHI dataset [4]. OpenAIClientSemanticEntropy.py contains the code to calculate the semantic entropy of responses using the OpenAI API. AnthropicClientSemanticEntropy.py contains the code to calculate the semantic entropy of responses using the Anthropic API. The code follows closely the work done by Karrtik Iyer and Parag Mahajani [9]. Data.csv contains the final semantic entropy values for the prompts. 

## References

[1] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in
large language models using semantic entropy. Nature, 630, 625–630. https:<!--This is a comment-->//doi.
org/10.1038/s41586-024-07421-0

[2] IBM. (2023, September 1). AI hallucinations. Ibm.com. https:<!--This is a comment-->//ww<!--This is a comment-->w.<!--This is a comment-->ibm.<!--This is a comment-->com/
think/topics/ai-hallucinations

[3] Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., & Gal, Y. (2024). Semantic
Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv. https:
//arxiv.org/abs/2406.15927

[4] Sun, D. Q., Abzaliev, A., Kotek, H., Xiu, Z., Klein, C., & Williams, J. D. (2023).
DELPHI: Data for Evaluating LLMs’ Performance in Handling Controversial Issues.
arXiv preprint arXiv:2310.18130. https<!--This is a comment-->://arxiv<!--This is a comment-->.org<!--This is a comment-->/abs/2310.18130

[5] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024, June 19). Detecting hal-
lucinations in large language models using semantic entropy. Oxford Applied and
Theoretical Machine Learning Group. https<!--This is a comment-->://<!--This is a comment-->oatml.cs.ox.ac.uk/blog/2024/06/19/
detecting_hallucinations_2024.html

[6] Attanasio, G., Nozza, D., Hovy, D., Baralis, E. (2022). Entropy-based Attention
Regularization Frees Unintended Bias Mitigation from Lists. Findings of the Associ-
ation for Computational Linguistics: ACL 2022. htt<!--This is a comment-->ps://doi.org<!--This is a comment-->/10.18653/<!--This is a comment-->v1/2022.
findings-acl.88

[7] Ganguli, D., Brundage, M., Clark, J., Askell, A., Krueger, G., Multon, P., ... &
Bowman, D. (2022). Red teaming language models with language models. arXiv.
https:<!--This is a comment-->//arxiv.<!--This is a comment-->org/abs/2202.03286

[8] Zhou, D., Schärli, N., Hou, L., Wei, J., & Le, Q. V. (2023). Least-to-most prompting
enables complex reasoning in large language models. arXiv. https:/<!--This is a comment-->/arxiv.org/abs/<!--This is a comment-->
2205.10625

[9] Iyer, K., & Mahajani, P. (2025, March 7).Evaluating LLMs using semantic en-
tropy. Thoughtworks. https:<!--This is a comment-->//ww<!--This is a comment-->w.<!--This is a comment-->thoughtworks.com<!--This is a comment-->/insights/blog/generative-ai/
Evaluating-LLM-using-semantic-entropy 

[10] Sun, M., Yin, Y., Xu, Z., Kolter, J. Z., & Liu, Z. (2025, Feb 17). Idiosyncrasies in
large language models. arXiv. https:<!--This is a comment-->//arxiv.org/abs<!--This is a comment-->/2502.12150
