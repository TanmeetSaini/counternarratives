# Fact-based Counter Narrative Generation to Combat Hate Speech
Online hatred has become an increasingly pervasive issue, affecting individuals and communities across various digital platforms. To combat hate speech in such platforms, counter narratives (CNs) are regarded as an effective method. In recent years, there has been growing interest in using generative AI tools to construct CNs. However, most of the generative models produce generic responses to hate speech and can hallucinate, reducing their effectiveness. To address the above limitations, we propose a counter narrative generation method that enhances CNs by providing non-aggressive, fact-based narratives with relevant background knowledge from two distinct sources, including a web search module. Furthermore we conduct a comprehensive evaluation using multiple metrics, including LLM-based measures for persuasion, factuality, and informativeness, along with human and traditional NLP evaluations. Our method significantly outperforms baselines, achieving an average factuality score of 0.915, compared to 0.741 and 0.701 for competitive baselines, and performs well in human evaluations.

## Counter Narrative Generation
Our LLM of choice is GPT-4o. Code for generating the counterspeech can be found in `/pipeline`.

## Experiments and Evaluation
We employed traditional NLP metrics as well as LLM-based measures to evaluate our results. Code for evaluation and experiments can be found in `/experiments`.



