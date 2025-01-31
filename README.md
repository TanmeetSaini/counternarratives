# Fact-based Counter Narrative Generation to Combat Hate Speech
This repository contains the implementation of **Fact-based Counter Narrative Generation to Combat Hate Speech**, a counter narrative generation method that enhances CNs by providing non-aggressive, fact-based narratives with relevant background knowledge from two distinct sources, including a web search module. Our method significantly outperforms baselines, achieving an average factuality score of 0.915, compared to 0.741 and 0.701 for competitive baselines, and performs well in human evaluations.![[figure.jpeg]]
Authors: [Brian Wilk](https://000brian.github.io/), [Homaira Huda Shomee](https://hhshomee.github.io/), [Suman Kalyan Maity](https://sites.mst.edu/smaity/), [Sourav Medya](https://souravmedya.github.io/)
## Installation and usage
### Step 1: Clone the repository
```
git clone https://github.com/000brian/counternarratives.git
cd counternarratives
```
### Step 2: Configure Python environment
Python 3.9.13 was used. An environment can be set up using `venv` or `conda`. Install the required Python packages using pip:
```
pip install -r requirements.txt
```
### Step 3: Prepare datasets
The [Newsroom](https://lil.nlp.cornell.edu/newsroom/]) dataset was used, which requires accepting the data licensing terms to use. Once downloaded, move `/release` from Newsroom to our `/newsroom`.
### Step 4: Build vectorstore
Run `/pipeline/embeddings.py` to build our vectorstore. We use MiniLM-L6-v2 to build a vectorstore from Newsroom. This runs locally, and is a compute and memory heavy task. Allow for ~70 gb of storage. See [LangChain's embedding models documentation](https://python.langchain.com/docs/integrations/text_embedding/) on changing the embedding function if you require something with a smaller disk footprint, or an embedding function that uses an API if running a embedding function locally will be inefficient.
### Step 5: Run pipeline
Use of the pipeline requires a [Tavily](https://tavily.com/) API key as well as an OpenAI API key. You can enter these keys at the first few lines of `/pipeline/pipeline.py`. Then, run it to generate counternarratives based off the ones provided in `mt_conan_no_dupes.csv`, sourced from Multitarget CONAN.

## Evaluation
We evaluate using traditional NLP metrics ([BLEU](https://www.nltk.org/_modules/nltk/translate/bleu_score.html), [BERTScore](https://github.com/Tiiiger/bert_score), [GRUEN](https://github.com/WanzhengZhu/GRUEN)), toxicity based on [Perspective API](https://www.perspectiveapi.com/), and [GPTScore](https://arxiv.org/abs/2302.04166) based metrics (factuality, persuasiveness, informativeness). Scripts for getting these metrics can be found in `/evaluation`.

## Example results
We provide a sample of generated counternarratives built by our pipeline. Results can be found in `/samples`.




