import pandas as pd
import nltk
import time
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
import re
from bert_score import score  # Import BERTScore
import requests
import time
import logging

# Download NLTK packages
nltk.download('punkt')
nltk.download('punkt_tab')

# Perspective API setup
API_KEY = 'AIzaSyB8p-KSBqNCBPRdSXbW-SS2Ohr_fU4I8cY'
url = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=' + API_KEY

# Initialize the Rouge scorer
rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Function to compute BLEU-2 score
def compute_bleu2(reference, hypothesis):
    ref_tokens = nltk.word_tokenize(reference)
    hyp_tokens = nltk.word_tokenize(hypothesis)
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1, weights=(0.5, 0.5))

# Function to compute ROUGE-L score
def compute_rouge_l(reference, hypothesis):
    score = rouge_scorer.score(reference, hypothesis)
    return score['rougeL'].fmeasure

# Function to compute repetition rate
def compute_repetition_rate(text):
    words = nltk.word_tokenize(text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    if total_words == 0:
        return 0
    return (total_words - unique_words) / total_words

# Function to get toxicity score from Perspective API
def get_toxicity(text):
    data = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}}
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        response_json = response.json()
        score = response_json['attributeScores']['TOXICITY']['summaryScore']['value']
    except (requests.exceptions.RequestException, KeyError) as e:
        logging.error(f"Error processing text: {text}. Error: {e}")
        score = None
    return score

# Load CSV
df = pd.read_csv('mt_conan_test_delete_me.csv')

# List of counterspeech columns, add more if needed
llms = ['gpt_4o_counterspeech']

# Add columns to store new metrics
for llm in llms:
    df[f'bleu2_{llm}'] = 0.0
    df[f'rouge_l_{llm}'] = 0.0
    df[f'composite_bleu2_{llm}'] = 0.0
    df[f'composite_rouge_l_{llm}'] = 0.0
    df[f'rr_{llm}'] = 0.0
    df[f'toxicity_hate_speech'] = 0.0
    df[f'toxicity_{llm}'] = 0.0
    df[f'bertscore_precision_{llm}'] = 0.0
    df[f'bertscore_recall_{llm}'] = 0.0
    df[f'bertscore_f1_{llm}'] = 0.0

# Process each row
for idx, row in tqdm(df.iterrows(), total=len(df)):
    for llm in llms:
        # Compute BLEU-2 and ROUGE-L
        bleu2_score = compute_bleu2(row['counter_narrative'], row[llm])
        rouge_l_score = compute_rouge_l(row['counter_narrative'], row[llm])

        # Composite BLEU-2 and ROUGE-L for the same TARGET
        target_rows = df[df['TARGET'] == row['TARGET']]
        composite_bleu2 = sum(compute_bleu2(row['counter_narrative'], x[llm]) for _, x in target_rows.iterrows()) / len(target_rows)
        composite_rouge_l = sum(compute_rouge_l(row['counter_narrative'], x[llm]) for _, x in target_rows.iterrows()) / len(target_rows)

        # Compute repetition rate
        rr_score = compute_repetition_rate(row[llm])

        # Get toxicity scores
        toxicity_hate_speech = get_toxicity(row['hate_speech'])
        time.sleep(1.5)  # To respect API rate limits
        toxicity_llm = get_toxicity(row[llm])

        # Compute BERTScore
        P, R, F1 = score([row[llm]], [row['counter_narrative']], lang="en", verbose=False)
        bert_precision, bert_recall, bert_f1 = P.mean().item(), R.mean().item(), F1.mean().item()

        # Update dataframe with scores
        df.at[idx, f'bleu2_{llm}'] = bleu2_score
        df.at[idx, f'rouge_l_{llm}'] = rouge_l_score
        df.at[idx, f'composite_bleu2_{llm}'] = composite_bleu2
        df.at[idx, f'composite_rouge_l_{llm}'] = composite_rouge_l
        df.at[idx, f'rr_{llm}'] = rr_score
        df.at[idx, 'toxicity_hate_speech'] = toxicity_hate_speech
        df.at[idx, f'toxicity_{llm}'] = toxicity_llm
        df.at[idx, f'bertscore_precision_{llm}'] = bert_precision
        df.at[idx, f'bertscore_recall_{llm}'] = bert_recall
        df.at[idx, f'bertscore_f1_{llm}'] = bert_f1

# Save the updated dataframe to a new CSV file
df.to_csv('updated_output.csv', index=False)