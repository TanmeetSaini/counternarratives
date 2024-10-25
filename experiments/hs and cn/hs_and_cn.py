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

# Load the CSV file
df = pd.read_csv('final_data.csv')

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

def calculate_bertscore(hate_speech, counterspeech):
    P, R, F1 = score([hate_speech], [counterspeech], lang="en", verbose=False)
    bert_precision, bert_recall, bert_f1 = P.mean().item(), R.mean().item(), F1.mean().item()
    return bert_f1

# Calculate metrics and progress
bleu_scores = []
rouge_scores = []
bertscore_values = []

for i, row in df.iterrows():
    hate_speech = row['HATE_SPEECH']
    counterspeech = row['COUNTER_NARRATIVE']
    
    bleu = compute_bleu2(hate_speech, counterspeech)
    rouge = compute_rouge_l(hate_speech, counterspeech)
    bertscore = calculate_bertscore(hate_speech, counterspeech)

    bleu_scores.append(bleu)
    rouge_scores.append(rouge)
    bertscore_values.append(bertscore)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} rows...")

# Add new columns to the DataFrame
df['bleu2'] = bleu_scores
df['rouge-l'] = rouge_scores
df['bertscore'] = bertscore_values

# Save the new DataFrame to a CSV file
output_file = 'experiment_462_mt_co.csv'
df.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")

# Print averages
averages = df[['bleu2', 'rouge-l', 'bertscore']].mean()
print("Averages:")
print(averages)
