import re
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from bert_score import score as bert_score
from tqdm import tqdm

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute BERTScore
def compute_bertscore(doc, summary):
    P, R, F1 = bert_score([summary], [doc], model_type='bert-base-uncased', lang='en')
    return F1.mean().item()

# Function to extract 'page_content' and summary from the CSV and calculate BERTScore
def extract_and_score(input_csv, output_csv):
    df = pd.read_csv(input_csv)  # Limit to first 100 entries for faster processing
    bert_scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        query_document_pairs = row['query_document_pairs']
        query_summarized_document_pairs = row['query_summarized_document_pairs']
        
        # Debugging output to check the raw input
        print(f"Row {idx}: Raw query_document_pairs: {query_document_pairs[:200]}...")
        print(f"Row {idx}: Raw query_summarized_document_pairs: {query_summarized_document_pairs[:200]}...")
        
        # Adjusted regex to handle potential issues with special characters or escaped quotes
        documents = re.findall(r"page_content=['\"](.*?)['\"]\s*\}", query_document_pairs)
        summaries = re.findall(r"'document':\s*['\"](.*?)['\"]", query_summarized_document_pairs)
        
        # Retry extraction with a different approach if initial regex fails
        if len(documents) != 6:
            documents = re.findall(r"page_content=['\"](.*?)['\"]", query_document_pairs)
        if len(summaries) != 6:
            summaries = re.findall(r"'document':\s*['\"](.*?)['\"]", query_summarized_document_pairs)
        
        # Debugging output to check extraction
        print(f"Row {idx}: Extracted {len(documents)} documents and {len(summaries)} summaries.")
        
        # Ensure that we have pairs of 6 documents and summaries
        if len(documents) == 6 and len(summaries) == 6:
            row_scores = []
            for doc, summary in zip(documents, summaries):
                score = compute_bertscore(doc, summary)
                row_scores.append(score)
            bert_scores.append(row_scores)
        else:
            print(f"Warning: Unexpected number of documents or summaries in row {idx}.")
            bert_scores.append([None] * 6)  # Handle rows with unexpected formats

    # Save scores to a new CSV
    score_df = pd.DataFrame(bert_scores, columns=[f'score_{i+1}' for i in range(6)])
    score_df.to_csv(output_csv, index=False)

# Example usage
input_csv = 'final_data.csv'
output_csv = 'bertscores_long_output.csv'
extract_and_score(input_csv, output_csv)