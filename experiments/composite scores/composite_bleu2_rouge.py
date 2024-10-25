import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Load the CSV file
df = pd.read_csv('final_data.csv')
df = df.drop_duplicates(subset='hate_speech', keep='first')

# Step 1: Drop rows with no counter_narrative
df = df.dropna(subset=['gpt_4o_counterspeech'])

# Function to calculate average ROUGE-L
def calculate_average_rouge(row, df):
    target = row['TARGET']
    mtkg_counterspeech = row['gpt_4o_counterspeech']
    
    # Get all hate_speech entries with the same target
    hate_speeches = df[df['TARGET'] == target]['hate_speech'].tolist()
    
    if hate_speeches:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(mtkg_counterspeech, hs)['rougeL'].fmeasure for hs in hate_speeches]
        return sum(scores) / len(scores) if scores else None
    return None

# Function to calculate average BLEU-2
def calculate_average_bleu(row, df):
    target = row['TARGET']
    mtkg_counterspeech = row['gpt_4o_counterspeech']
    
    # Get all hate_speech entries with the same target
    hate_speeches = df[df['TARGET'] == target]['hate_speech'].tolist()
    
    if hate_speeches:
        scores = [sentence_bleu([hs.split()], mtkg_counterspeech.split(), weights=(0.5, 0.5)) for hs in hate_speeches]
        return sum(scores) / len(scores) if scores else None
    return None

# Apply the functions to create new columns
df['composite_rouge_l_ours_counterspeech'] = df.apply(calculate_average_rouge, axis=1, df=df)
df['composite_bleu_2_ours_counterspeech'] = df.apply(calculate_average_bleu, axis=1, df=df)

# Save the modified DataFrame to a new CSV file
df.to_csv('composite_ours_b2_r.csv', index=False)
